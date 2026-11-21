"""Extract plain text from uploaded documents.

Supported: PDF, DOCX, XLSX/XLSM, PPTX, ODT/ODS/ODP, CSV/TSV, TXT/MD/LOG, JSON, XML, HTML.
Legacy binary Office formats (.doc/.xls/.ppt) are not supported - the caller gets a clear error.

Every extractor works on a file path (uploads are streamed to disk, never held in memory)
and returns (text, info) where info describes the document (pages, sheets, ...).
"""
import csv
import difflib
import html.parser
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, Tuple

SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx", ".odt", ".ods", ".odp",
    ".csv", ".tsv", ".txt", ".md", ".log", ".json", ".xml", ".html", ".htm",
)
LEGACY_EXTENSIONS = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx", ".rtf": ".docx or .txt"}


class ExtractionError(Exception):
    """A problem with the document that the user should be told about."""


# ── helpers ──────────────────────────────────────────────────────────────

def _clean(e: Exception, path: Path) -> str:
    """Error text without the server-side temp path."""
    return str(e).replace(str(path), "the file")


def _decode(raw: bytes) -> str:
    """Decode text bytes: BOM-aware UTF-8/UTF-16, then cp1252, never fails."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _tidy(text: str) -> str:
    """Normalise whitespace without destroying paragraph structure."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class _TextOnly(html.parser.HTMLParser):
    _SKIP = {"script", "style", "head", "noscript"}
    _BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


# ── PDF line reflow ──────────────────────────────────────────────────────
# PDFs store positioned words, not paragraphs, so extracted text has hard line breaks in the middle of sentences.
# Justified text is worse: some PDFs come out as "word\n\nword\n\nword". Translators treat every line break as a
# sentence boundary, which produces word-by-word nonsense, so lines are rebuilt into real paragraphs first.

_BULLETS = "●•▪■◦◆‣▶"
_BULLET_SPLIT = re.compile(r"\s*([" + _BULLETS + r"])\s*")
_LIST_ITEM = re.compile(r"^(\d{1,3}[.)]|[-–—*])\s")
_TERMINAL = (".", "!", "?", ":", ";", "。", "।", "！", "？")


def _reflow(text: str) -> str:
    text = re.sub(r"[ \t\u00a0]+", " ", text)                  # layout mode pads justified lines with many spaces
    lines = [l.strip() for l in text.split("\n")]
    filled = [l for l in lines if l]
    if not filled:
        return ""

    blank_ratio = 1 - len(filled) / len(lines)
    avg_words = sum(len(l.split()) for l in filled) / len(filled)
    lengths = sorted(len(l) for l in filled)
    width = lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))]   # typical full line length on this page

    def join(parts):
        out = ""
        for part in parts:
            if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and part[:1].islower():
                out = out[:-1] + part                          # "transfor-" + "mation": undo hyphenation
            else:
                out = f"{out} {part}" if out else part
        return out

    if blank_ratio > 0.3 and avg_words < 3:
        # fragmented page (a word or two per line, blank lines between): line breaks carry no meaning at all
        paragraphs = [join(filled)]
    else:
        paragraphs, cur = [], []
        for line in lines:
            if not line:                                       # real blank line = paragraph break
                if cur:
                    paragraphs.append(join(cur))
                    cur = []
                continue
            if cur:
                new_item = line[0] in _BULLETS or bool(_LIST_ITEM.match(line))
                short_end = cur[-1].endswith(_TERMINAL) and len(cur[-1]) < 0.75 * width
                # two short lines in a row are not a flowing paragraph (labels, addresses, headings)
                both_short = len(cur[-1]) < 0.6 * width and len(line) < 0.6 * width
                if new_item or short_end or both_short:        # list item, end of paragraph, or label-like lines
                    paragraphs.append(join(cur))
                    cur = []
            cur.append(line)
        if cur:
            paragraphs.append(join(cur))

    # bullets that share a line ("... GIT ● Architect ...") become separate items
    out = []
    for para in paragraphs:
        para = _BULLET_SPLIT.sub(lambda m: "\n" + m.group(1) + " ", para)
        out.extend(x.strip() for x in para.split("\n") if x.strip())
    return "\n".join(out)


# ── extractors ───────────────────────────────────────────────────────────

LAYOUT_MAX_PAGES = 300        # layout mode is slower; very long PDFs use the plain mode


def _page_text(page, use_layout: bool) -> str:
    """Plain text of one page. Layout mode (which keeps real line breaks and paragraph gaps) is used only when it
    contains the same words in the same order as the plain mode, so multi-column pages are never interleaved."""
    plain = page.extract_text() or ""
    if not use_layout or not plain.strip():
        return plain
    try:
        layout = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        return plain
    pw, lw = plain.split(), layout.split()
    if pw == lw:
        return layout
    if not lw or abs(len(pw) - len(lw)) > max(5, 0.05 * len(pw)):
        return plain
    if len(pw) <= 4000 and difflib.SequenceMatcher(None, pw, lw, autojunk=False).ratio() >= 0.97:
        return layout
    return plain


def _pdf(path: Path) -> Tuple[str, Dict]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")          # many PDFs are "encrypted" with an empty password
            except Exception:
                ok = 0
            if not ok:
                raise ExtractionError("This PDF is password-protected. Remove the password and upload it again.")
        pages = []
        use_layout = len(reader.pages) <= LAYOUT_MAX_PAGES
        for page in reader.pages:
            try:
                pages.append(_page_text(page, use_layout).strip())
            except Exception:
                pages.append("")                 # one bad page should not sink the whole document
    except PdfReadError as e:
        raise ExtractionError(f"This PDF could not be read ({_clean(e, path)}).")
    pages = [_reflow(p) for p in pages]
    text = ""
    for p in (p for p in pages if p):
        if text and not text.endswith(_TERMINAL) and p[:1].islower():
            text += " " + p                                    # sentence continues on the next page
        else:
            text += ("\n" if text else "") + p
    if not text.strip():
        raise ExtractionError(
            "No selectable text was found in this PDF. It is probably a scan (images); "
            "OCR is not supported - upload a text-based PDF or a Word/text version."
        )
    empty = sum(1 for p in pages if not p)
    return text, {"pages": len(pages), "empty_pages": empty}


def _docx(path: Path) -> Tuple[str, Dict]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    try:
        document = docx.Document(str(path))
    except Exception as e:
        raise ExtractionError(f"This Word file could not be read ({_clean(e, path)}).")
    lines = []
    for child in document.element.body.iterchildren():       # body order: paragraphs and tables interleaved
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append(Paragraph(child, document).text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                seen, cells = set(), []
                for cell in row.cells:
                    if id(cell._tc) in seen:                  # merged cells repeat
                        continue
                    seen.add(id(cell._tc))
                    if cell.text.strip():
                        cells.append(cell.text.strip().replace("\n", " "))
                if cells:
                    lines.append(" | ".join(cells))
    for section in document.sections:                          # headers/footers often hold names and dates
        for part in (section.header, section.footer):
            for p in part.paragraphs:
                if p.text.strip():
                    lines.append(p.text)
    return "\n".join(lines), {"paragraphs": len(document.paragraphs), "tables": len(document.tables)}


def _xlsx(path: Path) -> Tuple[str, Dict]:
    import openpyxl
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as e:
        raise ExtractionError(f"This spreadsheet could not be read ({_clean(e, path)}).")
    out, rows_total = [], 0
    try:
        for ws in wb.worksheets:
            out.append(f"[Sheet: {ws.title}]")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip() != ""]
                if cells:
                    out.append(" | ".join(cells))
                    rows_total += 1
    finally:
        wb.close()
    return "\n".join(out), {"sheets": len(wb.sheetnames), "rows": rows_total}


def _pptx(path: Path) -> Tuple[str, Dict]:
    from pptx import Presentation
    try:
        prs = Presentation(str(path))
    except Exception as e:
        raise ExtractionError(f"This presentation could not be read ({_clean(e, path)}).")

    def shape_lines(shapes):
        for shape in shapes:
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                yield shape.text_frame.text
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        yield " | ".join(cells)
            if getattr(shape, "shapes", None):                  # grouped shapes
                yield from shape_lines(shape.shapes)

    out = []
    for i, slide in enumerate(prs.slides, 1):
        out.append(f"[Slide {i}]")
        out.extend(shape_lines(slide.shapes))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                out.append(notes)
    return "\n".join(out), {"slides": len(prs.slides)}


def _odf(path: Path) -> Tuple[str, Dict]:
    try:
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("content.xml"))
    except Exception as e:
        raise ExtractionError(f"This OpenDocument file could not be read ({_clean(e, path)}).")
    lines = []
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] in ("p", "h"):
            t = "".join(el.itertext()).strip()
            if t:
                lines.append(t)
    return "\n".join(lines), {}


def _delimited(path: Path) -> Tuple[str, Dict]:
    raw = _decode(path.read_bytes())
    delim = "\t" if path.suffix.lower() == ".tsv" else None
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|") if delim is None else csv.excel_tab
    except csv.Error:
        dialect = csv.excel
    rows = 0
    out = []
    for row in csv.reader(io.StringIO(raw), dialect):
        cells = [c.strip() for c in row if c.strip()]
        if cells:
            out.append(" | ".join(cells))
            rows += 1
    return "\n".join(out), {"rows": rows}


def _plain(path: Path) -> Tuple[str, Dict]:
    return _decode(path.read_bytes()), {}


def _markup(path: Path) -> Tuple[str, Dict]:
    raw = _decode(path.read_bytes())
    if path.suffix.lower() == ".xml":
        try:
            return "\n".join(t.strip() for t in ET.fromstring(raw).itertext() if t.strip()), {}
        except ET.ParseError:
            return raw, {}                                      # not well-formed: keep the raw text
    parser = _TextOnly()
    parser.feed(raw)
    return "".join(parser.parts), {}


_EXTRACTORS: Dict[str, Callable[[Path], Tuple[str, Dict]]] = {
    ".pdf": _pdf, ".docx": _docx, ".xlsx": _xlsx, ".xlsm": _xlsx, ".pptx": _pptx,
    ".odt": _odf, ".ods": _odf, ".odp": _odf,
    ".csv": _delimited, ".tsv": _delimited,
    ".txt": _plain, ".md": _plain, ".log": _plain, ".json": _plain,
    ".xml": _markup, ".html": _markup, ".htm": _markup,
}


def check_supported(original_name: str) -> str:
    """Return the lowercase extension if the type is supported, else raise ExtractionError."""
    ext = Path(original_name).suffix.lower()
    if ext in LEGACY_EXTENSIONS:
        raise ExtractionError(
            f"{ext} files are an old binary format that is not supported. "
            f"Open it and save it as {LEGACY_EXTENSIONS[ext]}, then upload that."
        )
    if ext not in _EXTRACTORS:
        raise ExtractionError(
            f"Unsupported file type '{ext or original_name}'. Supported: "
            + ", ".join(SUPPORTED_EXTENSIONS) + "."
        )
    return ext


def extract_text(path: Path, original_name: str) -> Tuple[str, Dict]:
    """Return (text, info) for the file, or raise ExtractionError with a user-facing message."""
    fn = _EXTRACTORS[check_supported(original_name)]
    try:
        text, info = fn(path)
    except ExtractionError:
        raise
    except MemoryError:
        raise ExtractionError("This file is too large to process.")
    except Exception as e:                                       # corrupt/unexpected files
        raise ExtractionError(f"This file could not be read ({type(e).__name__}: {_clean(e, path)}).")
    text = _tidy(text)
    if not text:
        raise ExtractionError("No text was found in this file.")
    return text, info
