"""Text extraction from uploaded documents: every supported type, and the failure messages for bad files."""
import zipfile

import pytest

import doc_extract as d

SENTENCE = "Sundar Pichai, the CEO of Google, met Satya Nadella of Microsoft in Seattle."


def extract(path):
    return d.extract_text(path, path.name)


# ── supported formats ────────────────────────────────────────────────────
def test_txt_md_json_log(tmp_path):
    for name in ("a.txt", "a.md", "a.json", "a.log"):
        p = tmp_path / name
        p.write_text(SENTENCE, encoding="utf-8")
        assert SENTENCE in extract(p)[0]


def test_utf16_and_cp1252_text(tmp_path):
    p = tmp_path / "hi.txt"
    p.write_text("नरेंद्र मोदी नई दिल्ली में हैं।", encoding="utf-16")
    assert "नरेंद्र मोदी" in extract(p)[0]
    q = tmp_path / "latin.txt"
    q.write_bytes("Café München".encode("cp1252"))
    assert extract(q)[0] == "Café München"


def test_csv_and_tsv(tmp_path):
    (tmp_path / "t.csv").write_text("name,company\nAsha,Google\n", encoding="utf-8")
    (tmp_path / "t.tsv").write_text("name\tcompany\nRavi\tTesla\n", encoding="utf-8")
    text, info = extract(tmp_path / "t.csv")
    assert "name | company" in text and "Asha | Google" in text and info["rows"] == 2
    assert "Ravi | Tesla" in extract(tmp_path / "t.tsv")[0]


def test_html_drops_scripts_and_styles(tmp_path):
    p = tmp_path / "p.html"
    p.write_text("<html><head><style>p{}</style></head><body><h1>Title</h1><p>Hello <b>Google</b>.</p>"
                 "<script>var secret=1;</script></body></html>", encoding="utf-8")
    text = extract(p)[0]
    assert "Title" in text and "Hello Google." in text and "secret" not in text and "p{}" not in text


def test_xml_text_content(tmp_path):
    p = tmp_path / "f.xml"
    p.write_text("<root><a>Amazon in Berlin</a><b>Tesla</b></root>", encoding="utf-8")
    assert extract(p)[0] == "Amazon in Berlin\nTesla"


def test_docx_paragraphs_tables_and_headers(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_heading("Report", 1)
    doc.add_paragraph(SENTENCE)
    table = doc.add_table(rows=2, cols=2)
    for i, row in enumerate([("Name", "City"), ("Asha", "Pune")]):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    doc.sections[0].header.paragraphs[0].text = "Confidential header"
    p = tmp_path / "r.docx"
    doc.save(p)
    text, info = extract(p)
    assert "Report" in text and SENTENCE in text and "Asha | Pune" in text and "Confidential header" in text
    assert info["tables"] == 1


def test_xlsx_every_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "People"
    wb.active.append(["Name", "Company"])
    wb.active.append(["Satya Nadella", "Microsoft"])
    wb.create_sheet("Notes").append([SENTENCE])
    p = tmp_path / "d.xlsx"
    wb.save(p)
    text, info = extract(p)
    assert "[Sheet: People]" in text and "Satya Nadella | Microsoft" in text and "[Sheet: Notes]" in text
    assert info == {"sheets": 2, "rows": 3}


def test_pptx_slides_and_notes(tmp_path):
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Partnership"
    slide.placeholders[1].text = SENTENCE
    slide.notes_slide.notes_text_frame.text = "Mention Tokyo and Berlin."
    p = tmp_path / "d.pptx"
    prs.save(p)
    text, info = extract(p)
    assert "[Slide 1]" in text and "Partnership" in text and SENTENCE in text and "Tokyo" in text and info["slides"] == 1


def test_opendocument(tmp_path):
    p = tmp_path / "d.odt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("content.xml", '<?xml version="1.0"?><office:document-content xmlns:office="o" xmlns:text="t">'
                                  f"<office:body><text:p>{SENTENCE}</text:p><text:h>Heading Mumbai</text:h></office:body>"
                                  "</office:document-content>")
    text = extract(p)[0]
    assert SENTENCE in text and "Heading Mumbai" in text


def _make_pdf(path, pages, blank=False):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    c = canvas.Canvas(str(path), pagesize=A4)
    for n in range(pages):
        if not blank:
            y = 800
            for line in simpleSplit(f"Page {n + 1}. " + SENTENCE * 4, "Helvetica", 11, 500):
                c.setFont("Helvetica", 11)
                c.drawString(50, y, line)
                y -= 15
        c.showPage()
    c.save()


def test_pdf_text_and_page_count(tmp_path):
    p = tmp_path / "r.pdf"
    _make_pdf(p, 3)
    text, info = extract(p)
    assert info["pages"] == 3 and "Page 1." in text and "Page 3." in text
    assert "Sundar Pichai, the CEO of Google" in text                  # wrapped lines were joined back together


def test_scanned_pdf_gets_a_clear_message(tmp_path):
    p = tmp_path / "scan.pdf"
    _make_pdf(p, 2, blank=True)
    with pytest.raises(d.ExtractionError, match="scan"):
        extract(p)


# ── bad input ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["bad.pdf", "bad.docx", "bad.xlsx", "bad.pptx", "bad.odt"])
def test_corrupt_files_raise_a_readable_error_without_leaking_paths(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"this is not a real document")
    with pytest.raises(d.ExtractionError) as exc:
        extract(p)
    assert str(tmp_path) not in str(exc.value)


def test_empty_file(tmp_path):
    p = tmp_path / "e.txt"
    p.write_text("  \n\n ", encoding="utf-8")
    with pytest.raises(d.ExtractionError, match="No text"):
        extract(p)


@pytest.mark.parametrize("name,hint", [("old.doc", ".docx"), ("old.xls", ".xlsx"), ("old.ppt", ".pptx")])
def test_legacy_office_formats_explain_what_to_do(tmp_path, name, hint):
    with pytest.raises(d.ExtractionError, match=hint):
        d.check_supported(name)


def test_unsupported_type_lists_the_supported_ones():
    with pytest.raises(d.ExtractionError, match=r"Supported: .*\.pdf"):
        d.check_supported("archive.zip")


# ── PDF line reflow (the justified-text bug) ─────────────────────────────
def test_reflow_repairs_one_word_per_line_pdfs():
    fragmented = "Our\n\ndiverse\n\nportfolio\n\nincludes\n\ngenerative\n\nAI,\n\ndigital\n\ntransformation."
    out = d._reflow(fragmented)
    assert out == "Our diverse portfolio includes generative AI, digital transformation."


def test_reflow_joins_wrapped_lines_but_keeps_paragraphs():
    text = ("The quick brown fox jumps over the lazy dog and keeps\nrunning through the forest until it reaches a river.\n"
            "Then it swims across.\n\nSecond paragraph starts here and is\nalso wrapped over two lines.")
    out = d._reflow(text).split("\n")
    assert out[0].startswith("The quick brown fox") and "reaches a river. Then it swims across." in out[0]
    assert out[1] == "Second paragraph starts here and is also wrapped over two lines."


def test_reflow_repairs_hyphenation():
    assert "transformation of enterprise" in d._reflow("A sentence about transfor-\nmation of enterprise products.")


def test_reflow_splits_bullets_and_numbered_items():
    out = d._reflow("Skills:\n1. Python and a long list of other things here to fill up the line width entirely\n"
                    "2. SQL and a long list of other things here to fill up the line width entirely\n● First ● Second").split("\n")
    assert out[0] == "Skills:" and out[1].startswith("1. Python") and out[2].startswith("2. SQL")
    assert out[3:] == ["● First", "● Second"]


def test_reflow_keeps_label_lines_apart():
    # "short" is judged against the page's typical line width, so the page needs some full-width body text too
    body = "This paragraph runs across the full width of the page so that short label lines can be told apart from it. " * 2
    page = f"{body.strip()}\n\nCompany: Gravity Engineering\nLocation: Raipur\nPosition: Internship"
    out = d._reflow(page).split("\n")
    assert out[1:] == ["Company: Gravity Engineering", "Location: Raipur", "Position: Internship"]


@pytest.mark.parametrize("text", ["a b c.\nd e f.", "x\n\ny\n\nz\n\nw", "line one is here\nline two is here.", "● a ● b ● c"])
def test_reflow_never_loses_or_invents_words(text):
    assert [w for w in d._reflow(text).split() if w != "●"] == [w for w in text.split() if w != "●"]
