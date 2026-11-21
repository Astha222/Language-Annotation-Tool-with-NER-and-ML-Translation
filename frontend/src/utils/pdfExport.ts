// PDF export via the browser's print dialog ("Save as PDF").
// Unlike a hand-built PDF, this renders any script (Hindi, Chinese, Arabic, ...),
// wraps long lines and paginates automatically.

const esc = (s: unknown) =>
  String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

export function exportPdf(title: string, rows: Record<string, string>[]): boolean {
  if (typeof window === 'undefined' || typeof document === 'undefined') return false;
  const w = window.open('', '_blank');
  if (!w) return false; // popup blocked

  const records = rows
    .map(
      (r, i) =>
        `<section><h2>Record ${i + 1}</h2><table>${Object.entries(r)
          .map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`)
          .join('')}</table></section>`
    )
    .join('');

  w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>${esc(title)}</title>
<style>
  body{font-family:system-ui,"Segoe UI","Noto Sans","Noto Sans Devanagari","Noto Sans CJK SC","Noto Sans Arabic",sans-serif;margin:24px;color:#111}
  h1{font-size:20px} h2{font-size:14px;margin:18px 0 6px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th,td{border:1px solid #ccc;padding:5px 8px;text-align:left;vertical-align:top;word-break:break-word;white-space:pre-wrap}
  th{width:150px;background:#f3f4f6} section{page-break-inside:avoid}
</style></head><body><h1>${esc(title)}</h1>${records}</body></html>`);
  w.document.close();
  w.focus();
  setTimeout(() => w.print(), 300);
  return true;
}
