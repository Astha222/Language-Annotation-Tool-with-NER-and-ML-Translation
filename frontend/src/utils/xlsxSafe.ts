import * as XLSX from 'xlsx';

// Excel allows at most 32,767 characters in one cell; document-sized text exceeds that.
const MAX_CELL = 32000;

// Normal sheet when every value fits; otherwise a "long" layout (record, field, part, value)
// where oversized values are split across consecutive rows so nothing is lost or rejected.
export function makeSheet(rows: Record<string, any>[]): XLSX.WorkSheet {
  const fits = rows.every((r) => Object.values(r).every((v) => String(v ?? '').length <= MAX_CELL));
  if (fits) return XLSX.utils.json_to_sheet(rows);

  const out: { record: string | number; field: string; part: number; value: string }[] = [];
  rows.forEach((r, i) => {
    const record = r.id ?? i + 1;
    Object.entries(r).forEach(([field, v]) => {
      const s = String(v ?? '');
      if (s.length <= MAX_CELL) {
        out.push({ record, field, part: 1, value: s });
      } else {
        for (let p = 0, n = 1; p < s.length; p += MAX_CELL, n++) {
          out.push({ record, field, part: n, value: s.slice(p, p + MAX_CELL) });
        }
      }
    });
  });
  return XLSX.utils.json_to_sheet(out);
}
