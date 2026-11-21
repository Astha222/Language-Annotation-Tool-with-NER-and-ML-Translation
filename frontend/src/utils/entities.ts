// Helpers for NER entities stored as a JSON string on each annotation.

export interface Entity {
  entity_group: string;
  word: string;
  score?: number;
  start?: number;
  end?: number;
}

const GROUP_LABEL: Record<string, string> = {
  PER: 'Person', ORG: 'Organization', LOC: 'Location', MISC: 'Miscellaneous', DATE: 'Date',
};

// Older annotations were saved with word-piece fragments ("Sa", "##ty", "##a Nadella").
// Merge every "##" piece back into the entity before it so they read "Satya Nadella".
export function cleanEntities(entities: Entity[]): Entity[] {
  const out: Entity[] = [];
  for (const e of entities) {
    const prev = out[out.length - 1];
    if (e.word.startsWith('##') && prev) {
      prev.word += e.word.slice(2);
      prev.end = e.end ?? prev.end;
      if (prev.score !== undefined && e.score !== undefined) prev.score = Math.min(prev.score, e.score);
    } else {
      out.push({ ...e, word: e.word.replace(/^##/, '') });
    }
  }
  return out;
}

export function parseEntities(json: string | null | undefined): Entity[] {
  if (!json) return [];
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) ? cleanEntities(parsed) : [];
  } catch {
    return [];
  }
}

// Clean JSON string for machine-readable formats (json/csv/xlsx/xml)
export function entitiesToJson(json: string | null | undefined): string {
  const list = parseEntities(json);
  return list.length ? JSON.stringify(list) : '';
}

// One entity per line for human-readable formats (pdf/txt)
export function entitiesToText(json: string | null | undefined): string {
  return parseEntities(json)
    .map((e) => {
      const pct = e.score !== undefined ? `, ${Math.round(e.score * 100)}%` : '';
      return `${e.word} (${GROUP_LABEL[e.entity_group] || e.entity_group}${pct})`;
    })
    .join('\n');
}
