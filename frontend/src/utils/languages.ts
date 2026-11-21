// Languages the app offers, plus the quality status of each Indian language's entity recognition.
//
// The status (stable / beta / experimental / blocked) is NOT typed here: the backend generates it from the evaluation results
// (evaluation/make_language_status.py) and serves it at GET /api/languages. This file only adds names and turns the response into
// what the screens need.
import { useEffect, useState } from 'react';
import axios from 'axios';
import { BACKEND_URL } from '../config';

export type Language = { code: string; name: string };

// Always offered. Hindi's quality badge still comes from the API.
export const BASE_LANGUAGES: Language[] = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'Hindi' },
  { code: 'fr', name: 'French' },
  { code: 'es', name: 'Spanish' },
  { code: 'de', name: 'German' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ar', name: 'Arabic' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ru', name: 'Russian' },
];

// Names for every language the app can ever show (history and exports use them even for languages not currently offered).
export const LANGUAGE_NAMES: Record<string, string> = {
  ...Object.fromEntries(BASE_LANGUAGES.map((l) => [l.code, l.name])),
  mr: 'Marathi', bn: 'Bengali', ta: 'Tamil', te: 'Telugu', gu: 'Gujarati',
  kn: 'Kannada', ml: 'Malayalam', pa: 'Punjabi', or: 'Odia', as: 'Assamese',
};

export type LanguageInfo = {
  code: string;
  name: string;
  offered: boolean;            // shown in the language lists
  status: string;              // stable | beta | experimental | blocked | not_measured
  f1: number | null;
  sentences: number;
  testSet: string;
  reasons: string[];
  offlineTranslation: boolean; // translation works without a cloud key
  azureWorks: boolean;         // Azure Translator handled a test sentence
};

export type LanguageTable = { byCode: Record<string, LanguageInfo>; order: string[]; azureConfigured: boolean; loaded: boolean };

const EMPTY: LanguageTable = { byCode: {}, order: [], azureConfigured: false, loaded: false };

// Pure function (easy to test): API response -> table.
export function parseLanguageTable(data: any): LanguageTable {
  const byCode: Record<string, LanguageInfo> = {};
  const order: string[] = [];
  for (const l of data?.languages ?? []) {
    byCode[l.code] = {
      code: l.code,
      name: l.name ?? LANGUAGE_NAMES[l.code] ?? l.code,
      offered: !!l.offered_in_app,
      status: l.ner?.status ?? 'not_measured',
      f1: l.ner?.f1 ?? null,
      sentences: l.ner?.sentences ?? 0,
      testSet: l.ner?.test_set ?? '',
      reasons: l.ner?.reasons ?? [],
      offlineTranslation: !!l.offline_translation,
      azureWorks: !!l.azure_translation_works,
    };
    order.push(l.code);
  }
  return { byCode, order, azureConfigured: !!data?.azure_configured, loaded: true };
}

// The languages to show: the base ones first, then every other offered language in the order the backend gives (priority).
export function offeredLanguages(table: LanguageTable): Language[] {
  const base = new Set(BASE_LANGUAGES.map((l) => l.code));
  const extra = table.order
    .filter((c) => !base.has(c) && table.byCode[c]?.offered)
    .map((c) => ({ code: c, name: table.byCode[c].name }));
  return [...BASE_LANGUAGES, ...extra];
}

// Short badge next to a language name ('' when there is nothing to warn about).
export function badgeFor(table: LanguageTable, code: string): string {
  const info = table.byCode[code];
  if (!info) return '';
  return info.status === 'stable' ? '' : info.status === 'beta' ? 'beta' : info.status === 'experimental' ? 'experimental' : '';
}

// One or two plain sentences about the selected language, or '' when there is nothing to say.
export function languageNote(table: LanguageTable, code: string, role: 'source' | 'target'): string {
  const info = table.byCode[code];
  if (!info) return '';
  const parts: string[] = [];
  if (role === 'source' && (info.status === 'beta' || info.status === 'experimental') && info.f1 !== null) {
    parts.push(
      `${info.name}: entity recognition is in ${info.status} (F1 ${info.f1.toFixed(2)} on ${info.sentences} human-labelled test sentences). ` +
      `Names may be missed or mislabelled, so review the results before using them as training data.`,
    );
  }
  if (role === 'target' && code !== 'en' && info && !info.offlineTranslation && !table.azureConfigured) {
    parts.push(`${info.name}: translation needs Azure Translator, which is not set up on this server.`);
  }
  return parts.join(' ');
}

// Loads /api/languages once. Failure is harmless: the base languages still work, just without badges.
export function useLanguageTable(): LanguageTable {
  const [table, setTable] = useState<LanguageTable>(EMPTY);
  useEffect(() => {
    let cancelled = false;
    axios.get(`${BACKEND_URL}/api/languages`, { timeout: 8000 })
      .then((r) => { if (!cancelled) setTable(parseLanguageTable(r.data)); })
      .catch(() => { /* keep the defaults */ });
    return () => { cancelled = true; };
  }, []);
  return table;
}
