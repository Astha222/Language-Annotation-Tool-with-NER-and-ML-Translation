import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ActivityIndicator, ScrollView, Platform,
} from 'react-native';
import axios from 'axios';
import { LANGUAGE_NAMES, useLanguageTable, offeredLanguages, badgeFor, languageNote } from '../utils/languages';
import * as XLSX from 'xlsx';
import { useAuth } from '../context/AuthContext';
import { exportPdf } from '../utils/pdfExport';
import { entitiesToText } from '../utils/entities';
import { makeSheet } from '../utils/xlsxSafe';

import { BACKEND_URL } from '../config';

// Files the backend can read (see backend/doc_extract.py); anything else is rejected with a clear message
const DOC_ACCEPT = '.pdf,.docx,.xlsx,.xlsm,.pptx,.odt,.ods,.odp,.csv,.tsv,.txt,.md,.log,.json,.xml,.html,.htm';
const MAX_UPLOAD_MB = 50;
// small plain-text files still load straight into the editable text box
const INLINE_TEXT_MAX_BYTES = 200 * 1024;

interface DocInfo {
  doc_id: string; filename: string; size: number; chars: number; words: number;
  info: Record<string, number>; preview: string; detected_language: string | null;
}
interface JobState { status: string; stage: string; progress: number }

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const fmtSize = (b: number) => (b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);
const INFO_LABEL: Record<string, string> = {
  pages: 'pages', sheets: 'sheets', rows: 'rows', slides: 'slides', paragraphs: 'paragraphs', tables: 'tables',
};

const NER_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  PER:  { bg: '#e3f2fd', text: '#1565c0', label: 'Person' },
  ORG:  { bg: '#e8f5e9', text: '#2e7d32', label: 'Organization' },
  LOC:  { bg: '#fff3e0', text: '#e65100', label: 'Location' },
  MISC: { bg: '#f3e5f5', text: '#6a1b9a', label: 'Miscellaneous' },
  DATE: { bg: '#e0f7fa', text: '#00695c', label: 'Date' },
};

// The language lists come from utils/languages.ts: the base languages plus every Indian language the backend says is offered.
const LANG_NAME: Record<string, string> = LANGUAGE_NAMES;

// ── Toast ──────────────────────────────────────────────────────
function Toast({ message, type }: { message: string; type: 'success' | 'error' | 'info' }) {
  const bg = type === 'success' ? '#16a34a' : type === 'error' ? '#dc2626' : '#2563eb';
  return (
    <View style={[styles.toast, { backgroundColor: bg }]}>
      <Text style={styles.toastText}>{message}</Text>
    </View>
  );
}

// ── Export helpers ─────────────────────────────────────────────
function triggerDownload(blob: Blob, filename: string) {
  if (typeof document === 'undefined') return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function buildExportRow(inputText: string, translation: string, ner: any[], src: string, tgt: string) {
  return {
    original_text: inputText,
    translated_text: translation,
    source_language: LANG_NAME[src] || src.toUpperCase(),
    target_language: LANG_NAME[tgt] || tgt.toUpperCase(),
    entities: JSON.stringify(ner),
    exported_at: new Date().toISOString(),
  };
}

const doExport: Record<string, (row: Record<string, string>) => void> = {
  json: (row) => triggerDownload(new Blob([JSON.stringify([row], null, 2)], { type: 'application/json' }), `annotation_${Date.now()}.json`),
  csv: (row) => {
    const h = Object.keys(row).join(',');
    const v = Object.values(row).map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',');
    triggerDownload(new Blob([`${h}\n${v}`], { type: 'text/csv' }), `annotation_${Date.now()}.csv`);
  },
  xlsx: (row) => {
    const ws = makeSheet([row]);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Annotation');
    XLSX.writeFile(wb, `annotation_${Date.now()}.xlsx`);
  },
  txt: (row) => triggerDownload(new Blob([Object.entries(row).map(([k, v]) => `${k}: ${v}`).join('\n')], { type: 'text/plain' }), `annotation_${Date.now()}.txt`),
  xml: (row) => {
    const esc = (s: string) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<annotation>\n${Object.entries(row).map(([k, v]) => `  <${k}>${esc(v)}</${k}>`).join('\n')}\n</annotation>`;
    triggerDownload(new Blob([xml], { type: 'application/xml' }), `annotation_${Date.now()}.xml`);
  },
  pdf: (row) => { exportPdf('Annotation', [{ ...row, entities: entitiesToText(row.entities) }]); },
};

const EXPORT_OPTIONS: { key: string; label: string }[] = [
  { key: 'json',  label: '.JSON' },
  { key: 'csv',   label: '.CSV' },
  { key: 'xlsx',  label: '.XLSX' },
  { key: 'txt',   label: '.TXT' },
  { key: 'xml',   label: '.XML' },
  { key: 'pdf',   label: '.PDF' },
  { key: 'conll', label: 'CoNLL / BIO' },
  { key: 'jsonl', label: 'JSONL' },
];

// ── Main Component ─────────────────────────────────────────────
export default function HomeScreen() {
  const { token } = useAuth();
  const authHeaders = () => ({ headers: { Authorization: `Bearer ${token}` } });

  const [inputText, setInputText]               = useState('');
  const [sourceLang, setSourceLang]             = useState('auto');
  const [targetLang, setTargetLang]             = useState('fr');
  const langTable = useLanguageTable();            // quality status of each Indian language (GET /api/languages)
  const targetLanguages = offeredLanguages(langTable);
  const sourceLanguages = [{ code: 'auto', name: 'Auto Detect' }, ...targetLanguages];
  const [nerLoading, setNerLoading]             = useState(false);
  const [transLoading, setTransLoading]         = useState(false);
  const [bothLoading, setBothLoading]           = useState(false);
  const loading = nerLoading || transLoading || bothLoading;
  const [nerRan, setNerRan]                     = useState(false);
  const [transRan, setTransRan]                 = useState(false);
  const [nerResults, setNerResults]             = useState<any[]>([]);
  const [translationResult, setTranslationResult] = useState('');
  const [detectedLang, setDetectedLang]         = useState('');
  const [toast, setToast]                       = useState<{ msg: string; type: 'success' | 'error' | 'info' } | null>(null);
  const [saved, setSaved]                       = useState(false);
  const [savedId, setSavedId]                   = useState<number | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [showExportMenu, setShowExportMenu]     = useState(false);

  // document (file upload) mode
  const [doc, setDoc]                           = useState<DocInfo | null>(null);
  const [uploading, setUploading]               = useState(false);
  const [uploadPct, setUploadPct]               = useState(0);
  const [job, setJob]                           = useState<JobState | null>(null);
  const [nerSummary, setNerSummary]             = useState<{ total: number; unique: number } | null>(null);
  const [resultNote, setResultNote]             = useState('');
  const [engineNote, setEngineNote]             = useState('');   // which translator produced the output
  const [jobWarnings, setJobWarnings]           = useState<string[]>([]);
  const jobIdRef = useRef('');
  const pollRef  = useRef({ stop: false });
  // keep the latest token/doc reachable from the file-input callback
  const uploadRef = useRef<(f: File) => void>(() => {});

  // bulk upload state
  const [bulkUploading, setBulkUploading]       = useState(false);
  const [bulkResults, setBulkResults]           = useState<{ uploaded: DocInfo[]; errors: { filename: string; error: string }[] } | null>(null);

  useEffect(() => () => { pollRef.current.stop = true; }, []);   // stop polling when leaving the screen

  // What "Analyze & Translate" is doing now, and a note when one step is slow. The first request for a language makes the
  // server download and load that language's model (measured: 1-1.5 minutes for a translation model on a slow connection,
  // a few seconds once it is cached), which otherwise looks as if the app had frozen.
  const SLOW_STEP_MS = 8000;
  const [stepLabel, setStepLabel] = useState('');
  const [stepSlow, setStepSlow]   = useState(false);
  const slowTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startStep = (label: string) => {
    setStepLabel(label); setStepSlow(false);
    if (slowTimer.current) clearTimeout(slowTimer.current);
    slowTimer.current = setTimeout(() => setStepSlow(true), SLOW_STEP_MS);
  };
  const endSteps = () => {
    if (slowTimer.current) clearTimeout(slowTimer.current);
    slowTimer.current = null;
    setStepLabel(''); setStepSlow(false);
  };
  useEffect(() => () => { if (slowTimer.current) clearTimeout(slowTimer.current); }, []);

  const showToast = (msg: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const lastAnalyzedText = useRef<string | null>(null);

  const resetResults = () => {
    setNerResults([]); setTranslationResult(''); setDetectedLang(''); setSaved(false); setSavedId(null);
    setNerSummary(null); setResultNote(''); setJobWarnings([]); setEngineNote('');
    setNerRan(false); setTransRan(false);
  };

  const handleTextChange = (text: string) => {
    setInputText(text);
    if (lastAnalyzedText.current !== null && text !== lastAnalyzedText.current) {
      resetResults();
      lastAnalyzedText.current = null;
    } else if (savedId !== null || nerRan || transRan || nerResults.length > 0 || translationResult) {
      resetResults();
      lastAnalyzedText.current = null;
    }
  };

  useEffect(() => {
    if (lastAnalyzedText.current !== null && inputText !== lastAnalyzedText.current) {
      resetResults();
      lastAnalyzedText.current = null;
    }
  }, [inputText]);

  const removeDocOnServer = (id: string) => {
    axios.delete(`${BACKEND_URL}/api/documents/${id}`, authHeaders()).catch(() => {});
  };

  const removeDoc = () => {
    if (doc) removeDocOnServer(doc.doc_id);
    setDoc(null); setUploadedFileName(''); lastAnalyzedText.current = null; resetResults();
  };

  const uploadDocument = async (file: File) => {
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      showToast(`File too large (max ${MAX_UPLOAD_MB} MB)`, 'error');
      return;
    }
    setUploading(true); setUploadPct(0); lastAnalyzedText.current = null; resetResults();
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await axios.post(`${BACKEND_URL}/api/documents`, form, {
        ...authHeaders(),
        onUploadProgress: (e) => { if (e.total) setUploadPct(Math.round((100 * e.loaded) / e.total)); },
      });
      if (doc) removeDocOnServer(doc.doc_id);       // replacing an earlier upload
      const d: DocInfo = res.data;
      setDoc(d); setInputText(''); setUploadedFileName(d.filename);
      lastAnalyzedText.current = null;
      resetResults();
      if (sourceLang === 'auto' && d.detected_language) setDetectedLang(d.detected_language);
      showToast(`"${d.filename}" ready - ${d.chars.toLocaleString()} characters`, 'success');
    } catch (err: any) {
      if (err?.response?.status === 401) return;     // session expired: the interceptor logs out
      const detail = err?.response?.data?.detail;
      showToast(typeof detail === 'string' ? detail : 'Upload failed - is the file under 50 MB?', 'error');
    } finally {
      setUploading(false);
    }
  };
  uploadRef.current = uploadDocument;

  // Programmatic file input — avoids DOM nesting error in React Native Web
  const triggerFileInput = useCallback(() => {
    if (typeof document === 'undefined') return;
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = DOC_ACCEPT;
    inp.onchange = (e: any) => {
      const file: File = e.target?.files?.[0];
      if (!file) return;
      if (/\.(txt|md)$/i.test(file.name) && file.size <= INLINE_TEXT_MAX_BYTES) {
        // small plain text: load into the editable box as before
        const reader = new FileReader();
        reader.onload = (evt) => {
          setDoc(null);
          setUploadedFileName(file.name);
          const t = (evt.target?.result as string) || '';
          setInputText(t);
          lastAnalyzedText.current = null;
          resetResults();
          showToast(`"${file.name}" loaded`, 'success');
        };
        reader.readAsText(file);
      } else {
        uploadRef.current(file);
      }
    };
    inp.click();
  }, []);

  // Bulk file upload trigger (up to 20 files)
  const triggerBulkFileInput = useCallback(() => {
    if (typeof document === 'undefined') return;
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = DOC_ACCEPT;
    inp.multiple = true;
    inp.onchange = async (e: any) => {
      const files: File[] = Array.from(e.target?.files || []);
      if (!files.length) return;
      if (files.length > 20) { showToast('Max 20 files at once', 'error'); return; }
      if (files.length === 1) { uploadRef.current(files[0]); return; }
      setBulkUploading(true);
      setBulkResults(null);
      try {
        const form = new FormData();
        files.forEach(f => form.append('files', f));
        const res = await axios.post(`${BACKEND_URL}/api/documents/bulk`, form, authHeaders());
        setBulkResults(res.data);
        const { uploaded, errors } = res.data;
        showToast(
          `${uploaded.length} file${uploaded.length !== 1 ? 's' : ''} uploaded${errors.length ? `, ${errors.length} failed` : ''}`,
          errors.length ? 'info' : 'success',
        );
        // Auto-select the first successfully uploaded document
        if (uploaded.length > 0) {
          if (doc) removeDocOnServer(doc.doc_id);
          setDoc(uploaded[0]);
          setUploadedFileName(uploaded[0].filename);
          if (sourceLang === 'auto' && uploaded[0].detected_language) setDetectedLang(uploaded[0].detected_language);
        }
      } catch (err: any) {
        if (err?.response?.status === 401) return;
        showToast(err?.response?.data?.detail || 'Bulk upload failed', 'error');
      } finally {
        setBulkUploading(false);
      }
    };
    inp.click();
  }, [doc, sourceLang]);

  const copyText = async (text: string, label = 'Copied!') => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard) {
        await navigator.clipboard.writeText(text);
        showToast(label, 'success');
      } else {
        showToast('Copy not supported here', 'info');
      }
    } catch { showToast('Copy failed', 'error'); }
  };

  const handleSaveInput = () => {
    if (!inputText.trim()) { showToast('No text to save', 'error'); return; }
    triggerDownload(new Blob([inputText], { type: 'text/plain' }), `input_${Date.now()}.txt`);
    showToast('Input saved as .txt', 'success');
  };

  // ── uploaded document: background job on the server, polled for progress ──
  const applyJobResult = (j: any) => {
    const res = j.result;
    const mode = res.mode || 'both';
    setDetectedLang(res.source_lang);
    setSavedId(res.annotation_id); setSaved(true);
    setJobWarnings(j.warnings || []);

    if (mode === 'ner' || mode === 'both') {
      setNerResults(res.entities || []);
      setNerSummary({ total: res.entity_total, unique: res.entity_unique });
      setNerRan(true);
    }
    if (mode === 'translate' || mode === 'both') {
      const preview: string = res.translated_preview || '';
      setTranslationResult(preview);
      setEngineNote((res.translation_engines || []).join(' + '));
      setTransRan(true);
      if (res.translated_chars > preview.length) {
        setResultNote(
          `Showing the first ${preview.length.toLocaleString()} of ${res.translated_chars.toLocaleString()} translated characters. ` +
          'The full translation is saved in History - use Export to get all of it.'
        );
      }
    }
    const msg = mode === 'ner'
      ? 'Document NER analysis completed and saved'
      : mode === 'translate'
      ? 'Document translation completed and saved'
      : 'Document analyzed and saved';
    showToast(j.warnings?.length ? 'Saved with a warning - see the Output panel' : msg,
              j.warnings?.length ? 'info' : 'success');
  };

  const analyzeDocument = async (mode: 'ner' | 'translate' | 'both') => {
    if (!doc) return;
    if (mode === 'ner') {
      setNerLoading(true);
      setNerResults([]);
      setNerSummary(null);
      setNerRan(false);
      setTranslationResult('');
      setResultNote('');
      setEngineNote('');
      setTransRan(false);
    } else if (mode === 'translate') {
      if (!targetLang) {
        showToast('Please select a target language', 'error');
        return;
      }
      setTransLoading(true);
      setTranslationResult('');
      setResultNote('');
      setEngineNote('');
      setTransRan(false);
      setNerResults([]);
      setNerSummary(null);
      setNerRan(false);
    } else {
      if (!targetLang) {
        showToast('Please select a target language', 'error');
        return;
      }
      setBothLoading(true);
      setNerResults([]);
      setNerSummary(null);
      setNerRan(false);
      setTranslationResult('');
      setResultNote('');
      setEngineNote('');
      setTransRan(false);
    }
    setJob({ status: 'queued', stage: 'Starting', progress: 0 });
    const ctl = { stop: false };
    pollRef.current.stop = true;          // end any earlier poll loop
    pollRef.current = ctl;
    try {
      const start = await axios.post(
        `${BACKEND_URL}/api/documents/${doc.doc_id}/analyze`,
        { source_lang: sourceLang, target_lang: targetLang, mode }, authHeaders(),
      );
      const jobId: string = start.data.job_id;
      jobIdRef.current = jobId;
      let failures = 0;
      while (!ctl.stop) {
        await sleep(1500);
        if (ctl.stop) return;
        let j: any;
        try {
          j = (await axios.get(`${BACKEND_URL}/api/jobs/${jobId}`, authHeaders())).data;
          failures = 0;
        } catch (e: any) {
          const code = e?.response?.status;
          if (code === 404 || code === 401 || ++failures >= 5) throw e;   // gone, logged out, or server unreachable
          continue;                                                        // brief network blip: try again
        }
        setJob({ status: j.status, stage: j.stage, progress: j.progress });
        if (j.status === 'done')      { applyJobResult(j); return; }
        if (j.status === 'error')     { showToast(j.error || 'Processing failed', 'error'); return; }
        if (j.status === 'cancelled') { showToast('Cancelled', 'info'); return; }
      }
    } catch (err: any) {
      if (err?.response?.status === 401) return;
      const detail = err?.response?.data?.detail;
      showToast(typeof detail === 'string' ? detail : 'Lost connection to the server', 'error');
    } finally {
      if (!ctl.stop) {
        setNerLoading(false);
        setTransLoading(false);
        setBothLoading(false);
        setJob(null);
        jobIdRef.current = '';
      }
    }
  };

  const cancelJob = () => {
    if (!jobIdRef.current) return;
    axios.post(`${BACKEND_URL}/api/jobs/${jobIdRef.current}/cancel`, {}, authHeaders()).catch(() => {});
    setJob((j) => (j ? { ...j, stage: 'Cancelling…' } : j));
  };

  const resolveSourceLanguage = async (text: string): Promise<string> => {
    if (sourceLang !== 'auto') {
      return sourceLang;
    }
    startStep('Detecting language…');
    let detected = 'en';
    try {
      const r = await axios.post(`${BACKEND_URL}/api/detect-language`, { text }, authHeaders());
      const lang = r.data.detected_language;
      if (lang && LANGUAGE_NAMES[lang]) {
        detected = lang;
      } else {
        showToast('Could not detect the language - assuming English', 'info');
      }
    } catch {
      showToast('Language detection failed - assuming English', 'info');
    }
    setDetectedLang(detected);
    return detected;
  };

  const executeNer = async (text: string, lang: string): Promise<any[]> => {
    startStep('Finding entities…');
    const nerRes = await axios.post(`${BACKEND_URL}/api/ner`, { text, lang }, authHeaders());
    return nerRes.data.entities || [];
  };

  const executeTranslate = async (text: string, srcLang: string, tgtLang: string): Promise<{ translatedText: string; provider: string }> => {
    startStep('Translating…');
    const transRes = await axios.post(`${BACKEND_URL}/api/translate`, {
      text,
      source_lang: srcLang,
      target_lang: tgtLang,
    }, authHeaders());
    return {
      translatedText: transRes.data.translated_text ?? '',
      provider: transRes.data.provider || '',
    };
  };

  const handleAnalyzeNer = async () => {
    if (nerLoading || transLoading || bothLoading || uploading) return;
    if (doc) { await analyzeDocument('ner'); return; }
    if (!inputText.trim()) { showToast('Please enter some text first, or upload a file', 'error'); return; }

    setTranslationResult('');
    setTransRan(false);
    setEngineNote('');

    setNerLoading(true);
    setNerResults([]); setNerSummary(null); setJobWarnings([]);
    startStep('Analyzing NER…');

    try {
      const srcLang = await resolveSourceLanguage(inputText);
      const nerEntities = await executeNer(inputText, srcLang);
      setNerResults(nerEntities);
      setNerRan(true);
      lastAnalyzedText.current = inputText;

      startStep('Saving…');
      if (savedId !== null) {
        await axios.put(`${BACKEND_URL}/api/annotations/${savedId}`, {
          ner_tags: JSON.stringify(nerEntities),
          source_lang: srcLang,
        }, authHeaders());
        setSaved(true);
      } else {
        const saveRes = await axios.post(`${BACKEND_URL}/api/annotations`, {
          original_text: inputText,
          translated_text: null,
          ner_tags: JSON.stringify(nerEntities),
          source_lang: srcLang,
          target_lang: targetLang || 'fr',
        }, authHeaders());
        setSavedId(saveRes.data.id);
        setSaved(true);
      }
      showToast(nerEntities.length > 0 ? `NER complete! Found ${nerEntities.length} entities` : 'NER complete (no entities found)', 'success');
    } catch (err: any) {
      if (err?.response?.status === 401) return;
      showToast(err?.response?.data?.detail || 'NER analysis failed', 'error');
    } finally {
      endSteps();
      setNerLoading(false);
    }
  };

  const handleTranslate = async () => {
    if (nerLoading || transLoading || bothLoading || uploading) return;
    if (doc) { await analyzeDocument('translate'); return; }
    if (!inputText.trim()) { showToast('Please enter some text first, or upload a file', 'error'); return; }
    if (!targetLang) { showToast('Please select a target language', 'error'); return; }

    setNerResults([]);
    setNerSummary(null);
    setNerRan(false);

    setTransLoading(true);
    setTranslationResult(''); setResultNote(''); setEngineNote(''); setJobWarnings([]);
    startStep('Translating…');

    try {
      const srcLang = await resolveSourceLanguage(inputText);
      const { translatedText, provider } = await executeTranslate(inputText, srcLang, targetLang);
      setEngineNote(provider);
      setTranslationResult(translatedText);
      setTransRan(true);
      lastAnalyzedText.current = inputText;

      startStep('Saving…');
      if (savedId !== null) {
        await axios.put(`${BACKEND_URL}/api/annotations/${savedId}`, {
          translated_text: translatedText,
          source_lang: srcLang,
          target_lang: targetLang,
        }, authHeaders());
        setSaved(true);
      } else {
        const saveRes = await axios.post(`${BACKEND_URL}/api/annotations`, {
          original_text: inputText,
          translated_text: translatedText,
          ner_tags: JSON.stringify([]),
          source_lang: srcLang,
          target_lang: targetLang,
        }, authHeaders());
        setSavedId(saveRes.data.id);
        setSaved(true);
      }
      showToast('Translation complete!', 'success');
    } catch (err: any) {
      if (err?.response?.status === 401) return;
      showToast(err?.response?.data?.detail || 'Translation failed', 'error');
    } finally {
      endSteps();
      setTransLoading(false);
    }
  };

  const handleAnalyzeAndTranslate = async () => {
    if (nerLoading || transLoading || bothLoading || uploading) return;
    if (doc) { await analyzeDocument('both'); return; }
    if (!inputText.trim()) { showToast('Please enter some text first, or upload a file', 'error'); return; }
    if (!targetLang) { showToast('Please select a target language', 'error'); return; }

    setBothLoading(true);
    setNerResults([]); setNerSummary(null); setJobWarnings([]);
    setTranslationResult(''); setResultNote(''); setEngineNote('');
    startStep('Analyzing & Translating…');

    try {
      const srcLang = await resolveSourceLanguage(inputText);

      startStep('Analyzing & Translating…');

      // Safe concurrent execution of independent NER and Machine Translation
      const [nerOutcome, transOutcome] = await Promise.allSettled([
        executeNer(inputText, srcLang),
        executeTranslate(inputText, srcLang, targetLang),
      ]);

      let nerEntities: any[] = [];
      let nerSuccess = false;
      if (nerOutcome.status === 'fulfilled') {
        nerEntities = nerOutcome.value;
        setNerResults(nerEntities);
        setNerRan(true);
        nerSuccess = true;
      } else {
        console.error('NER failed during combined run', nerOutcome.reason);
      }

      let translatedText: string | null = null;
      let transSuccess = false;
      if (transOutcome.status === 'fulfilled') {
        translatedText = transOutcome.value.translatedText;
        setEngineNote(transOutcome.value.provider);
        setTranslationResult(translatedText);
        setTransRan(true);
        transSuccess = true;
      } else {
        console.error('Translation failed during combined run', transOutcome.reason);
      }

      if (!nerSuccess && !transSuccess) {
        showToast('Both NER and Translation failed', 'error');
        return;
      }

      lastAnalyzedText.current = inputText;

      startStep('Saving…');
      // Save exactly ONE history record containing both results
      if (savedId !== null) {
        await axios.put(`${BACKEND_URL}/api/annotations/${savedId}`, {
          original_text: inputText,
          translated_text: translatedText,
          ner_tags: JSON.stringify(nerEntities),
          source_lang: srcLang,
          target_lang: targetLang,
        }, authHeaders());
        setSaved(true);
      } else {
        const saveRes = await axios.post(`${BACKEND_URL}/api/annotations`, {
          original_text: inputText,
          translated_text: translatedText,
          ner_tags: JSON.stringify(nerEntities),
          source_lang: srcLang,
          target_lang: targetLang,
        }, authHeaders());
        setSavedId(saveRes.data.id);
        setSaved(true);
      }

      if (nerSuccess && transSuccess) {
        showToast('Analysis & Translation complete!', 'success');
      } else if (nerSuccess && !transSuccess) {
        showToast('NER complete, but translation failed', 'info');
      } else {
        showToast('Translation complete, but NER failed', 'info');
      }
    } catch (err: any) {
      if (err?.response?.status === 401) return;
      showToast(err?.response?.data?.detail || 'Analysis & Translation failed', 'error');
    } finally {
      endSteps();
      setBothLoading(false);
    }
  };

  const handleClear = () => {
    if (job) cancelJob();
    removeDoc();
    setInputText(''); setUploadedFileName('');
    lastAnalyzedText.current = null;
    resetResults();
  };

  const handleExport = async (fmt: string) => {
    setShowExportMenu(false);
    if (!inputText && !translationResult && savedId === null) { showToast('Nothing to export yet', 'error'); return; }

    if (fmt === 'conll' || fmt === 'jsonl') {
      if (savedId !== null) {
        try {
          const res = await axios.get(`${BACKEND_URL}/api/export?format=${fmt}&annotation_id=${savedId}`, {
            ...authHeaders(),
            responseType: 'text',
          });
          const mimeType = fmt === 'jsonl' ? 'application/x-ndjson' : 'text/plain';
          const ext = fmt === 'conll' ? 'conll' : 'jsonl';
          triggerDownload(new Blob([res.data], { type: mimeType }), `annotation_${Date.now()}.${ext}`);
          showToast(`Exported as ${fmt === 'conll' ? 'CoNLL / BIO' : 'JSONL'}`, 'success');
          return;
        } catch {
          showToast(`Failed to export as ${fmt}`, 'error');
          return;
        }
      }
    }

    // Prefer the saved server copy so the file matches what is in History even if the input box was edited
    // after analysis; fall back to what is on screen if it was not saved or the fetch fails.
    let row = buildExportRow(inputText, translationResult, nerResults, detectedLang || sourceLang, targetLang);
    if (savedId !== null) {
      try {
        const res = await axios.get(`${BACKEND_URL}/api/annotations/${savedId}`, authHeaders());
        const a = res.data;
        let ner: any[] = [];
        try { ner = a.ner_tags ? JSON.parse(a.ner_tags) : []; } catch { ner = []; }
        row = buildExportRow(a.original_text, a.translated_text ?? '', ner, a.source_lang, a.target_lang);
      } catch (err: any) {
        if (err?.response?.status === 401) return; // session expired: the interceptor logs out
        showToast('Could not load the saved copy - exporting what is on screen', 'info');
      }
    }

    doExport[fmt]?.(row);
    showToast(`Exported as .${fmt}`, 'success');
  };

  const getNerColor = (eg: string) => NER_COLORS[eg] || { bg: '#fce4ec', text: '#880e4f', label: eg };
  const hasResults = !!(translationResult || nerResults.length > 0 || (nerRan && !nerLoading));
  const resolvedSrc = detectedLang || (sourceLang !== 'auto' ? sourceLang : '');

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
      {toast ? <Toast message={toast.msg} type={toast.type} /> : null}

      {/* Language Direction Banner — shown once a language is detected or selected */}
      {resolvedSrc ? (
        <View style={styles.langBanner}>
          <View style={styles.langBannerChip}>
            <Text style={styles.langBannerLabel}>Source</Text>
            <Text style={styles.langBannerValue}>{LANG_NAME[resolvedSrc] || resolvedSrc.toUpperCase()}</Text>
          </View>
          <Text style={styles.langBannerArrow}>{'→'}</Text>
          <View style={[styles.langBannerChip, styles.langBannerChipTarget]}>
            <Text style={styles.langBannerLabel}>Target</Text>
            <Text style={[styles.langBannerValue, styles.langBannerValueTarget]}>{LANG_NAME[targetLang] || targetLang}</Text>
          </View>
          {sourceLang === 'auto' && detectedLang ? (
            <View style={styles.autoDetectedTag}>
              <Text style={styles.autoDetectedTagText}>Auto-detected</Text>
            </View>
          ) : null}
        </View>
      ) : null}

      {/* Two-column layout */}
      <View style={styles.twoCol}>

        {/* LEFT - INPUT */}
        <View style={[styles.card, styles.col]}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>Input</Text>
            <View style={styles.cardActions}>
              {!doc ? (
                <>
                  <TouchableOpacity style={styles.iconBtn} onPress={() => copyText(inputText, 'Input copied!')}>
                    <Text style={styles.iconBtnText}>Copy</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={styles.iconBtn} onPress={handleSaveInput}>
                    <Text style={styles.iconBtnText}>Save</Text>
                  </TouchableOpacity>
                </>
              ) : null}
              {Platform.OS === 'web' ? (
                <>
                  <TouchableOpacity
                    style={[styles.iconBtn, styles.iconBtnPrimary, (uploading || loading || bulkUploading) && styles.btnDisabled]}
                    onPress={triggerFileInput}
                    disabled={uploading || loading || bulkUploading}
                  >
                    <Text style={styles.iconBtnTextPrimary}>Upload File</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.iconBtn, styles.iconBtnPrimary, (uploading || loading || bulkUploading) && styles.btnDisabled]}
                    onPress={triggerBulkFileInput}
                    disabled={uploading || loading || bulkUploading}
                  >
                    <Text style={styles.iconBtnTextPrimary}>{bulkUploading ? 'Uploading…' : 'Bulk Upload'}</Text>
                  </TouchableOpacity>
                </>
              ) : null}
            </View>
          </View>

          {bulkUploading ? (
            <View style={styles.docCard}>
              <Text style={styles.docName}>Uploading multiple files…</Text>
              <Text style={styles.docMeta}>Please wait while files are being processed</Text>
            </View>
          ) : bulkResults ? (
            <View style={styles.docCard}>
              <View style={styles.docHeader}>
                <Text style={styles.docName}>Bulk Upload Complete</Text>
                <TouchableOpacity onPress={() => setBulkResults(null)}>
                  <Text style={styles.docRemove}>Dismiss</Text>
                </TouchableOpacity>
              </View>
              {bulkResults.uploaded.length > 0 && (
                <Text style={styles.docMeta}>
                  {bulkResults.uploaded.length} file{bulkResults.uploaded.length !== 1 ? 's' : ''} ready:{' '}
                  {bulkResults.uploaded.map(d => d.filename).join(', ')}
                </Text>
              )}
              {bulkResults.errors.length > 0 && (
                <Text style={[styles.docMeta, { color: '#dc2626' }]}>
                  {bulkResults.errors.length} failed:{' '}
                  {bulkResults.errors.map(e => `${e.filename} (${e.error})`).join('; ')}
                </Text>
              )}
              <Text style={styles.docHint}>
                First file selected. Click Analyze &amp; Translate to process it, then switch between files if needed.
              </Text>
            </View>
          ) : uploading ? (
            <View style={styles.docCard}>
              <Text style={styles.docName}>{uploadPct < 100 ? `Uploading… ${uploadPct}%` : 'Reading the document…'}</Text>
              <View style={styles.progressTrack}>
                <View style={[styles.progressFill, { width: `${Math.max(3, uploadPct)}%` }]} />
              </View>
              <Text style={styles.docMeta}>
                {uploadPct < 100 ? 'Sending the file to the server' : 'Extracting text - large PDFs can take a minute'}
              </Text>
            </View>
          ) : doc ? (
            <View style={styles.docCard}>
              <View style={styles.docHeader}>
                <Text style={styles.docName} numberOfLines={1}>{doc.filename}</Text>
                <TouchableOpacity onPress={removeDoc} disabled={loading}>
                  <Text style={[styles.docRemove, loading && styles.btnDisabled]}>Remove</Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.docMeta}>
                {[
                  fmtSize(doc.size),
                  `${doc.chars.toLocaleString()} characters`,
                  `${doc.words.toLocaleString()} words`,
                  ...Object.entries(doc.info || {})
                    .filter(([k, v]) => INFO_LABEL[k] && v)
                    .map(([k, v]) => `${v.toLocaleString()} ${INFO_LABEL[k]}`),
                ].join(' · ')}
              </Text>
              <Text style={styles.docPreviewLabel}>Preview</Text>
              <Text style={styles.docPreview} numberOfLines={7}>{doc.preview}</Text>
              <Text style={styles.docHint}>
                The whole document is analyzed and translated when you press Analyze & Translate.
              </Text>
            </View>
          ) : (
            <>
              {uploadedFileName ? (
                <View style={styles.fileTag}>
                  <Text style={styles.fileTagText}>{uploadedFileName}</Text>
                </View>
              ) : null}

              <TextInput
                style={styles.input}
                multiline
                numberOfLines={8}
                placeholder="Type or paste text, or upload a file (PDF, Word, Excel, PowerPoint, CSV, TXT…)"
                placeholderTextColor="#b0b8c9"
                value={inputText}
                onChangeText={handleTextChange}
              />
            </>
          )}

          {/* Source Language */}
          <Text style={styles.langSectionLabel}>Source Language</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.langScroll}>
            <View style={styles.langRow}>
              {sourceLanguages.map((lang) => (
                <TouchableOpacity
                  key={lang.code}
                  style={[styles.langChip, sourceLang === lang.code && styles.langChipActive]}
                  onPress={() => setSourceLang(lang.code)}
                >
                  <Text style={[styles.langChipText, sourceLang === lang.code && styles.langChipTextActive]}>
                    {lang.name}
                    {badgeFor(langTable, lang.code) ? <Text style={styles.langBadge}>{'  ' + badgeFor(langTable, lang.code)}</Text> : null}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </ScrollView>

          {/* Target Language */}
          <Text style={styles.langSectionLabel}>Target Language</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.langScroll}>
            <View style={styles.langRow}>
              {targetLanguages.map((lang) => (
                <TouchableOpacity
                  key={lang.code}
                  style={[styles.langChip, targetLang === lang.code && styles.langChipActive]}
                  onPress={() => setTargetLang(lang.code)}
                >
                  <Text style={[styles.langChipText, targetLang === lang.code && styles.langChipTextActive]}>
                    {lang.name}
                    {badgeFor(langTable, lang.code) ? <Text style={styles.langBadge}>{'  ' + badgeFor(langTable, lang.code)}</Text> : null}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </ScrollView>

          {(() => {
            const noteSource = sourceLang === 'auto' ? detectedLang : sourceLang;
            const note = [languageNote(langTable, noteSource, 'source'), languageNote(langTable, targetLang, 'target')].filter(Boolean).join(' ');
            return note ? <Text style={styles.langNote}>{note}</Text> : null;
          })()}

          {/* Action Buttons */}
          <View style={styles.btnRow}>
            <TouchableOpacity
              style={[
                styles.actionBtn,
                styles.nerBtn,
                (nerLoading || transLoading || bothLoading || uploading) && styles.btnDisabled,
              ]}
              onPress={handleAnalyzeNer}
              disabled={nerLoading || transLoading || bothLoading || uploading}
            >
              <Text style={styles.actionBtnText}>
                {job && job.stage.toLowerCase().includes('finding') && nerLoading
                  ? `Finding NER ${Math.round(job.progress * 100)}%`
                  : nerLoading
                  ? (stepLabel || 'Analyzing…')
                  : 'Analyze NER'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.actionBtn,
                styles.translateBtn,
                (nerLoading || transLoading || bothLoading || uploading) && styles.btnDisabled,
              ]}
              onPress={handleTranslate}
              disabled={nerLoading || transLoading || bothLoading || uploading}
            >
              <Text style={styles.actionBtnText}>
                {job && job.stage.toLowerCase().includes('translating') && transLoading
                  ? `Translating ${Math.round(job.progress * 100)}%`
                  : transLoading
                  ? (stepLabel || 'Translating…')
                  : 'Translate'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.actionBtn,
                styles.bothBtn,
                (nerLoading || transLoading || bothLoading || uploading) && styles.btnDisabled,
              ]}
              onPress={handleAnalyzeAndTranslate}
              disabled={nerLoading || transLoading || bothLoading || uploading}
            >
              <Text style={styles.actionBtnText}>
                {job && bothLoading
                  ? `Processing ${Math.round(job.progress * 100)}%`
                  : bothLoading
                  ? (stepLabel || 'Analyzing & Translating…')
                  : 'Analyze & Translate'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.ghostBtn}
              onPress={handleClear}
              disabled={nerLoading || transLoading || bothLoading || uploading}
            >
              <Text style={styles.ghostBtnText}>Clear</Text>
            </TouchableOpacity>
          </View>
          {loading && !job && stepSlow ? (
            <Text style={styles.resultNote}>
              {`Still working: ${stepLabel.replace('…', '').toLowerCase()} takes longer the first time a language is used, because the server is downloading and loading its model (it can take 1-2 minutes). Later requests take a few seconds.`}
            </Text>
          ) : null}

          {job ? (
            <View style={styles.progressWrap}>
              <View style={styles.progressTrack}>
                <View style={[styles.progressFill, { width: `${Math.max(3, Math.round(job.progress * 100))}%` }]} />
              </View>
              <View style={styles.progressRow}>
                <Text style={styles.progressText}>{job.stage}</Text>
                <TouchableOpacity onPress={cancelJob}>
                  <Text style={styles.progressCancel}>Cancel</Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.docHint}>Large documents can take several minutes. You can leave this page open and keep working.</Text>
            </View>
          ) : loading ? (
            <>
              <ActivityIndicator size="large" color="#7c3aed" style={{ marginTop: 14 }} />
              <Text style={styles.docHint}>Offline models download on first use and can take a few minutes.</Text>
            </>
          ) : null}
        </View>

        {/* RIGHT - OUTPUT */}
        <View style={[styles.card, styles.col]}>
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>Output</Text>
            {hasResults ? (
              <View style={styles.exportWrap}>
                <TouchableOpacity
                  style={styles.exportDropBtn}
                  onPress={() => setShowExportMenu(!showExportMenu)}
                >
                  <Text style={styles.exportDropBtnText}>{`Export  ${showExportMenu ? '▲' : '▼'}`}</Text>
                </TouchableOpacity>
                {showExportMenu ? (
                  <View style={styles.exportMenu}>
                    {EXPORT_OPTIONS.map((opt) => (
                      <TouchableOpacity key={opt.key} style={styles.exportMenuItem} onPress={() => handleExport(opt.key)}>
                        <Text style={styles.exportMenuText}>{opt.label}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                ) : null}
              </View>
            ) : null}
          </View>

          {/* Source → Target language row in output panel */}
          {resolvedSrc ? (
            <View style={styles.outputLangRow}>
              <Text style={styles.outputLangText}>{LANG_NAME[resolvedSrc] || resolvedSrc.toUpperCase()}</Text>
              {translationResult ? (
                <>
                  <Text style={styles.outputLangArrow}>{'→'}</Text>
                  <Text style={[styles.outputLangText, styles.outputLangTarget]}>{LANG_NAME[targetLang] || targetLang}</Text>
                </>
              ) : null}
            </View>
          ) : null}

          {/* Job warnings (e.g. translation unavailable) stay visible even when there is no translation to show */}
          {jobWarnings.length > 0 ? (
            <View style={styles.warningBox}>
              {jobWarnings.map((w, i) => <Text key={i} style={styles.warningBoxText}>{w}</Text>)}
            </View>
          ) : null}

          {!hasResults && !loading && jobWarnings.length === 0 ? (
            <View style={styles.emptyOutput}>
              <View style={styles.emptyOutputIcon}>
                <Text style={styles.emptyOutputIconText}>O</Text>
              </View>
              <Text style={styles.emptyOutputTitle}>No output yet</Text>
              <Text style={styles.emptyOutputSub}>Enter text or upload a document on the left and click Analyze NER, Translate, or Analyze & Translate</Text>
            </View>
          ) : null}

          {/* Translation */}
          {translationResult ? (
            <View style={styles.resultBlock}>
              <View style={styles.resultBlockHeader}>
                <Text style={styles.resultBlockTitle}>Translation</Text>
                <TouchableOpacity style={styles.copySmallBtn} onPress={() => copyText(translationResult, 'Translation copied!')}>
                  <Text style={styles.copySmallBtnText}>Copy</Text>
                </TouchableOpacity>
              </View>
              <View style={styles.translationBox}>
                <Text style={styles.translationText}>{translationResult}</Text>
              </View>
              {engineNote ? <Text style={styles.resultNote}>{`Translated with: ${engineNote}`}</Text> : null}
              {resultNote ? <Text style={styles.resultNote}>{resultNote}</Text> : null}
              {saved ? <Text style={styles.savedBadge}>Saved to database</Text> : null}
            </View>
          ) : null}

          {/* NER Results */}
          {nerResults.length > 0 ? (
            <View style={styles.resultBlock}>
              <View style={styles.resultBlockHeader}>
                <Text style={styles.resultBlockTitle}>Named Entities</Text>
                <View style={styles.nerCountBadge}>
                  <Text style={styles.nerCountText}>
                    {nerSummary ? `${nerSummary.unique.toLocaleString()} unique · ${nerSummary.total.toLocaleString()} total` : nerResults.length}
                  </Text>
                </View>
              </View>
              {nerSummary && nerSummary.unique > nerResults.length ? (
                <Text style={styles.resultNote}>
                  Showing the {nerResults.length} most frequent of {nerSummary.unique.toLocaleString()} unique entities. Export for the full list.
                </Text>
              ) : null}
              <View style={styles.legend}>
                {Object.entries(NER_COLORS).map(([k, v]) => (
                  <View key={k} style={styles.legendItem}>
                    <View style={[styles.legendDot, { backgroundColor: v.text }]} />
                    <Text style={styles.legendLabel}>{v.label}</Text>
                  </View>
                ))}
              </View>
              <View style={styles.nerList}>
                {nerResults.map((entity, index) => {
                  const color = getNerColor(entity.entity_group);
                  return (
                    <View key={index} style={[styles.nerTag, { backgroundColor: color.bg, borderColor: color.text }]}>
                      <Text style={[styles.nerWord, { color: color.text }]}>{entity.word}</Text>
                      <View style={[styles.nerBadge, { backgroundColor: color.text }]}>
                        <Text style={styles.nerBadgeText}>{color.label}</Text>
                      </View>
                      {entity.count > 1 ? <Text style={styles.nerCount}>{`×${entity.count}`}</Text> : null}
                      <Text style={styles.nerScore}>{(entity.score * 100).toFixed(0)}%</Text>
                    </View>
                  );
                })}
              </View>
              {saved && !translationResult ? <Text style={styles.savedBadge}>Saved to database</Text> : null}
            </View>
          ) : (
            !nerLoading && nerRan ? (
              <Text style={styles.noEntities}>No named entities detected.</Text>
            ) : null
          )}
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1 },
  container: {
    padding: 16,
    paddingBottom: 40,
    backgroundColor: '#f0f2f5',
    flexGrow: 1,
  },

  // Language banner
  langBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 12,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#e2e5ed',
  },
  langBannerChip: {
    alignItems: 'center',
    backgroundColor: '#f0f2f5',
    paddingVertical: 6,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  langBannerChipTarget: {
    backgroundColor: '#e8f0fe',
    marginLeft: 0,
  },
  langBannerLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: '#8b92a9',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  langBannerValue: {
    fontSize: 15,
    fontWeight: '700',
    color: '#1a1f36',
    marginTop: 2,
  },
  langBannerValueTarget: {
    color: '#7c3aed',
  },
  langBannerArrow: {
    fontSize: 18,
    color: '#c5cad8',
    fontWeight: '700',
    flex: 1,
    textAlign: 'center',
  },
  autoDetectedTag: {
    backgroundColor: '#f0fdf4',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#bbf7d0',
    marginLeft: 8,
  },
  autoDetectedTagText: {
    fontSize: 11,
    color: '#16a34a',
    fontWeight: '600',
  },

  // Two-col
  twoCol: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  col: {
    flex: 1,
    minWidth: 280,
    marginRight: 14,
    marginBottom: 0,
  },

  // Card
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#e2e5ed',
    overflow: 'visible' as any,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
    zIndex: 100,
    overflow: 'visible' as any,
  },
  cardTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: '#8b92a9',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  cardActions: {
    flexDirection: 'row',
  },
  iconBtn: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
    backgroundColor: '#F3F2FF',
    borderWidth: 1.5,
    borderColor: '#EDE9FE',
    marginLeft: 6,
  },
  iconBtnText: {
    fontSize: 12,
    color: '#7c3aed',
    fontWeight: '600',
  },
  iconBtnPrimary: {
    backgroundColor: '#EDE9FE',
    borderColor: '#c4b5fd',
  },
  iconBtnTextPrimary: {
    fontSize: 12,
    color: '#6d28d9',
    fontWeight: '700',
  },

  fileTag: {
    backgroundColor: '#f0fdf4',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#bbf7d0',
    alignSelf: 'flex-start',
  },
  fileTagText: {
    fontSize: 12,
    color: '#16a34a',
    fontWeight: '600',
  },

  input: {
    borderWidth: 1.5,
    borderColor: '#e2e5ed',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    backgroundColor: '#fafbfc',
    minHeight: 140,
    textAlignVertical: 'top',
    color: '#1a1f36',
    lineHeight: 22,
  },

  // Language selectors
  langSectionLabel: {
    fontSize: 10,
    fontWeight: '700',
    color: '#8b92a9',
    letterSpacing: 0.8,
    textTransform: 'uppercase',
    marginTop: 14,
    marginBottom: 8,
  },
  langScroll: {
    marginBottom: 4,
    // a horizontal ScrollView grows by default on web; without this it absorbs the card's extra height
    flexGrow: 0,
    flexShrink: 0,
  },
  langRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  langChip: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
    backgroundColor: '#f0f2f5',
    borderWidth: 1.5,
    borderColor: '#e2e5ed',
    marginRight: 6,
  },
  langChipActive: {
    backgroundColor: '#EDE9FE',
    borderColor: '#7c3aed',
  },
  langChipText: {
    fontSize: 12,
    color: '#555',
    fontWeight: '500',
  },
  langBadge: {
    fontSize: 10,
    fontWeight: '600',
    color: '#b45309',
  },
  langNote: {
    fontSize: 12,
    lineHeight: 17,
    color: '#92400e',
    backgroundColor: '#fffbeb',
    borderColor: '#fcd34d',
    borderWidth: 1,
    borderRadius: 8,
    padding: 8,
    marginTop: 8,
  },
  langChipTextActive: {
    color: '#7c3aed',
    fontWeight: '700',
  },

  // Buttons
  btnRow: {
    flexDirection: 'row',
    marginTop: 16,
  },
  actionBtn: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 13,
    paddingHorizontal: 6,
    alignItems: 'center',
    marginRight: 6,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 3,
  },
  nerBtn: {
    backgroundColor: '#2563eb',
    shadowColor: '#2563eb',
  },
  translateBtn: {
    backgroundColor: '#7c3aed',
    shadowColor: '#7c3aed',
  },
  bothBtn: {
    backgroundColor: '#4f46e5',
    shadowColor: '#4f46e5',
  },
  actionBtnText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '700',
    letterSpacing: 0.2,
    textAlign: 'center',
  },
  primaryBtn: {
    flex: 1,
    backgroundColor: '#7c3aed',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    marginRight: 8,
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
  },
  primaryBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
  ghostBtn: {
    backgroundColor: '#fff',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 18,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#EDE9FE',
  },
  ghostBtnText: {
    color: '#7c3aed',
    fontSize: 14,
    fontWeight: '600',
  },

  // Output section
  outputLangRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f8f9fc',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: '#e2e5ed',
  },
  outputLangText: {
    fontSize: 13,
    fontWeight: '700',
    color: '#555',
  },
  outputLangArrow: {
    flex: 1,
    textAlign: 'center',
    fontSize: 16,
    color: '#c5cad8',
  },
  outputLangTarget: {
    color: '#1a73e8',
  },

  emptyOutput: {
    minHeight: 220,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyOutputIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#f0f2f5',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  emptyOutputIconText: {
    fontSize: 20,
    fontWeight: '900',
    color: '#c5cad8',
  },
  emptyOutputTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#555',
    marginBottom: 6,
  },
  emptyOutputSub: {
    fontSize: 13,
    color: '#b0b8c9',
    textAlign: 'center',
    lineHeight: 20,
    maxWidth: 220,
  },

  resultBlock: {
    marginBottom: 16,
  },
  resultBlockHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultBlockTitle: {
    fontSize: 10,
    fontWeight: '700',
    color: '#8b92a9',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  copySmallBtn: {
    backgroundColor: '#f0f2f5',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#e2e5ed',
  },
  copySmallBtnText: {
    color: '#555',
    fontWeight: '600',
    fontSize: 11,
  },
  translationBox: {
    backgroundColor: '#EDE9FE',
    borderRadius: 10,
    padding: 14,
    borderLeftWidth: 3,
    borderLeftColor: '#7c3aed',
  },
  translationText: {
    fontSize: 15,
    color: '#1a1f36',
    lineHeight: 24,
  },
  savedBadge: {
    marginTop: 8,
    fontSize: 11,
    color: '#16a34a',
    fontWeight: '700',
  },

  // NER
  nerCountBadge: {
    backgroundColor: '#7c3aed',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  btnDisabled: { opacity: 0.5 },

  // Uploaded document card
  docCard: {
    borderWidth: 1.5, borderColor: '#c7d7f5', backgroundColor: '#f5f8ff', borderRadius: 10, padding: 14,
  },
  docHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  docName: { fontSize: 14, fontWeight: '700', color: '#1a1f36', flex: 1, marginRight: 10 },
  docRemove: { fontSize: 12, fontWeight: '700', color: '#dc2626' },
  docMeta: { fontSize: 12, color: '#6b7280', marginTop: 4 },
  docPreviewLabel: {
    fontSize: 10, fontWeight: '700', color: '#8b92a9', textTransform: 'uppercase', letterSpacing: 0.8, marginTop: 12, marginBottom: 4,
  },
  docPreview: { fontSize: 13, color: '#374151', lineHeight: 19 },
  docHint: { fontSize: 11, color: '#8b92a9', marginTop: 10 },

  // Progress
  progressWrap: { marginTop: 14 },
  progressTrack: { height: 8, borderRadius: 4, backgroundColor: '#EDE9FE', overflow: 'hidden', marginTop: 8 },
  progressFill: { height: 8, borderRadius: 4, backgroundColor: '#7c3aed' },
  progressRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 6 },
  progressText: { fontSize: 12, color: '#6d28d9', fontWeight: '600', flex: 1 },
  progressCancel: { fontSize: 12, color: '#ec4899', fontWeight: '700' },

  resultNote: { fontSize: 12, color: '#6b7280', marginTop: 8, lineHeight: 17 },
  warningBox: { backgroundColor: '#fffbeb', borderWidth: 1, borderColor: '#fde68a', borderRadius: 8, padding: 10, marginBottom: 14 },
  warningBoxText: { fontSize: 12, color: '#92400e', lineHeight: 18 },
  nerCount: { fontSize: 11, color: '#374151', fontWeight: '700', marginRight: 8 },

  nerCountText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
  },
  legend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 10,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
    marginBottom: 4,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 4,
  },
  legendLabel: {
    fontSize: 11,
    color: '#8b92a9',
    fontWeight: '500',
  },
  nerList: {},
  nerTag: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 6,
  },
  nerWord: {
    fontSize: 14,
    fontWeight: '700',
    flex: 1,
  },
  nerBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 12,
    marginRight: 8,
  },
  nerBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  nerScore: {
    fontSize: 11,
    color: '#8b92a9',
    fontWeight: '600',
  },
  noEntities: {
    fontSize: 13,
    color: '#b0b8c9',
    fontStyle: 'italic',
    textAlign: 'center',
    marginTop: 16,
  },

  // Export dropdown
  exportWrap: {
    position: 'relative',
    zIndex: 1000,
    overflow: 'visible' as any,
  },
  exportDropBtn: {
    backgroundColor: '#7c3aed',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 10,
  },
  exportDropBtnText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 13,
  },
  exportMenu: {
    position: 'absolute',
    top: 36,
    right: 0,
    backgroundColor: '#fff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e2e5ed',
    zIndex: 99999,
    minWidth: 130,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 16,
    elevation: 999,
  },
  exportMenuItem: {
    paddingVertical: 11,
    paddingHorizontal: 18,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f2f5',
  },
  exportMenuText: {
    fontSize: 13,
    color: '#1a1f36',
    fontWeight: '600',
  },

  // Toast
  toast: {
    position: 'absolute',
    top: 10,
    left: 16,
    right: 16,
    zIndex: 9999,
    padding: 13,
    borderRadius: 10,
    alignItems: 'center',
  },
  toastText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 13,
  },
});
