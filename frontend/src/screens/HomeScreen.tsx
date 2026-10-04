import React, { useState, useCallback } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ActivityIndicator, ScrollView, Platform,
} from 'react-native';
import axios from 'axios';
import * as XLSX from 'xlsx';
import { useAuth } from '../context/AuthContext';

const BACKEND_URL = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

const NER_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  PER:  { bg: '#e3f2fd', text: '#1565c0', label: 'Person' },
  ORG:  { bg: '#e8f5e9', text: '#2e7d32', label: 'Organization' },
  LOC:  { bg: '#fff3e0', text: '#e65100', label: 'Location' },
  MISC: { bg: '#f3e5f5', text: '#6a1b9a', label: 'Miscellaneous' },
};

const SOURCE_LANGUAGES = [
  { code: 'auto', name: 'Auto Detect' },
  { code: 'en',   name: 'English' },
  { code: 'hi',   name: 'Hindi' },
  { code: 'fr',   name: 'French' },
  { code: 'es',   name: 'Spanish' },
  { code: 'de',   name: 'German' },
  { code: 'zh',   name: 'Chinese' },
  { code: 'ar',   name: 'Arabic' },
  { code: 'ja',   name: 'Japanese' },
  { code: 'ru',   name: 'Russian' },
];

const TARGET_LANGUAGES = SOURCE_LANGUAGES.filter((l) => l.code !== 'auto');

const LANG_NAME: Record<string, string> = Object.fromEntries(SOURCE_LANGUAGES.map((l) => [l.code, l.name]));

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
    const ws = XLSX.utils.json_to_sheet([row]);
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
  pdf: (row) => {
    const lines = Object.entries(row).map(([k, v]) => `${k}: ${v}`);
    const body = lines.map((l, i) =>
      `BT /F1 10 Tf 40 ${720 - i * 16} Td (${l.replace(/\\/g,'\\\\').replace(/\(/g,'\\(').replace(/\)/g,'\\)')}) Tj ET`
    ).join('\n');
    const pdf = `%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj\n4 0 obj<</Length ${body.length}>>\nstream\n${body}\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n0\n%%EOF`;
    triggerDownload(new Blob([pdf], { type: 'application/pdf' }), `annotation_${Date.now()}.pdf`);
  },
};

const EXPORT_FORMATS = ['json', 'csv', 'xlsx', 'txt', 'xml', 'pdf'];

// ── Main Component ─────────────────────────────────────────────
export default function HomeScreen() {
  const { token } = useAuth();
  const authHeaders = () => ({ headers: { Authorization: `Bearer ${token}` } });

  const [inputText, setInputText]               = useState('');
  const [sourceLang, setSourceLang]             = useState('auto');
  const [targetLang, setTargetLang]             = useState('fr');
  const [loading, setLoading]                   = useState(false);
  const [nerResults, setNerResults]             = useState<any[]>([]);
  const [translationResult, setTranslationResult] = useState('');
  const [detectedLang, setDetectedLang]         = useState('');
  const [toast, setToast]                       = useState<{ msg: string; type: 'success' | 'error' | 'info' } | null>(null);
  const [saved, setSaved]                       = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [showExportMenu, setShowExportMenu]     = useState(false);

  const showToast = (msg: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  // Programmatic file input — avoids DOM nesting error in React Native Web
  const triggerFileInput = useCallback(() => {
    if (typeof document === 'undefined') return;
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = '.txt,.csv,.json,.xml,.md';
    inp.onchange = (e: any) => {
      const file: File = e.target?.files?.[0];
      if (!file) return;
      setUploadedFileName(file.name);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setInputText(evt.target?.result as string || '');
        showToast(`"${file.name}" loaded`, 'success');
      };
      reader.readAsText(file);
    };
    inp.click();
  }, []);

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

  const handleAnalyzeAndTranslate = async () => {
    if (!inputText.trim()) { showToast('Please enter some text first', 'error'); return; }
    setLoading(true);
    setNerResults([]); setTranslationResult(''); setDetectedLang(''); setSaved(false);

    let srcLang = sourceLang === 'auto' ? 'en' : sourceLang;
    let nerEntities: any[] = [];
    let translatedText: string | null = null;

    try {
      if (sourceLang === 'auto') {
        try {
          const r = await axios.post(`${BACKEND_URL}/api/detect-language`, { text: inputText }, authHeaders());
          srcLang = r.data.detected_language || 'en';
        } catch { srcLang = 'en'; }
      }
      setDetectedLang(srcLang);

      const nerRes = await axios.post(`${BACKEND_URL}/api/ner`, { text: inputText }, authHeaders());
      nerEntities = nerRes.data.entities;
      setNerResults(nerEntities);

      try {
        const transRes = await axios.post(`${BACKEND_URL}/api/translate`, {
          text: inputText, source_lang: srcLang, target_lang: targetLang,
        }, authHeaders());
        translatedText = transRes.data.translated_text;
        setTranslationResult(translatedText ?? '');
      } catch (transErr: any) {
        showToast(transErr?.response?.data?.detail || `Translation to "${targetLang}" not available`, 'info');
      }

      await axios.post(`${BACKEND_URL}/api/annotations`, {
        original_text: inputText, translated_text: translatedText,
        ner_tags: JSON.stringify(nerEntities), source_lang: srcLang, target_lang: targetLang,
      }, authHeaders());
      setSaved(true);
      showToast(translatedText ? 'Annotation saved!' : 'Annotation saved (no translation)', 'success');
    } catch (err: any) {
      showToast(err?.response?.data?.detail || 'Error connecting to backend', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInputText(''); setNerResults([]); setTranslationResult('');
    setDetectedLang(''); setSaved(false); setUploadedFileName('');
  };

  const handleExport = (fmt: string) => {
    setShowExportMenu(false);
    if (!inputText && !translationResult) { showToast('Nothing to export yet', 'error'); return; }
    const row = buildExportRow(inputText, translationResult, nerResults, detectedLang || sourceLang, targetLang);
    doExport[fmt]?.(row);
    showToast(`Exported as .${fmt}`, 'success');
  };

  const getNerColor = (eg: string) => NER_COLORS[eg] || { bg: '#fce4ec', text: '#880e4f', label: eg };
  const hasResults = !!(translationResult || nerResults.length > 0);
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
              <TouchableOpacity style={styles.iconBtn} onPress={() => copyText(inputText, 'Input copied!')}>
                <Text style={styles.iconBtnText}>Copy</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.iconBtn} onPress={handleSaveInput}>
                <Text style={styles.iconBtnText}>Save</Text>
              </TouchableOpacity>
              {Platform.OS === 'web' ? (
                <TouchableOpacity style={[styles.iconBtn, styles.iconBtnPrimary]} onPress={triggerFileInput}>
                  <Text style={styles.iconBtnTextPrimary}>Upload File</Text>
                </TouchableOpacity>
              ) : null}
            </View>
          </View>

          {uploadedFileName ? (
            <View style={styles.fileTag}>
              <Text style={styles.fileTagText}>{uploadedFileName}</Text>
            </View>
          ) : null}

          <TextInput
            style={styles.input}
            multiline
            numberOfLines={8}
            placeholder="Type, paste, or upload a file..."
            placeholderTextColor="#b0b8c9"
            value={inputText}
            onChangeText={setInputText}
          />

          {/* Source Language */}
          <Text style={styles.langSectionLabel}>Source Language</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.langScroll}>
            <View style={styles.langRow}>
              {SOURCE_LANGUAGES.map((lang) => (
                <TouchableOpacity
                  key={lang.code}
                  style={[styles.langChip, sourceLang === lang.code && styles.langChipActive]}
                  onPress={() => setSourceLang(lang.code)}
                >
                  <Text style={[styles.langChipText, sourceLang === lang.code && styles.langChipTextActive]}>
                    {lang.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </ScrollView>

          {/* Target Language */}
          <Text style={styles.langSectionLabel}>Target Language</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.langScroll}>
            <View style={styles.langRow}>
              {TARGET_LANGUAGES.map((lang) => (
                <TouchableOpacity
                  key={lang.code}
                  style={[styles.langChip, targetLang === lang.code && styles.langChipActive]}
                  onPress={() => setTargetLang(lang.code)}
                >
                  <Text style={[styles.langChipText, targetLang === lang.code && styles.langChipTextActive]}>
                    {lang.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </ScrollView>

          {/* Action Buttons */}
          <View style={styles.btnRow}>
            <TouchableOpacity style={styles.primaryBtn} onPress={handleAnalyzeAndTranslate} disabled={loading}>
              <Text style={styles.primaryBtnText}>{loading ? 'Analyzing...' : 'Analyze & Translate'}</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.ghostBtn} onPress={handleClear}>
              <Text style={styles.ghostBtnText}>Clear</Text>
            </TouchableOpacity>
          </View>

          {loading ? <ActivityIndicator size="large" color="#1a73e8" style={{ marginTop: 14 }} /> : null}
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
                    {EXPORT_FORMATS.map((fmt) => (
                      <TouchableOpacity key={fmt} style={styles.exportMenuItem} onPress={() => handleExport(fmt)}>
                        <Text style={styles.exportMenuText}>{`.${fmt.toUpperCase()}`}</Text>
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
              <Text style={styles.outputLangArrow}>{'→'}</Text>
              <Text style={[styles.outputLangText, styles.outputLangTarget]}>{LANG_NAME[targetLang] || targetLang}</Text>
            </View>
          ) : null}

          {!hasResults && !loading ? (
            <View style={styles.emptyOutput}>
              <View style={styles.emptyOutputIcon}>
                <Text style={styles.emptyOutputIconText}>O</Text>
              </View>
              <Text style={styles.emptyOutputTitle}>No output yet</Text>
              <Text style={styles.emptyOutputSub}>Enter text on the left and click Analyze & Translate</Text>
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
              {saved ? <Text style={styles.savedBadge}>Saved to database</Text> : null}
            </View>
          ) : null}

          {/* NER Results */}
          {nerResults.length > 0 ? (
            <View style={styles.resultBlock}>
              <View style={styles.resultBlockHeader}>
                <Text style={styles.resultBlockTitle}>Named Entities</Text>
                <View style={styles.nerCountBadge}>
                  <Text style={styles.nerCountText}>{nerResults.length}</Text>
                </View>
              </View>
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
                      <Text style={styles.nerScore}>{(entity.score * 100).toFixed(0)}%</Text>
                    </View>
                  );
                })}
              </View>
            </View>
          ) : (
            !loading && translationResult ? (
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
    color: '#1a73e8',
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
    borderRadius: 6,
    backgroundColor: '#f0f2f5',
    borderWidth: 1,
    borderColor: '#e2e5ed',
    marginLeft: 6,
  },
  iconBtnText: {
    fontSize: 12,
    color: '#555',
    fontWeight: '600',
  },
  iconBtnPrimary: {
    backgroundColor: '#e8f0fe',
    borderColor: '#1a73e8',
  },
  iconBtnTextPrimary: {
    fontSize: 12,
    color: '#1a73e8',
    fontWeight: '600',
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
  },
  langRow: {
    flexDirection: 'row',
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
    backgroundColor: '#e8f0fe',
    borderColor: '#1a73e8',
  },
  langChipText: {
    fontSize: 12,
    color: '#555',
    fontWeight: '500',
  },
  langChipTextActive: {
    color: '#1a73e8',
    fontWeight: '700',
  },

  // Buttons
  btnRow: {
    flexDirection: 'row',
    marginTop: 16,
  },
  primaryBtn: {
    flex: 1,
    backgroundColor: '#1a73e8',
    borderRadius: 10,
    paddingVertical: 13,
    alignItems: 'center',
    marginRight: 8,
  },
  primaryBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '700',
  },
  ghostBtn: {
    backgroundColor: '#fff',
    borderRadius: 10,
    paddingVertical: 13,
    paddingHorizontal: 18,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#e2e5ed',
  },
  ghostBtnText: {
    color: '#555',
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
    backgroundColor: '#f0f4ff',
    borderRadius: 8,
    padding: 14,
    borderLeftWidth: 3,
    borderLeftColor: '#1a73e8',
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
    backgroundColor: '#1a73e8',
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
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
    backgroundColor: '#1a73e8',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 8,
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
