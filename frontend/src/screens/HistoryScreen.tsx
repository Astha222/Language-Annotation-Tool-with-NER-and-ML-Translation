import React, { useState, useCallback, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  ScrollView, ActivityIndicator, Alert, Platform
} from 'react-native';
import axios from 'axios';
import * as XLSX from 'xlsx';
import { useAuth } from '../context/AuthContext';
import { exportPdf } from '../utils/pdfExport';
import { parseEntities, entitiesToJson, entitiesToText } from '../utils/entities';
import { makeSheet } from '../utils/xlsxSafe';

import { BACKEND_URL } from '../config';
import { LANGUAGE_NAMES } from '../utils/languages';

const NER_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  PER:  { bg: '#e3f2fd', text: '#1565c0', label: 'Person' },
  ORG:  { bg: '#e8f5e9', text: '#2e7d32', label: 'Organization' },
  LOC:  { bg: '#fff3e0', text: '#e65100', label: 'Location' },
  MISC: { bg: '#f3e5f5', text: '#6a1b9a', label: 'Miscellaneous' },
  DATE: { bg: '#e0f7fa', text: '#00695c', label: 'Date' },
};

const LANG_NAMES: Record<string, string> = LANGUAGE_NAMES;

function Toast({ message, type }: { message: string; type: 'success' | 'error' | 'info' }) {
  const colors = { success: '#2e7d32', error: '#c62828', info: '#1565c0' };
  return (
    <View style={[styles.toast, { backgroundColor: colors[type] }]}>
      <Text style={styles.toastText}>{message}</Text>
    </View>
  );
}

interface Annotation {
  id: number;
  original_text: string;
  translated_text: string | null;
  ner_tags: string | null;
  source_lang: string;
  target_lang: string;
  source_name?: string | null;   // uploaded file name, if the annotation came from a document
  truncated?: boolean;           // the list view clipped a very long document
}

const MAX_ENTITIES_SHOWN = 200;

// ----------------------------------------------------------------
// Export helpers
// ----------------------------------------------------------------
function triggerDownload(blob: Blob, filename: string) {
  if (typeof document === 'undefined') return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// readable=true lists entities one per line (txt/pdf); otherwise a clean JSON string (json/csv/xlsx/xml)
function buildRows(annotations: Annotation[], readable = false) {
  return annotations.map((a) => ({
    id: String(a.id),
    source_name: a.source_name ?? '',
    original_text: a.original_text,
    translated_text: a.translated_text ?? '',
    source_lang: LANG_NAMES[a.source_lang] || a.source_lang,
    target_lang: LANG_NAMES[a.target_lang] || a.target_lang,
    ner_tags: readable ? entitiesToText(a.ner_tags) : entitiesToJson(a.ner_tags),
  }));
}

function exportJSON(annotations: Annotation[]) {
  const blob = new Blob([JSON.stringify(buildRows(annotations), null, 2)], { type: 'application/json' });
  triggerDownload(blob, `annotations_${new Date().toISOString().slice(0, 10)}.json`);
}

function exportCSV(annotations: Annotation[]) {
  const rows = buildRows(annotations);
  if (!rows.length) return;
  const headers = Object.keys(rows[0]).join(',');
  const lines = rows.map((r) =>
    Object.values(r).map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',')
  );
  const blob = new Blob([[headers, ...lines].join('\n')], { type: 'text/csv' });
  triggerDownload(blob, `annotations_${new Date().toISOString().slice(0, 10)}.csv`);
}

function exportExcel(annotations: Annotation[]) {
  const ws = makeSheet(buildRows(annotations));
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Annotations');
  XLSX.writeFile(wb, `annotations_${new Date().toISOString().slice(0, 10)}.xlsx`);
}

function exportTXT(annotations: Annotation[]) {
  const lines = buildRows(annotations, true).map((r, i) =>
    `--- Record ${i + 1} ---\n` + Object.entries(r).map(([k, v]) => `${k}: ${v}`).join('\n')
  );
  const blob = new Blob([lines.join('\n\n')], { type: 'text/plain' });
  triggerDownload(blob, `annotations_${new Date().toISOString().slice(0, 10)}.txt`);
}

function exportXML(annotations: Annotation[]) {
  const items = buildRows(annotations).map((r) => {
    const fields = Object.entries(r)
      .map(([k, v]) =>
        `    <${k}>${String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</${k}>`
      )
      .join('\n');
    return `  <annotation>\n${fields}\n  </annotation>`;
  });
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<annotations>\n${items.join('\n')}\n</annotations>`;
  const blob = new Blob([xml], { type: 'application/xml' });
  triggerDownload(blob, `annotations_${new Date().toISOString().slice(0, 10)}.xml`);
}

function exportPDF(annotations: Annotation[]): boolean {
  return exportPdf('Annotations', buildRows(annotations, true));
}

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

// ----------------------------------------------------------------
// Main Component
// ----------------------------------------------------------------
export default function HistoryScreen() {
  const { token } = useAuth();
  const authHeaders = () => ({ headers: { Authorization: `Bearer ${token}` } });

  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [loading, setLoading]         = useState(false);
  const [expanded, setExpanded]       = useState<number | null>(null);
  const [toast, setToast]             = useState<{ msg: string; type: 'success' | 'error' | 'info' } | null>(null);
  const [showExportMenu, setShowExportMenu] = useState(false);

  const showToast = (msg: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const fetchAnnotations = useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${BACKEND_URL}/api/annotations`, authHeaders());
      setAnnotations(res.data);
    } catch {
      showToast('Failed to load annotations', 'error');
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    fetchAnnotations();
  }, [fetchAnnotations]);

  const deleteAnnotation = async (id: number) => {
    try {
      await axios.delete(`${BACKEND_URL}/api/annotations/${id}`, authHeaders());
      setAnnotations(prev => prev.filter(a => a.id !== id));
      if (expanded === id) setExpanded(null);
      showToast('Annotation deleted', 'success');
    } catch {
      showToast('Failed to delete annotation', 'error');
    }
  };

  const handleDelete = (id: number) => {
    const message = 'Are you sure you want to delete this annotation? This cannot be undone.';
    // Alert.alert with buttons is a no-op on react-native-web, so use window.confirm there
    if (Platform.OS === 'web') {
      if (typeof window !== 'undefined' && window.confirm(message)) deleteAnnotation(id);
      return;
    }
    Alert.alert('Delete Annotation', message, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => deleteAnnotation(id) },
    ]);
  };

  const handleExport = async (format: string) => {
    setShowExportMenu(false);

    if (format === 'conll' || format === 'jsonl') {
      try {
        const res = await axios.get(`${BACKEND_URL}/api/export?format=${format}`, {
          ...authHeaders(),
          responseType: 'text',
        });
        if (!res.data || !res.data.trim()) {
          showToast('No annotations to export', 'error');
          return;
        }
        const mimeType = format === 'jsonl' ? 'application/x-ndjson' : 'text/plain';
        const ext = format === 'conll' ? 'conll' : 'jsonl';
        triggerDownload(new Blob([res.data], { type: mimeType }), `annotations_${new Date().toISOString().slice(0, 10)}.${ext}`);
        showToast(`Exported annotations as ${format === 'conll' ? 'CoNLL / BIO' : 'JSONL'}`, 'success');
      } catch {
        showToast(`Failed to export as ${format}`, 'error');
      }
      return;
    }

    // Export every saved annotation from the server (the list view only holds the first page)
    let all: Annotation[];
    try {
      const res = await axios.get(`${BACKEND_URL}/api/export`, authHeaders());
      all = res.data.map((r: any) => ({
        id: r.id,
        original_text: r.original_text,
        translated_text: r.translated_text ?? null,
        // /api/export returns ner_tags as an array; the row builders expect a JSON string
        ner_tags: Array.isArray(r.ner_tags) && r.ner_tags.length ? JSON.stringify(r.ner_tags) : null,
        source_lang: r.source_lang,
        target_lang: r.target_lang,
        source_name: r.source_name ?? null,
      }));
    } catch {
      showToast('Failed to fetch annotations for export', 'error');
      return;
    }

    if (!all.length) {
      showToast('No annotations to export', 'error');
      return;
    }
    switch (format) {
      case 'json':  exportJSON(all);  break;
      case 'csv':   exportCSV(all);   break;
      case 'xlsx':  exportExcel(all); break;
      case 'txt':   exportTXT(all);   break;
      case 'xml':   exportXML(all);   break;
      case 'pdf':
        if (!exportPDF(all)) {
          showToast('Allow pop-ups to export as PDF', 'error');
          return;
        }
        break;
    }
    showToast(`Exported ${all.length} annotation${all.length !== 1 ? 's' : ''} as .${format}`, 'success');
  };

  const getNerColor = (entityGroup: string) =>
    NER_COLORS[entityGroup] || { bg: '#fce4ec', text: '#880e4f', label: entityGroup };

  const parseNerTags = (nerTags: string | null) => parseEntities(nerTags);

  return (
    <View style={styles.root}>
      {/* Action Bar */}
      <View style={styles.actionBar}>
        <TouchableOpacity style={styles.refreshBtn} onPress={fetchAnnotations}>
          <Text style={styles.refreshBtnText}>Load History</Text>
        </TouchableOpacity>

        {/* Export Dropdown */}
        <View style={styles.exportWrapper}>
          <TouchableOpacity
            style={styles.exportBtn}
            onPress={() => setShowExportMenu(!showExportMenu)}
          >
            <Text style={styles.exportBtnText}>{`Export  ${showExportMenu ? '▲' : '▼'}`}</Text>
          </TouchableOpacity>
          {showExportMenu && (
            <View style={styles.exportMenu}>
              {EXPORT_OPTIONS.map((opt) => (
                <TouchableOpacity
                  key={opt.key}
                  style={styles.exportMenuItem}
                  onPress={() => handleExport(opt.key)}
                >
                  <Text style={styles.exportMenuText}>{opt.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>
      </View>

      {/* Status / Success Toast Message */}
      {toast && <Toast message={toast.msg} type={toast.type} />}

      {/* Count */}
      {annotations.length > 0 && (
        <Text style={styles.countText}>
          {annotations.length} annotation{annotations.length !== 1 ? 's' : ''} found
        </Text>
      )}

      {loading
        ? <ActivityIndicator size="large" color="#1a73e8" style={{ marginTop: 40 }} />
        : annotations.length === 0
          ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyTitle}>No annotations yet</Text>
              <Text style={styles.emptySubtitle}>
                Go to the Annotate tab, enter text and analyze it. Your saved annotations will appear here.
              </Text>
              <TouchableOpacity style={styles.refreshBtn} onPress={fetchAnnotations}>
                <Text style={styles.refreshBtnText}>Refresh</Text>
              </TouchableOpacity>
            </View>
          )
          : (
            <ScrollView contentContainerStyle={styles.list}>
              {annotations.map((item) => {
                const isExpanded = expanded === item.id;
                const nerTags = parseNerTags(item.ner_tags);

                return (
                  <View key={item.id} style={styles.card}>
                    {/* Card Header */}
                    <TouchableOpacity onPress={() => setExpanded(isExpanded ? null : item.id)}>
                      <View style={styles.cardHeader}>
                        <View style={styles.cardMeta}>
                          <Text style={styles.cardId}>#{item.id}</Text>
                          <View style={styles.langBadges}>
                            <Text style={styles.langBadge}>{LANG_NAMES[item.source_lang] || item.source_lang}</Text>
                            <Text style={styles.arrow}>→</Text>
                            <Text style={styles.langBadge}>{LANG_NAMES[item.target_lang] || item.target_lang}</Text>
                          </View>
                          {item.source_name ? (
                            <Text style={styles.fileBadge} numberOfLines={1}>{item.source_name}</Text>
                          ) : null}
                        </View>
                        <Text style={styles.expandIcon}>{isExpanded ? '▲' : '▼'}</Text>
                      </View>

                      <Text style={styles.originalText} numberOfLines={isExpanded ? undefined : 2}>
                        {item.original_text}
                      </Text>
                    </TouchableOpacity>

                    {isExpanded && (
                      <View style={styles.expandedContent}>
                        {item.truncated ? (
                          <Text style={styles.truncNote}>
                            This is a long document, so only the beginning is shown here. Use Export to get the complete text and every entity.
                          </Text>
                        ) : null}

                        {/* Translation */}
                        {item.translated_text ? (
                          <View style={styles.section}>
                            <Text style={styles.sectionLabel}>Translation</Text>
                            <Text style={styles.translationText}>{item.translated_text}</Text>
                          </View>
                        ) : null}

                        {/* NER Tags */}
                        {nerTags.length > 0 ? (
                          <View style={styles.section}>
                            <Text style={styles.sectionLabel}>
                              Entities ({nerTags.length > MAX_ENTITIES_SHOWN ? `first ${MAX_ENTITIES_SHOWN} of ${nerTags.length}` : nerTags.length}{item.truncated ? '+' : ''})
                            </Text>
                            <View style={styles.nerGrid}>
                              {nerTags.slice(0, MAX_ENTITIES_SHOWN).map((entity: any, i: number) => {
                                const color = getNerColor(entity.entity_group);
                                return (
                                  <View key={i} style={[styles.nerTag, { backgroundColor: color.bg, borderColor: color.text }]}>
                                    <Text style={[styles.nerWord, { color: color.text }]}>{entity.word}</Text>
                                    <View style={[styles.nerBadge, { backgroundColor: color.text }]}>
                                      <Text style={styles.nerBadgeText}>{color.label}</Text>
                                    </View>
                                  </View>
                                );
                              })}
                            </View>
                          </View>
                        ) : (
                          <Text style={styles.noEntities}>No named entities found</Text>
                        )}

                        {/* Delete Button */}
                        <TouchableOpacity style={styles.deleteBtn} onPress={() => handleDelete(item.id)}>
                          <Text style={styles.deleteBtnText}>Delete Annotation</Text>
                        </TouchableOpacity>
                      </View>
                    )}
                  </View>
                );
              })}
            </ScrollView>
          )
      }
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#f5f7fa',
    overflow: 'visible' as any,
  },
  actionBar: {
    flexDirection: 'row',
    padding: 12,
    zIndex: 1000,
    overflow: 'visible' as any,
  },
  refreshBtn: {
    flex: 1,
    backgroundColor: '#1a73e8',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
    marginRight: 10,
  },
  refreshBtnText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
  },
  exportWrapper: {
    flex: 1,
    position: 'relative',
    zIndex: 1000,
    overflow: 'visible' as any,
  },
  exportBtn: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#1a73e8',
  },
  exportBtnText: {
    color: '#1a73e8',
    fontWeight: '700',
    fontSize: 14,
  },
  exportMenu: {
    position: 'absolute',
    top: 50,
    right: 0,
    left: 0,
    backgroundColor: '#fff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 16,
    elevation: 999,
    zIndex: 99999,
  },
  exportMenuItem: {
    paddingVertical: 13,
    paddingHorizontal: 18,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  exportMenuText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '600',
  },
  countText: {
    fontSize: 13,
    color: '#888',
    textAlign: 'center',
    marginBottom: 6,
  },
  list: {
    padding: 12,
    paddingTop: 4,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  cardMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
  },
  cardId: {
    fontSize: 12,
    color: '#888',
    fontWeight: '600',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
  },
  fileBadge: {
    fontSize: 11,
    color: '#1a73e8',
    fontWeight: '600',
    backgroundColor: '#e8f0fe',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
    maxWidth: 220,
  },
  truncNote: {
    fontSize: 12,
    color: '#b45309',
    backgroundColor: '#fffbeb',
    padding: 8,
    borderRadius: 6,
    marginBottom: 10,
    lineHeight: 17,
  },
  langBadges: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  langBadge: {
    fontSize: 12,
    color: '#555',
    fontWeight: '500',
  },
  arrow: {
    fontSize: 12,
    color: '#aaa',
  },
  expandIcon: {
    fontSize: 12,
    color: '#aaa',
  },
  originalText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  expandedContent: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  section: {
    marginBottom: 12,
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '700',
    color: '#555',
    marginBottom: 6,
  },
  translationText: {
    fontSize: 14,
    color: '#1a73e8',
    lineHeight: 22,
    backgroundColor: '#e8f0fe',
    padding: 10,
    borderRadius: 8,
  },
  nerGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  nerTag: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    marginRight: 6,
    marginBottom: 4,
  },
  nerWord: {
    fontSize: 13,
    fontWeight: '700',
    marginRight: 6,
  },
  nerBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 10,
  },
  nerBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  noEntities: {
    fontSize: 13,
    color: '#aaa',
    fontStyle: 'italic',
    marginBottom: 10,
  },
  deleteBtn: {
    backgroundColor: '#ffebee',
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ef9a9a',
    marginTop: 4,
  },
  deleteBtnText: {
    color: '#c62828',
    fontWeight: '700',
    fontSize: 14,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
    marginTop: 60,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#333',
    marginBottom: 10,
  },
  emptySubtitle: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 24,
  },
  toast: {
    marginHorizontal: 12,
    marginBottom: 10,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toastText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
    textAlign: 'center',
  },
});
