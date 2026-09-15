import React, { useState, useCallback, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  ScrollView, ActivityIndicator, Alert, Platform
} from 'react-native';
import axios from 'axios';

const BACKEND_URL = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

const NER_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  PER:  { bg: '#e3f2fd', text: '#1565c0', label: 'Person' },
  ORG:  { bg: '#e8f5e9', text: '#2e7d32', label: 'Organization' },
  LOC:  { bg: '#fff3e0', text: '#e65100', label: 'Location' },
  MISC: { bg: '#f3e5f5', text: '#6a1b9a', label: 'Miscellaneous' },
};

const LANG_NAMES: Record<string, string> = {
  en: '🇬🇧 English', hi: '🇮🇳 Hindi', fr: '🇫🇷 French',
  es: '🇪🇸 Spanish', de: '🇩🇪 German', zh: '🇨🇳 Chinese',
  ar: '🇸🇦 Arabic', ja: '🇯🇵 Japanese', ru: '🇷🇺 Russian',
};

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
}

export default function HistoryScreen() {
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [loading, setLoading]         = useState(false);
  const [expanded, setExpanded]       = useState<number | null>(null);
  const [toast, setToast]             = useState<{ msg: string; type: 'success' | 'error' | 'info' } | null>(null);

  const showToast = (msg: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const fetchAnnotations = useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${BACKEND_URL}/api/annotations`);
      setAnnotations(res.data);
    } catch {
      showToast('Failed to load annotations', 'error');
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-load when screen mounts
  useEffect(() => {
    fetchAnnotations();
  }, [fetchAnnotations]);

  const handleDelete = (id: number) => {
    Alert.alert(
      'Delete Annotation',
      'Are you sure you want to delete this annotation? This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete', style: 'destructive',
          onPress: async () => {
            try {
              await axios.delete(`${BACKEND_URL}/api/annotations/${id}`);
              setAnnotations(prev => prev.filter(a => a.id !== id));
              if (expanded === id) setExpanded(null);
              showToast('Annotation deleted', 'success');
            } catch {
              showToast('Failed to delete annotation', 'error');
            }
          },
        },
      ]
    );
  };

  const handleExportJSON = async () => {
    try {
      const res = await axios.get(`${BACKEND_URL}/api/export`);
      const jsonStr = JSON.stringify(res.data, null, 2);

      if (typeof document !== 'undefined') {
        // Web: trigger download
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `annotations_${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('✅ Export downloaded!', 'success');
      } else {
        showToast('Export only available on web', 'info');
      }
    } catch {
      showToast('Export failed', 'error');
    }
  };

  const getNerColor = (entityGroup: string) =>
    NER_COLORS[entityGroup] || { bg: '#fce4ec', text: '#880e4f', label: entityGroup };

  const parseNerTags = (nerTags: string | null) => {
    if (!nerTags) return [];
    try { return JSON.parse(nerTags); } catch { return []; }
  };

  return (
    <View style={styles.root}>
      {toast && <Toast message={toast.msg} type={toast.type} />}

      {/* Action Bar */}
      <View style={styles.actionBar}>
        <TouchableOpacity style={styles.refreshBtn} onPress={fetchAnnotations}>
          <Text style={styles.refreshBtnText}>🔄 Load History</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.exportBtn} onPress={handleExportJSON}>
          <Text style={styles.exportBtnText}>⬇️ Export JSON</Text>
        </TouchableOpacity>
      </View>

      {/* Count */}
      {annotations.length > 0 && (
        <Text style={styles.countText}>{annotations.length} annotation{annotations.length !== 1 ? 's' : ''} found</Text>
      )}

      {loading
        ? <ActivityIndicator size="large" color="#1a73e8" style={{ marginTop: 40 }} />
        : annotations.length === 0
          ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyIcon}>📭</Text>
              <Text style={styles.emptyTitle}>No annotations yet</Text>
              <Text style={styles.emptySubtitle}>Go to the Annotate tab, enter text and analyze it. Your saved annotations will appear here.</Text>
              <TouchableOpacity style={styles.refreshBtn} onPress={fetchAnnotations}>
                <Text style={styles.refreshBtnText}>🔄 Refresh</Text>
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
                        </View>
                        <Text style={styles.expandIcon}>{isExpanded ? '▲' : '▼'}</Text>
                      </View>

                      <Text style={styles.originalText} numberOfLines={isExpanded ? undefined : 2}>
                        {item.original_text}
                      </Text>
                    </TouchableOpacity>

                    {isExpanded && (
                      <View style={styles.expandedContent}>
                        {/* Translation */}
                        {item.translated_text ? (
                          <View style={styles.section}>
                            <Text style={styles.sectionLabel}>🌐 Translation</Text>
                            <Text style={styles.translationText}>{item.translated_text}</Text>
                          </View>
                        ) : null}

                        {/* NER Tags */}
                        {nerTags.length > 0 ? (
                          <View style={styles.section}>
                            <Text style={styles.sectionLabel}>🏷️ Entities ({nerTags.length})</Text>
                            <View style={styles.nerGrid}>
                              {nerTags.map((entity: any, i: number) => {
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
                          <Text style={styles.deleteBtnText}>🗑️ Delete Annotation</Text>
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
  },
  actionBar: {
    flexDirection: 'row',
    padding: 12,
    gap: 10,
  },
  refreshBtn: {
    flex: 1,
    backgroundColor: '#1a73e8',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  refreshBtnText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
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
  emptyIcon: {
    fontSize: 64,
    marginBottom: 16,
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
    position: 'absolute',
    top: 10,
    left: 12,
    right: 12,
    zIndex: 999,
    padding: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  toastText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
  },
});
