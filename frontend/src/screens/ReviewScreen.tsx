import React, { useState, useCallback, useEffect } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, ScrollView,
  ActivityIndicator, TextInput, Platform,
} from 'react-native';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { BACKEND_URL } from '../config';
import { LANGUAGE_NAMES } from '../utils/languages';

const NER_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  PER:  { bg: '#e3f2fd', text: '#1565c0', label: 'Person' },
  ORG:  { bg: '#e8f5e9', text: '#2e7d32', label: 'Organization' },
  LOC:  { bg: '#fff3e0', text: '#e65100', label: 'Location' },
  MISC: { bg: '#f3e5f5', text: '#6a1b9a', label: 'Miscellaneous' },
  DATE: { bg: '#e0f7fa', text: '#00695c', label: 'Date' },
};

const STATUS_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  pending:  { bg: '#fffbeb', text: '#b45309', label: 'Pending' },
  approved: { bg: '#f0fdf4', text: '#16a34a', label: 'Approved' },
  rejected: { bg: '#fef2f2', text: '#dc2626', label: 'Rejected' },
};

const FILTER_TABS = [
  { key: 'all',      label: 'All' },
  { key: 'pending',  label: 'Pending' },
  { key: 'approved', label: 'Approved' },
  { key: 'rejected', label: 'Rejected' },
];

interface ReviewItem {
  id: number;
  original_text: string;
  translated_text: string | null;
  ner_tags: string | null;
  source_lang: string;
  target_lang: string;
  source_name: string | null;
  review_status: string;
  review_notes: string | null;
  reviewed_by: string | null;
  reviewed_at: string | null;
}

function Toast({ message, type }: { message: string; type: 'success' | 'error' | 'info' }) {
  const colors = { success: '#16a34a', error: '#dc2626', info: '#2563eb' };
  return (
    <View style={[styles.toast, { backgroundColor: colors[type] }]}>
      <Text style={styles.toastText}>{message}</Text>
    </View>
  );
}

function NerBadges({ nerJson }: { nerJson: string | null }) {
  if (!nerJson) return null;
  let entities: any[] = [];
  try { entities = JSON.parse(nerJson); } catch { return null; }
  if (!entities.length) return null;
  const shown = entities.slice(0, 30);
  return (
    <View style={styles.nerGrid}>
      {shown.map((e: any, i: number) => {
        const c = NER_COLORS[e.entity_group] || { bg: '#fce4ec', text: '#880e4f', label: e.entity_group };
        return (
          <View key={i} style={[styles.nerTag, { backgroundColor: c.bg, borderColor: c.text }]}>
            <Text style={[styles.nerWord, { color: c.text }]}>{e.word}</Text>
            <View style={[styles.nerBadge, { backgroundColor: c.text }]}>
              <Text style={styles.nerBadgeText}>{c.label}</Text>
            </View>
          </View>
        );
      })}
      {entities.length > 30 && (
        <Text style={styles.moreEntities}>+{entities.length - 30} more</Text>
      )}
    </View>
  );
}

export default function ReviewScreen() {
  const { token } = useAuth();
  const authHeaders = () => ({ headers: { Authorization: `Bearer ${token}` } });

  const [items, setItems]           = useState<ReviewItem[]>([]);
  const [loading, setLoading]       = useState(false);
  const [filterStatus, setFilter]   = useState('all');
  const [page, setPage]             = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal]           = useState(0);
  const [toast, setToast]           = useState<{ msg: string; type: 'success' | 'error' | 'info' } | null>(null);
  const [expanded, setExpanded]     = useState<number | null>(null);
  const [editId, setEditId]         = useState<number | null>(null);
  const [editOrig, setEditOrig]     = useState('');
  const [editTrans, setEditTrans]   = useState('');
  const [editNer, setEditNer]       = useState('');
  const [saving, setSaving]         = useState(false);
  const [rejectId, setRejectId]     = useState<number | null>(null);
  const [rejectNote, setRejectNote] = useState('');

  const showToast = (msg: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const fetchItems = useCallback(async (p = 1, status = filterStatus) => {
    setLoading(true);
    try {
      const res = await axios.get(
        `${BACKEND_URL}/api/review?page=${p}&limit=20&status=${status}`,
        authHeaders(),
      );
      const data = res.data;
      setItems(p === 1 ? data.items : prev => [...prev, ...data.items]);
      setPage(data.page);
      setTotalPages(data.pages);
      setTotal(data.total);
    } catch {
      showToast('Failed to load review queue', 'error');
    } finally {
      setLoading(false);
    }
  }, [token, filterStatus]);

  useEffect(() => { fetchItems(1, filterStatus); }, [filterStatus]);

  const handleFilter = (status: string) => {
    setFilter(status);
    setItems([]);
    setPage(1);
  };

  const handleLoadMore = () => {
    if (page < totalPages && !loading) fetchItems(page + 1);
  };

  // ── Review actions ──────────────────────────────────────
  const handleApprove = async (id: number) => {
    try {
      const res = await axios.patch(
        `${BACKEND_URL}/api/annotations/${id}/review`,
        { review_status: 'approved' },
        authHeaders(),
      );
      setItems(prev => prev.map(it => it.id === id ? { ...it, ...res.data } : it));
      showToast('Approved', 'success');
    } catch { showToast('Failed to approve', 'error'); }
  };

  const handleRejectSubmit = async () => {
    if (rejectId === null) return;
    setSaving(true);
    try {
      const res = await axios.patch(
        `${BACKEND_URL}/api/annotations/${rejectId}/review`,
        { review_status: 'rejected', review_notes: rejectNote.trim() || null },
        authHeaders(),
      );
      setItems(prev => prev.map(it => it.id === rejectId ? { ...it, ...res.data } : it));
      showToast('Rejected', 'info');
      setRejectId(null); setRejectNote('');
    } catch { showToast('Failed to reject', 'error'); }
    finally { setSaving(false); }
  };

  const handleResetReview = async (id: number) => {
    try {
      const res = await axios.patch(
        `${BACKEND_URL}/api/annotations/${id}/review`,
        { review_status: 'pending' },
        authHeaders(),
      );
      setItems(prev => prev.map(it => it.id === id ? { ...it, ...res.data } : it));
      showToast('Reset to pending', 'info');
    } catch { showToast('Failed to reset', 'error'); }
  };

  // ── Edit actions ────────────────────────────────────────
  const startEdit = (item: ReviewItem) => {
    setEditId(item.id);
    setEditOrig(item.original_text);
    setEditTrans(item.translated_text || '');
    setEditNer(item.ner_tags || '[]');
    setExpanded(item.id);
  };

  const handleSaveEdit = async () => {
    if (editId === null) return;
    setSaving(true);
    try {
      const res = await axios.patch(
        `${BACKEND_URL}/api/annotations/${editId}`,
        { original_text: editOrig, translated_text: editTrans || null, ner_tags: editNer },
        authHeaders(),
      );
      setItems(prev => prev.map(it => it.id === editId ? { ...it, ...res.data } : it));
      showToast('Saved — review status reset to Pending', 'success');
      setEditId(null);
    } catch (e: any) {
      showToast(e?.response?.data?.detail || 'Save failed', 'error');
    } finally { setSaving(false); }
  };

  const handleCancelEdit = () => { setEditId(null); };

  const getNerColor = (g: string) => NER_COLORS[g] || { bg: '#fce4ec', text: '#880e4f', label: g };

  return (
    <View style={styles.root}>
      {toast && <Toast message={toast.msg} type={toast.type} />}

      {/* Filter tabs */}
      <View style={styles.filterBar}>
        {FILTER_TABS.map(tab => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.filterTab, filterStatus === tab.key && styles.filterTabActive]}
            onPress={() => handleFilter(tab.key)}
          >
            <Text style={[styles.filterTabText, filterStatus === tab.key && styles.filterTabTextActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
        <TouchableOpacity style={styles.refreshBtn} onPress={() => fetchItems(1, filterStatus)}>
          <Text style={styles.refreshBtnText}>↺ Refresh</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.countText}>
        {total > 0 ? `${total} annotation${total !== 1 ? 's' : ''} in queue` : ''}
      </Text>

      {/* Reject modal */}
      {rejectId !== null && (
        <View style={styles.modalOverlay}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>Reject Annotation #{rejectId}</Text>
            <Text style={styles.modalLabel}>Reason (optional)</Text>
            <TextInput
              style={styles.modalInput}
              placeholder="Explain why this annotation is rejected…"
              placeholderTextColor="#aaa"
              multiline
              numberOfLines={3}
              value={rejectNote}
              onChangeText={setRejectNote}
            />
            <View style={styles.modalBtns}>
              <TouchableOpacity style={styles.modalCancel} onPress={() => { setRejectId(null); setRejectNote(''); }}>
                <Text style={styles.modalCancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.modalConfirm} onPress={handleRejectSubmit} disabled={saving}>
                <Text style={styles.modalConfirmText}>{saving ? 'Saving…' : 'Confirm Reject'}</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      )}

      {loading && items.length === 0 ? (
        <ActivityIndicator size="large" color="#1a73e8" style={{ marginTop: 40 }} />
      ) : items.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyTitle}>No annotations</Text>
          <Text style={styles.emptySub}>
            {filterStatus === 'all' ? 'Go to Annotate tab to create some.' : `No ${filterStatus} annotations.`}
          </Text>
        </View>
      ) : (
        <ScrollView contentContainerStyle={styles.list}>
          {items.map(item => {
            const isExpanded = expanded === item.id;
            const isEditing  = editId === item.id;
            const st = STATUS_STYLES[item.review_status] || STATUS_STYLES.pending;

            return (
              <View key={item.id} style={styles.card}>
                {/* Card header */}
                <TouchableOpacity onPress={() => setExpanded(isExpanded ? null : item.id)}>
                  <View style={styles.cardHeader}>
                    <View style={styles.cardMeta}>
                      <Text style={styles.cardId}>#{item.id}</Text>
                      <View style={[styles.statusBadge, { backgroundColor: st.bg }]}>
                        <Text style={[styles.statusBadgeText, { color: st.text }]}>{st.label}</Text>
                      </View>
                      <Text style={styles.langChip}>
                        {LANGUAGE_NAMES[item.source_lang] || item.source_lang} → {LANGUAGE_NAMES[item.target_lang] || item.target_lang}
                      </Text>
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

                {/* Action buttons */}
                <View style={styles.actionRow}>
                  {item.review_status !== 'approved' && (
                    <TouchableOpacity style={styles.approveBtn} onPress={() => handleApprove(item.id)}>
                      <Text style={styles.approveBtnText}>Approve</Text>
                    </TouchableOpacity>
                  )}
                  {item.review_status !== 'rejected' && (
                    <TouchableOpacity style={styles.rejectBtn} onPress={() => { setRejectId(item.id); setRejectNote(''); }}>
                      <Text style={styles.rejectBtnText}>Reject</Text>
                    </TouchableOpacity>
                  )}
                  {item.review_status !== 'pending' && (
                    <TouchableOpacity style={styles.resetBtn} onPress={() => handleResetReview(item.id)}>
                      <Text style={styles.resetBtnText}>Reset</Text>
                    </TouchableOpacity>
                  )}
                  <TouchableOpacity style={styles.editBtn} onPress={() => isEditing ? handleCancelEdit() : startEdit(item)}>
                    <Text style={styles.editBtnText}>{isEditing ? 'Cancel' : 'Edit'}</Text>
                  </TouchableOpacity>
                </View>

                {/* Review notes */}
                {item.review_notes ? (
                  <View style={styles.reviewNote}>
                    <Text style={styles.reviewNoteLabel}>Note: </Text>
                    <Text style={styles.reviewNoteText}>{item.review_notes}</Text>
                  </View>
                ) : null}
                {item.reviewed_by ? (
                  <Text style={styles.reviewedBy}>
                    Reviewed by {item.reviewed_by}
                    {item.reviewed_at ? ` · ${new Date(item.reviewed_at).toLocaleDateString()}` : ''}
                  </Text>
                ) : null}

                {/* Expanded details */}
                {isExpanded && !isEditing && (
                  <View style={styles.expandedContent}>
                    {item.translated_text ? (
                      <View style={styles.section}>
                        <Text style={styles.sectionLabel}>Translation</Text>
                        <Text style={styles.translationText}>{item.translated_text}</Text>
                      </View>
                    ) : null}
                    <View style={styles.section}>
                      <Text style={styles.sectionLabel}>Named Entities</Text>
                      <NerBadges nerJson={item.ner_tags} />
                      {(!item.ner_tags || item.ner_tags === '[]') && (
                        <Text style={styles.noEntities}>No entities detected</Text>
                      )}
                    </View>
                  </View>
                )}

                {/* Edit mode */}
                {isEditing && (
                  <View style={styles.editForm}>
                    <Text style={styles.editFormTitle}>✏️ Edit Annotation</Text>

                    <Text style={styles.editLabel}>Original Text</Text>
                    <TextInput
                      style={styles.editInput}
                      multiline
                      numberOfLines={4}
                      value={editOrig}
                      onChangeText={setEditOrig}
                      placeholder="Original text…"
                      placeholderTextColor="#aaa"
                    />

                    <Text style={styles.editLabel}>Translation</Text>
                    <TextInput
                      style={styles.editInput}
                      multiline
                      numberOfLines={4}
                      value={editTrans}
                      onChangeText={setEditTrans}
                      placeholder="Translated text…"
                      placeholderTextColor="#aaa"
                    />

                    <Text style={styles.editLabel}>Entities (JSON array)</Text>
                    <TextInput
                      style={[styles.editInput, { fontFamily: Platform.OS === 'web' ? 'monospace' : undefined }]}
                      multiline
                      numberOfLines={5}
                      value={editNer}
                      onChangeText={setEditNer}
                      placeholder='[{"word":"…","entity_group":"PER","score":0.99}]'
                      placeholderTextColor="#aaa"
                    />
                    <Text style={styles.editHint}>
                      Editing entities resets review status to Pending.
                    </Text>

                    <View style={styles.editBtns}>
                      <TouchableOpacity style={styles.editCancelBtn} onPress={handleCancelEdit}>
                        <Text style={styles.editCancelBtnText}>Cancel</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.editSaveBtn} onPress={handleSaveEdit} disabled={saving}>
                        <Text style={styles.editSaveBtnText}>{saving ? 'Saving…' : 'Save Changes'}</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                )}
              </View>
            );
          })}

          {/* Load more */}
          {page < totalPages && (
            <TouchableOpacity style={styles.loadMoreBtn} onPress={handleLoadMore} disabled={loading}>
              <Text style={styles.loadMoreText}>{loading ? 'Loading…' : 'Load More'}</Text>
            </TouchableOpacity>
          )}
          {loading && items.length > 0 && (
            <ActivityIndicator size="small" color="#7c3aed" style={{ marginVertical: 12 }} />
          )}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#F3F2FF' },

  toast: {
    marginHorizontal: 12, marginBottom: 8, paddingVertical: 12, paddingHorizontal: 16,
    borderRadius: 12, alignItems: 'center',
  },
  toastText: { color: '#fff', fontWeight: '700', fontSize: 14, textAlign: 'center' },

  filterBar: {
    flexDirection: 'row', alignItems: 'center', padding: 12, gap: 8,
    backgroundColor: '#ffffff', borderBottomWidth: 1, borderBottomColor: '#EDE9FE',
    flexWrap: 'wrap',
  },
  filterTab: {
    paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20,
    backgroundColor: '#F3F2FF', borderWidth: 1.5, borderColor: '#EDE9FE',
  },
  filterTabActive: { backgroundColor: '#7c3aed', borderColor: '#7c3aed' },
  filterTabText: { fontSize: 13, fontWeight: '600', color: '#7c6f9e' },
  filterTabTextActive: { color: '#fff' },
  refreshBtn: {
    marginLeft: 'auto' as any, paddingHorizontal: 14, paddingVertical: 7,
    borderRadius: 20, backgroundColor: '#F3F2FF', borderWidth: 1.5, borderColor: '#EDE9FE',
  },
  refreshBtnText: { fontSize: 13, fontWeight: '600', color: '#7c3aed' },

  countText: { fontSize: 12, color: '#a78bfa', textAlign: 'center', paddingVertical: 6, fontWeight: '500' },

  // Modal
  modalOverlay: {
    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
    backgroundColor: 'rgba(30,27,75,0.5)', zIndex: 9999,
    justifyContent: 'center', alignItems: 'center',
  },
  modal: {
    backgroundColor: '#fff', borderRadius: 20, padding: 24,
    width: '90%', maxWidth: 480,
    shadowColor: '#7c3aed', shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.2, shadowRadius: 24, elevation: 20,
    borderWidth: 1, borderColor: '#EDE9FE',
  },
  modalTitle: { fontSize: 18, fontWeight: '800', color: '#1e1b4b', marginBottom: 16 },
  modalLabel: { fontSize: 13, fontWeight: '600', color: '#6d28d9', marginBottom: 6 },
  modalInput: {
    borderWidth: 1.5, borderColor: '#EDE9FE', borderRadius: 12, padding: 12,
    fontSize: 14, color: '#1e1b4b', minHeight: 80, textAlignVertical: 'top',
    backgroundColor: '#F8F7FF',
  },
  modalBtns: { flexDirection: 'row', gap: 10, marginTop: 16, justifyContent: 'flex-end' },
  modalCancel: {
    paddingHorizontal: 20, paddingVertical: 10, borderRadius: 10,
    backgroundColor: '#F3F2FF', borderWidth: 1.5, borderColor: '#EDE9FE',
  },
  modalCancelText: { color: '#7c6f9e', fontWeight: '600' },
  modalConfirm: {
    paddingHorizontal: 20, paddingVertical: 10, borderRadius: 10, backgroundColor: '#ec4899',
  },
  modalConfirmText: { color: '#fff', fontWeight: '700' },

  // List
  list: { padding: 14, paddingTop: 6 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 40, marginTop: 60 },
  emptyTitle: { fontSize: 20, fontWeight: '700', color: '#1e1b4b', marginBottom: 10 },
  emptySub: { fontSize: 14, color: '#a78bfa', textAlign: 'center', lineHeight: 22 },

  // Card
  card: {
    backgroundColor: '#fff', borderRadius: 16, padding: 16, marginBottom: 12,
    shadowColor: '#7c3aed', shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07, shadowRadius: 12, elevation: 3,
    borderWidth: 1, borderColor: '#EDE9FE',
  },
  cardHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10,
  },
  cardMeta: { flexDirection: 'row', alignItems: 'center', gap: 6, flex: 1, flexWrap: 'wrap' },
  cardId: {
    fontSize: 12, color: '#a78bfa', fontWeight: '700', backgroundColor: '#F3F2FF',
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8,
  },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20 },
  statusBadgeText: { fontSize: 12, fontWeight: '700' },
  langChip: { fontSize: 12, color: '#7c6f9e', fontWeight: '500' },
  fileBadge: {
    fontSize: 11, color: '#7c3aed', fontWeight: '600', backgroundColor: '#EDE9FE',
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: 8, maxWidth: 180,
  },
  expandIcon: { fontSize: 12, color: '#c4b5fd' },
  originalText: { fontSize: 14, color: '#1e1b4b', lineHeight: 22, marginBottom: 10 },

  // Action row
  actionRow: { flexDirection: 'row', gap: 8, flexWrap: 'wrap', marginBottom: 8 },
  approveBtn: {
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 10,
    backgroundColor: '#f0fdf4', borderWidth: 1.5, borderColor: '#86efac',
  },
  approveBtnText: { color: '#16a34a', fontWeight: '700', fontSize: 13 },
  rejectBtn: {
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 10,
    backgroundColor: '#fdf2f8', borderWidth: 1.5, borderColor: '#fbcfe8',
  },
  rejectBtnText: { color: '#ec4899', fontWeight: '700', fontSize: 13 },
  resetBtn: {
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 10,
    backgroundColor: '#fffbeb', borderWidth: 1.5, borderColor: '#fcd34d',
  },
  resetBtnText: { color: '#b45309', fontWeight: '700', fontSize: 13 },
  editBtn: {
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 10,
    backgroundColor: '#EDE9FE', borderWidth: 1.5, borderColor: '#c4b5fd',
  },
  editBtnText: { color: '#7c3aed', fontWeight: '700', fontSize: 13 },

  reviewNote: { flexDirection: 'row', backgroundColor: '#fffbeb', borderRadius: 10, padding: 10, marginBottom: 6 },
  reviewNoteLabel: { fontSize: 12, fontWeight: '700', color: '#b45309' },
  reviewNoteText: { fontSize: 12, color: '#78350f', flex: 1 },
  reviewedBy: { fontSize: 11, color: '#c4b5fd', marginBottom: 4, fontStyle: 'italic' },

  // Expanded
  expandedContent: { marginTop: 10, paddingTop: 12, borderTopWidth: 1, borderTopColor: '#EDE9FE' },
  section: { marginBottom: 12 },
  sectionLabel: { fontSize: 13, fontWeight: '700', color: '#7c3aed', marginBottom: 6 },
  translationText: {
    fontSize: 14, color: '#6d28d9', lineHeight: 22,
    backgroundColor: '#EDE9FE', padding: 12, borderRadius: 10,
  },
  nerGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  nerTag: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 10, paddingVertical: 5,
    borderRadius: 20, borderWidth: 1, marginBottom: 4,
  },
  nerWord: { fontSize: 13, fontWeight: '700', marginRight: 6 },
  nerBadge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 10 },
  nerBadgeText: { color: '#fff', fontSize: 10, fontWeight: '700' },
  noEntities: { fontSize: 13, color: '#c4b5fd', fontStyle: 'italic' },
  moreEntities: { fontSize: 12, color: '#a78bfa', paddingVertical: 4 },

  // Edit form
  editForm: {
    marginTop: 12, paddingTop: 14, borderTopWidth: 1, borderTopColor: '#EDE9FE',
  },
  editFormTitle: { fontSize: 15, fontWeight: '800', color: '#1e1b4b', marginBottom: 12 },
  editLabel: { fontSize: 13, fontWeight: '600', color: '#6d28d9', marginBottom: 4 },
  editInput: {
    borderWidth: 1.5, borderColor: '#EDE9FE', borderRadius: 10, padding: 10,
    fontSize: 13, color: '#1e1b4b', textAlignVertical: 'top',
    marginBottom: 10, minHeight: 80, backgroundColor: '#F8F7FF',
  },
  editHint: { fontSize: 12, color: '#a78bfa', marginBottom: 10 },
  editBtns: { flexDirection: 'row', gap: 10, justifyContent: 'flex-end' },
  editCancelBtn: {
    paddingHorizontal: 18, paddingVertical: 9, borderRadius: 10,
    backgroundColor: '#F3F2FF', borderWidth: 1.5, borderColor: '#EDE9FE',
  },
  editCancelBtnText: { color: '#7c6f9e', fontWeight: '600' },
  editSaveBtn: {
    paddingHorizontal: 18, paddingVertical: 9, borderRadius: 10, backgroundColor: '#7c3aed',
    shadowColor: '#7c3aed', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8,
  },
  editSaveBtnText: { color: '#fff', fontWeight: '700' },

  // Load more
  loadMoreBtn: {
    backgroundColor: '#fff', borderRadius: 12, paddingVertical: 14,
    alignItems: 'center', marginVertical: 8,
    borderWidth: 1.5, borderColor: '#c4b5fd',
  },
  loadMoreText: { color: '#7c3aed', fontWeight: '700', fontSize: 14 },
});
