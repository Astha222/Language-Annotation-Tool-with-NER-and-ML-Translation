import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ActivityIndicator, ScrollView, Platform, Alert
} from 'react-native';
import axios from 'axios';

const BACKEND_URL = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

// NER entity type → color mapping
const NER_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  PER:  { bg: '#e3f2fd', text: '#1565c0', label: 'Person' },
  ORG:  { bg: '#e8f5e9', text: '#2e7d32', label: 'Organization' },
  LOC:  { bg: '#fff3e0', text: '#e65100', label: 'Location' },
  MISC: { bg: '#f3e5f5', text: '#6a1b9a', label: 'Miscellaneous' },
};

const LANGUAGES = [
  { code: 'hi', name: 'Hindi',   flag: '🇮🇳' },
  { code: 'fr', name: 'French',  flag: '🇫🇷' },
  { code: 'es', name: 'Spanish', flag: '🇪🇸' },
  { code: 'de', name: 'German',  flag: '🇩🇪' },
  { code: 'zh', name: 'Chinese', flag: '🇨🇳' },
  { code: 'ar', name: 'Arabic',  flag: '🇸🇦' },
  { code: 'ja', name: 'Japanese',flag: '🇯🇵' },
  { code: 'ru', name: 'Russian', flag: '🇷🇺' },
];

function Toast({ message, type }: { message: string; type: 'success' | 'error' | 'info' }) {
  const colors = { success: '#2e7d32', error: '#c62828', info: '#1565c0' };
  return (
    <View style={[styles.toast, { backgroundColor: colors[type] }]}>
      <Text style={styles.toastText}>{message}</Text>
    </View>
  );
}

export default function HomeScreen() {
  const [inputText, setInputText]           = useState('');
  const [targetLang, setTargetLang]         = useState('fr');
  const [loading, setLoading]               = useState(false);
  const [nerResults, setNerResults]         = useState<any[]>([]);
  const [translationResult, setTranslationResult] = useState('');
  const [detectedLang, setDetectedLang]     = useState('');
  const [toast, setToast]                   = useState<{ msg: string; type: 'success' | 'error' | 'info' } | null>(null);
  const [saved, setSaved]                   = useState(false);

  const showToast = (msg: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const copyToClipboard = async (text: string) => {
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard) {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
      } else {
        showToast('Copy not supported on this platform', 'info');
      }
    } catch {
      showToast('Failed to copy', 'error');
    }
  };

  const handleAnalyzeAndTranslate = async () => {
    if (!inputText.trim()) {
      showToast('Please enter some text first', 'error');
      return;
    }

    setLoading(true);
    setNerResults([]);
    setTranslationResult('');
    setDetectedLang('');
    setSaved(false);

    let srcLang = 'en';
    let nerEntities: any[] = [];
    let translatedText: string | null = null;

    try {
      // Step 1: Auto-detect source language
      try {
        const detectRes = await axios.post(`${BACKEND_URL}/api/detect-language`, { text: inputText });
        srcLang = detectRes.data.detected_language || 'en';
        setDetectedLang(srcLang);
      } catch {
        setDetectedLang('en');
      }

      // Step 2: NER (required — abort if this fails)
      const nerRes = await axios.post(`${BACKEND_URL}/api/ner`, { text: inputText });
      nerEntities = nerRes.data.entities;
      setNerResults(nerEntities);

      // Step 3: Translation (optional — show warning if it fails, but continue)
      try {
        const transRes = await axios.post(`${BACKEND_URL}/api/translate`, {
          text: inputText,
          source_lang: srcLang,
          target_lang: targetLang,
        });
        translatedText = transRes.data.translated_text;
        setTranslationResult(translatedText ?? '');
      } catch (transErr: any) {
        const transDetail = transErr?.response?.data?.detail || `Translation to "${targetLang}" not available`;
        showToast(`⚠️ ${transDetail}`, 'info');
      }

      // Step 4: Always save annotation (even without translation)
      await axios.post(`${BACKEND_URL}/api/annotations`, {
        original_text:   inputText,
        translated_text: translatedText,
        ner_tags:        JSON.stringify(nerEntities),
        source_lang:     srcLang,
        target_lang:     targetLang,
      });
      setSaved(true);
      if (translatedText) {
        showToast('✅ Annotation saved successfully!', 'success');
      } else {
        showToast('✅ Annotation saved (no translation)', 'success');
      }

    } catch (err: any) {
      const detail = err?.response?.data?.detail || 'Error connecting to backend. Is the server running?';
      showToast(`❌ ${detail}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setInputText('');
    setNerResults([]);
    setTranslationResult('');
    setDetectedLang('');
    setSaved(false);
  };

  const getNerColor = (entityGroup: string) =>
    NER_COLORS[entityGroup] || { bg: '#fce4ec', text: '#880e4f', label: entityGroup };

  return (
    <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
      {/* Toast */}
      {toast && <Toast message={toast.msg} type={toast.type} />}

      {/* Input Section */}
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>📝 Input Text</Text>
        <TextInput
          style={styles.input}
          multiline
          numberOfLines={4}
          placeholder="e.g. Steve Jobs founded Apple in California in 1976..."
          placeholderTextColor="#aaa"
          value={inputText}
          onChangeText={setInputText}
        />
        {detectedLang ? (
          <Text style={styles.detectedLang}>🌐 Detected language: <Text style={styles.detectedLangValue}>{detectedLang.toUpperCase()}</Text></Text>
        ) : null}
      </View>

      {/* Language Selector */}
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>🌍 Target Language</Text>
        <View style={styles.langGrid}>
          {LANGUAGES.map((lang) => (
            <TouchableOpacity
              key={lang.code}
              style={[styles.langChip, targetLang === lang.code && styles.langChipActive]}
              onPress={() => setTargetLang(lang.code)}
            >
              <Text style={styles.langFlag}>{lang.flag}</Text>
              <Text style={[styles.langName, targetLang === lang.code && styles.langNameActive]}>
                {lang.name}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Buttons */}
      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.primaryBtn} onPress={handleAnalyzeAndTranslate} disabled={loading}>
          <Text style={styles.primaryBtnText}>{loading ? 'Analyzing...' : '🔍 Analyze & Translate'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.secondaryBtn} onPress={handleClear}>
          <Text style={styles.secondaryBtnText}>🗑 Clear</Text>
        </TouchableOpacity>
      </View>

      {loading && <ActivityIndicator size="large" color="#1a73e8" style={{ marginTop: 20 }} />}

      {/* Translation Result */}
      {translationResult ? (
        <View style={styles.card}>
          <View style={styles.resultHeader}>
            <Text style={styles.sectionTitle}>🌐 Translation</Text>
            <TouchableOpacity onPress={() => copyToClipboard(translationResult)} style={styles.copyBtn}>
              <Text style={styles.copyBtnText}>📋 Copy</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.translationText}>{translationResult}</Text>
          {saved && <Text style={styles.savedBadge}>✅ Saved to database</Text>}
        </View>
      ) : null}

      {/* NER Results */}
      {nerResults.length > 0 ? (
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>🏷️ Named Entities ({nerResults.length} found)</Text>
          <View style={styles.nerGrid}>
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
        !loading && translationResult && nerResults.length === 0 ? (
          <View style={styles.card}>
            <Text style={styles.noEntities}>ℹ️ No named entities found in this text.</Text>
          </View>
        ) : null
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    backgroundColor: '#f5f7fa',
    flexGrow: 1,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#333',
    marginBottom: 10,
  },
  input: {
    borderWidth: 1.5,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    backgroundColor: '#fafafa',
    minHeight: 100,
    textAlignVertical: 'top',
    color: '#222',
  },
  detectedLang: {
    marginTop: 8,
    fontSize: 13,
    color: '#666',
  },
  detectedLangValue: {
    fontWeight: 'bold',
    color: '#1a73e8',
  },
  langGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  langChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
    borderWidth: 1.5,
    borderColor: '#ddd',
    marginRight: 8,
    marginBottom: 8,
  },
  langChipActive: {
    backgroundColor: '#e8f0fe',
    borderColor: '#1a73e8',
  },
  langFlag: {
    fontSize: 16,
    marginRight: 5,
  },
  langName: {
    fontSize: 13,
    color: '#555',
    fontWeight: '500',
  },
  langNameActive: {
    color: '#1a73e8',
    fontWeight: '700',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 14,
  },
  primaryBtn: {
    flex: 1,
    backgroundColor: '#1a73e8',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
  },
  primaryBtnText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '700',
  },
  secondaryBtn: {
    backgroundColor: '#fff',
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 18,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#ddd',
  },
  secondaryBtnText: {
    color: '#555',
    fontSize: 15,
    fontWeight: '600',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  copyBtn: {
    backgroundColor: '#e8f0fe',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  copyBtnText: {
    color: '#1a73e8',
    fontWeight: '600',
    fontSize: 13,
  },
  translationText: {
    fontSize: 16,
    color: '#222',
    lineHeight: 24,
  },
  savedBadge: {
    marginTop: 10,
    fontSize: 13,
    color: '#2e7d32',
    fontWeight: '600',
  },
  nerGrid: {
    gap: 8,
  },
  nerTag: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 6,
  },
  nerWord: {
    fontSize: 15,
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
    fontSize: 11,
    fontWeight: '700',
  },
  nerScore: {
    fontSize: 12,
    color: '#888',
    fontWeight: '500',
  },
  noEntities: {
    fontSize: 14,
    color: '#888',
    fontStyle: 'italic',
    textAlign: 'center',
  },
  toast: {
    position: 'absolute',
    top: 10,
    left: 16,
    right: 16,
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
