import React, { useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ActivityIndicator, ScrollView, Platform } from 'react-native';
import axios from 'axios';

// IMPORTANT: For Android Emulator to access local backend, use 10.0.2.2 instead of 127.0.0.1
const BACKEND_URL = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';

export default function HomeScreen() {
  const [inputText, setInputText] = useState('');
  const [targetLang, setTargetLang] = useState('fr'); // Default to French
  const [loading, setLoading] = useState(false);
  const [nerResults, setNerResults] = useState<any[]>([]);
  const [translationResult, setTranslationResult] = useState('');
  const [error, setError] = useState('');

  const handleAnalyzeAndTranslate = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text');
      return;
    }
    
    setLoading(true);
    setError('');
    setNerResults([]);
    setTranslationResult('');

    try {
      // 1. Call NER API
      const nerRes = await axios.post(`${BACKEND_URL}/api/ner`, {
        text: inputText
      });
      setNerResults(nerRes.data.entities);

      // 2. Call Translate API
      const transRes = await axios.post(`${BACKEND_URL}/api/translate`, {
        text: inputText,
        source_lang: 'en',
        target_lang: targetLang
      });
      setTranslationResult(transRes.data.translated_text);

      // 3. Save to database
      await axios.post(`${BACKEND_URL}/api/annotations`, {
        original_text: inputText,
        translated_text: transRes.data.translated_text,
        ner_tags: JSON.stringify(nerRes.data.entities)
      });

    } catch (err: any) {
      console.error(err);
      setError('Error connecting to backend. Is the server running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Enter Text (English):</Text>
      <TextInput
        style={styles.input}
        multiline
        placeholder="e.g. Steve Jobs founded Apple in California"
        value={inputText}
        onChangeText={setInputText}
      />

      <Text style={styles.title}>Select Target Language:</Text>
      <View style={styles.languageContainer}>
        {[
          { code: 'hi', name: 'Hindi' },
          { code: 'fr', name: 'French' },
          { code: 'es', name: 'Spanish' },
          { code: 'de', name: 'German' }
        ].map((lang) => (
          <Text 
            key={lang.code} 
            style={[styles.langChip, targetLang === lang.code && styles.langChipActive]}
            onPress={() => setTargetLang(lang.code)}
          >
            {lang.name}
          </Text>
        ))}
      </View>

      <View style={styles.buttonContainer}>
        <Button title="Analyze & Translate" onPress={handleAnalyzeAndTranslate} />
      </View>

      {loading ? <ActivityIndicator size="large" color="#0000ff" style={{ marginTop: 20 }} /> : null}
      {error ? <Text style={styles.errorText}>{error}</Text> : null}

      {/* Results Section */}
      {translationResult ? (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>Translation:</Text>
          <Text style={styles.resultText}>{translationResult}</Text>
        </View>
      ) : null}

      {nerResults.length > 0 ? (
        <View style={styles.resultContainer}>
          <Text style={styles.resultTitle}>NER Tags found:</Text>
          {nerResults.map((entity, index) => (
            <Text key={index} style={styles.nerText}>
              • {entity.word} - <Text style={{fontWeight: 'bold', color: 'blue'}}>{entity.entity_group}</Text>
            </Text>
          ))}
        </View>
      ) : (
        (!loading && inputText !== '' && translationResult !== '' && nerResults.length === 0) ? (
          <Text style={{marginTop: 10, fontStyle: 'italic'}}>No Named Entities found.</Text>
        ) : null
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#fff',
    flexGrow: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
    marginTop: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 10,
    fontSize: 16,
    backgroundColor: '#f9f9f9',
  },
  buttonContainer: {
    marginTop: 20,
  },
  resultContainer: {
    marginTop: 20,
    padding: 15,
    backgroundColor: '#e6f7ff',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#91d5ff',
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#0050b3',
  },
  resultText: {
    fontSize: 16,
    color: '#333',
  },
  nerText: {
    fontSize: 16,
    marginBottom: 5,
  },
  errorText: {
    color: 'red',
    marginTop: 15,
    fontWeight: 'bold',
    textAlign: 'center'
  },
  languageContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 5,
    marginBottom: 5,
  },
  langChip: {
    paddingVertical: 8,
    paddingHorizontal: 15,
    backgroundColor: '#f0f0f0',
    borderRadius: 20,
    marginRight: 10,
    marginBottom: 10,
    color: '#333',
    borderWidth: 1,
    borderColor: '#ccc',
    overflow: 'hidden'
  },
  langChipActive: {
    backgroundColor: '#1890ff',
    color: '#fff',
    borderColor: '#1890ff',
    fontWeight: 'bold'
  }
});
