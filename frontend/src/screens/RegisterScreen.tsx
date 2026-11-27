import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { BACKEND_URL } from '../config';

interface RegisterScreenProps {
  onSwitchToLogin: () => void;
}

export default function RegisterScreen({ onSwitchToLogin }: RegisterScreenProps) {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const handleRegister = async () => {
    if (!username || !email || !password || !confirmPassword) {
      setError('Please fill in all fields');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);
    setError('');
    try {
      // Register
      await axios.post(`${BACKEND_URL}/api/auth/register`, { username: username.trim(), email: email.trim(), password });
      
      // Auto-login
      const loginRes = await axios.post(`${BACKEND_URL}/api/auth/login`, { username: username.trim(), password });
      const token = loginRes.data.access_token;
      
      const userRes = await axios.get(`${BACKEND_URL}/api/auth/me`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      await login(token, userRes.data);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={styles.logoBox}><Text style={styles.logoText}>LAT</Text></View>
        <Text style={styles.headerTitle}>Language Annotation Tool</Text>
        <Text style={styles.headerSub}>Create an account</Text>
      </View>
      <View style={styles.form}>
        {error ? <Text style={styles.errorText}>{error}</Text> : null}
        
        <TextInput
          style={styles.input}
          placeholder="Username"
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
        />

        <TextInput
          style={styles.input}
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />
        
        <TextInput
          style={styles.input}
          placeholder="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        <TextInput
          style={styles.input}
          placeholder="Confirm Password"
          value={confirmPassword}
          onChangeText={setConfirmPassword}
          secureTextEntry
        />
        
        <TouchableOpacity style={styles.button} onPress={handleRegister} disabled={loading}>
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Register</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity onPress={onSwitchToLogin} style={styles.linkContainer}>
          <Text style={styles.linkText}>Already have an account? Login</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F3F2FF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingBottom: 28,
  },
  logoBox: {
    backgroundColor: '#7c3aed',
    width: 72,
    height: 72,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
  },
  logoText: {
    color: '#ffffff',
    fontSize: 26,
    fontWeight: '900',
    letterSpacing: 1,
  },
  headerTitle: {
    color: '#1e1b4b',
    fontSize: 24,
    fontWeight: '800',
    marginBottom: 6,
  },
  headerSub: {
    color: '#a78bfa',
    fontSize: 14,
    fontWeight: '500',
  },
  form: {
    backgroundColor: '#ffffff',
    borderRadius: 24,
    padding: 28,
    width: 360,
    maxWidth: '90%' as any,
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.1,
    shadowRadius: 24,
    borderWidth: 1,
    borderColor: '#EDE9FE',
  },
  input: {
    backgroundColor: '#F8F7FF',
    padding: 15,
    borderRadius: 12,
    marginBottom: 14,
    borderWidth: 1.5,
    borderColor: '#EDE9FE',
    fontSize: 15,
    color: '#1e1b4b',
  },
  button: {
    backgroundColor: '#7c3aed',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 6,
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  errorText: {
    color: '#ec4899',
    marginBottom: 14,
    textAlign: 'center',
    fontSize: 13,
    fontWeight: '600',
    backgroundColor: '#fdf2f8',
    padding: 10,
    borderRadius: 8,
  },
  linkContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  linkText: {
    color: '#7c3aed',
    fontSize: 14,
    fontWeight: '600',
  },
});
