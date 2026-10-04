import React, { useState } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  SafeAreaView, Platform, useWindowDimensions, ActivityIndicator,
} from 'react-native';
import { AuthProvider, useAuth } from './src/context/AuthContext';
import HomeScreen from './src/screens/HomeScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import LoginScreen from './src/screens/LoginScreen';
import RegisterScreen from './src/screens/RegisterScreen';

type Tab = 'annotate' | 'history';

const NAV_ITEMS: { key: Tab; label: string; icon: string }[] = [
  { key: 'annotate', label: 'Annotate', icon: 'A' },
  { key: 'history',  label: 'History',  icon: 'H' },
];

// ── Inner app (wrapped by AuthProvider) ────────────────────────────────────
function AppInner() {
  const { token, user, logout, isLoading } = useAuth();
  const [activeTab, setActiveTab] = useState<Tab>('annotate');
  const [showRegister, setShowRegister] = useState(false);
  const { width } = useWindowDimensions();
  const isSidebar = Platform.OS === 'web' && width >= 768;

  // Loading state while AsyncStorage hydrates
  if (isLoading) {
    return (
      <View style={styles.loadingContainer}>
        <View style={styles.logoBox}>
          <Text style={styles.logoText}>LAT</Text>
        </View>
        <ActivityIndicator size="large" color="#1a73e8" style={{ marginTop: 24 }} />
        <Text style={styles.loadingText}>Loading…</Text>
      </View>
    );
  }

  // Not logged in → show Login or Register
  if (!token) {
    if (showRegister) {
      return <RegisterScreen onSwitchToLogin={() => setShowRegister(false)} />;
    }
    return <LoginScreen onSwitchToRegister={() => setShowRegister(true)} />;
  }

  // Logged in → show main app
  return (
    <SafeAreaView style={styles.root}>

      {/* TOP HEADER */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.logoBox}>
            <Text style={styles.logoText}>LAT</Text>
          </View>
          <View>
            <Text style={styles.headerTitle}>Language Annotation Tool</Text>
            <Text style={styles.headerSub}>NER · Translation · Export</Text>
          </View>
        </View>

        <View style={styles.headerRight}>
          {/* Top-bar tabs (only on small / mobile) */}
          {!isSidebar && (
            <View style={styles.headerTabs}>
              {NAV_ITEMS.map((item) => (
                <TouchableOpacity
                  key={item.key}
                  style={[styles.headerTab, activeTab === item.key && styles.headerTabActive]}
                  onPress={() => setActiveTab(item.key)}
                >
                  <Text style={[styles.headerTabText, activeTab === item.key && styles.headerTabTextActive]}>
                    {item.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}

          {/* User info + logout */}
          <View style={styles.userArea}>
            <Text style={styles.userGreeting}>👤 {user?.username}</Text>
            <TouchableOpacity style={styles.logoutBtn} onPress={logout}>
              <Text style={styles.logoutBtnText}>Logout</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* BODY: sidebar + content */}
      <View style={styles.body}>

        {/* SIDEBAR — web wide screens only */}
        {isSidebar && (
          <View style={styles.sidebar}>
            <Text style={styles.sidebarSection}>Navigation</Text>
            {NAV_ITEMS.map((item) => (
              <TouchableOpacity
                key={item.key}
                style={[styles.navItem, activeTab === item.key && styles.navItemActive]}
                onPress={() => setActiveTab(item.key)}
              >
                <View style={[styles.navIcon, activeTab === item.key && styles.navIconActive]}>
                  <Text style={[styles.navIconText, activeTab === item.key && styles.navIconTextActive]}>
                    {item.icon}
                  </Text>
                </View>
                <Text style={[styles.navLabel, activeTab === item.key && styles.navLabelActive]}>
                  {item.label}
                </Text>
              </TouchableOpacity>
            ))}

            {/* Sidebar footer */}
            <View style={styles.sidebarFooter}>
              <View style={styles.statusDot} />
              <Text style={styles.sidebarFooterText}>
                {user?.username ? `${user.username}` : 'Connected'}
              </Text>
            </View>
          </View>
        )}

        {/* MAIN CONTENT */}
        <View style={styles.content}>
          {/* Page title bar */}
          <View style={styles.pageTitleBar}>
            <Text style={styles.pageTitle}>
              {activeTab === 'annotate' ? 'Annotate Text' : 'Annotation History'}
            </Text>
            <Text style={styles.pageSubtitle}>
              {activeTab === 'annotate'
                ? 'Upload or type text to detect entities and translate'
                : 'Browse, search and export saved annotations'}
            </Text>
          </View>

          {activeTab === 'annotate' ? <HomeScreen /> : <HistoryScreen />}
        </View>
      </View>
    </SafeAreaView>
  );
}

// ── Root export (wraps with AuthProvider) ──────────────────────────────────
export default function App() {
  return (
    <AuthProvider>
      <AppInner />
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  // Loading
  loadingContainer: {
    flex: 1,
    backgroundColor: '#1a1f36',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#8b92a9',
    marginTop: 12,
    fontSize: 14,
  },

  root: {
    flex: 1,
    backgroundColor: '#f0f2f5',
  },

  // ── HEADER
  header: {
    backgroundColor: '#1a1f36',
    paddingVertical: 0,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 60,
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 6,
    zIndex: 100,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  logoBox: {
    width: 36,
    height: 36,
    borderRadius: 8,
    backgroundColor: '#1a73e8',
    justifyContent: 'center',
    alignItems: 'center',
  },
  logoText: {
    color: '#fff',
    fontWeight: '900',
    fontSize: 13,
    letterSpacing: 0.5,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  headerSub: {
    color: '#8b92a9',
    fontSize: 11,
    marginTop: 1,
  },
  headerTabs: {
    flexDirection: 'row',
    gap: 4,
  },
  headerTab: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 6,
  },
  headerTabActive: {
    backgroundColor: '#1a73e8',
  },
  headerTabText: {
    color: '#8b92a9',
    fontWeight: '600',
    fontSize: 13,
  },
  headerTabTextActive: {
    color: '#fff',
  },
  userArea: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginLeft: 8,
  },
  userGreeting: {
    color: '#8b92a9',
    fontSize: 13,
    fontWeight: '500',
  },
  logoutBtn: {
    backgroundColor: '#2a2f4a',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#3a4060',
  },
  logoutBtnText: {
    color: '#ef4444',
    fontSize: 12,
    fontWeight: '700',
  },

  // ── BODY
  body: {
    flex: 1,
    flexDirection: 'row',
  },

  // ── SIDEBAR
  sidebar: {
    width: 200,
    backgroundColor: '#1a1f36',
    paddingTop: 24,
    paddingHorizontal: 12,
    borderRightWidth: 1,
    borderRightColor: '#252a43',
  },
  sidebarSection: {
    fontSize: 10,
    fontWeight: '700',
    color: '#5a6177',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
    marginBottom: 10,
    paddingLeft: 8,
  },
  navItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderRadius: 8,
    marginBottom: 4,
  },
  navItemActive: {
    backgroundColor: '#252a43',
  },
  navIcon: {
    width: 28,
    height: 28,
    borderRadius: 6,
    backgroundColor: '#252a43',
    justifyContent: 'center',
    alignItems: 'center',
  },
  navIconActive: {
    backgroundColor: '#1a73e8',
  },
  navIconText: {
    color: '#5a6177',
    fontWeight: '700',
    fontSize: 12,
  },
  navIconTextActive: {
    color: '#fff',
  },
  navLabel: {
    color: '#8b92a9',
    fontSize: 14,
    fontWeight: '500',
  },
  navLabelActive: {
    color: '#fff',
    fontWeight: '700',
  },
  sidebarFooter: {
    position: 'absolute',
    bottom: 20,
    left: 12,
    right: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#252a43',
    borderRadius: 8,
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#34d399',
  },
  sidebarFooterText: {
    color: '#8b92a9',
    fontSize: 12,
    fontWeight: '500',
  },

  // ── CONTENT
  content: {
    flex: 1,
    backgroundColor: '#f0f2f5',
  },
  pageTitleBar: {
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: 10,
    backgroundColor: '#f0f2f5',
    borderBottomWidth: 1,
    borderBottomColor: '#e2e5ed',
  },
  pageTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: '#1a1f36',
    letterSpacing: 0.2,
  },
  pageSubtitle: {
    fontSize: 13,
    color: '#8b92a9',
    marginTop: 2,
  },
});
