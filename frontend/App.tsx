import React, { useState } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  SafeAreaView, Platform, useWindowDimensions, ActivityIndicator,
} from 'react-native';
import { AuthProvider, useAuth } from './src/context/AuthContext';
import HomeScreen from './src/screens/HomeScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import ReviewScreen from './src/screens/ReviewScreen';
import LoginScreen from './src/screens/LoginScreen';
import RegisterScreen from './src/screens/RegisterScreen';

type Tab = 'annotate' | 'history' | 'review';

const NAV_ITEMS: { key: Tab; label: string; icon: string }[] = [
  { key: 'annotate', label: 'Annotate', icon: 'A' },
  { key: 'history',  label: 'History',  icon: 'H' },
  { key: 'review',   label: 'Review',   icon: 'R' },
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
              {activeTab === 'annotate' ? 'Annotate Text' : activeTab === 'review' ? 'Human Verification Queue' : 'Annotation History'}
            </Text>
            <Text style={styles.pageSubtitle}>
              {activeTab === 'annotate'
                ? 'Upload or type text to detect entities and translate'
                : activeTab === 'review'
                ? 'Review, edit and approve annotations'
                : 'Browse, search and export saved annotations'}
            </Text>
          </View>

          {activeTab === 'annotate' ? <HomeScreen /> : activeTab === 'review' ? <ReviewScreen /> : <HistoryScreen />}
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
    backgroundColor: '#F8F7FF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#7c3aed',
    marginTop: 12,
    fontSize: 14,
    fontWeight: '600',
  },

  root: {
    flex: 1,
    backgroundColor: '#F3F2FF',
  },

  // ── HEADER
  header: {
    backgroundColor: '#ffffff',
    paddingVertical: 0,
    paddingHorizontal: 24,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 64,
    borderBottomWidth: 1,
    borderBottomColor: '#E8E6FF',
    elevation: 0,
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
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
    width: 38,
    height: 38,
    borderRadius: 10,
    backgroundColor: '#7c3aed',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  logoText: {
    color: '#fff',
    fontWeight: '900',
    fontSize: 13,
    letterSpacing: 0.5,
  },
  headerTitle: {
    color: '#1e1b4b',
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 0.2,
  },
  headerSub: {
    color: '#a78bfa',
    fontSize: 11,
    marginTop: 1,
    fontWeight: '500',
  },
  headerTabs: {
    flexDirection: 'row',
    gap: 4,
  },
  headerTab: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 20,
  },
  headerTabActive: {
    backgroundColor: '#7c3aed',
  },
  headerTabText: {
    color: '#a78bfa',
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
    color: '#6d28d9',
    fontSize: 13,
    fontWeight: '600',
  },
  logoutBtn: {
    backgroundColor: '#fdf2f8',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#fbcfe8',
  },
  logoutBtnText: {
    color: '#ec4899',
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
    width: 210,
    backgroundColor: '#ffffff',
    paddingTop: 28,
    paddingHorizontal: 14,
    borderRightWidth: 1,
    borderRightColor: '#EDE9FE',
    shadowColor: '#7c3aed',
    shadowOffset: { width: 2, height: 0 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
  },
  sidebarSection: {
    fontSize: 10,
    fontWeight: '700',
    color: '#c4b5fd',
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    marginBottom: 12,
    paddingLeft: 10,
  },
  navItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 11,
    paddingHorizontal: 12,
    borderRadius: 12,
    marginBottom: 4,
  },
  navItemActive: {
    backgroundColor: '#EDE9FE',
  },
  navIcon: {
    width: 30,
    height: 30,
    borderRadius: 8,
    backgroundColor: '#F3F2FF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  navIconActive: {
    backgroundColor: '#7c3aed',
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 6,
  },
  navIconText: {
    color: '#a78bfa',
    fontWeight: '700',
    fontSize: 12,
  },
  navIconTextActive: {
    color: '#fff',
  },
  navLabel: {
    color: '#7c6f9e',
    fontSize: 14,
    fontWeight: '500',
  },
  navLabelActive: {
    color: '#6d28d9',
    fontWeight: '700',
  },
  sidebarFooter: {
    position: 'absolute',
    bottom: 20,
    left: 14,
    right: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#F3F2FF',
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: '#EDE9FE',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#10b981',
  },
  sidebarFooterText: {
    color: '#7c3aed',
    fontSize: 12,
    fontWeight: '600',
  },

  // ── CONTENT
  content: {
    flex: 1,
    backgroundColor: '#F3F2FF',
  },
  pageTitleBar: {
    paddingHorizontal: 24,
    paddingTop: 20,
    paddingBottom: 14,
    backgroundColor: '#F3F2FF',
    borderBottomWidth: 1,
    borderBottomColor: '#EDE9FE',
  },
  pageTitle: {
    fontSize: 22,
    fontWeight: '800',
    color: '#1e1b4b',
    letterSpacing: 0.1,
  },
  pageSubtitle: {
    fontSize: 13,
    color: '#a78bfa',
    marginTop: 3,
    fontWeight: '500',
  },
});
