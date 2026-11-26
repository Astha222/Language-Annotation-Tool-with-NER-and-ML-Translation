import { Platform } from 'react-native';

// Where the backend API lives. Override it with EXPO_PUBLIC_API_URL (for example in frontend/.env:
//   EXPO_PUBLIC_API_URL=http://192.168.1.20:8000
// to test on a phone). The Android emulator reaches the host machine at 10.0.2.2.
export const BACKEND_URL: string =
  process.env.EXPO_PUBLIC_API_URL ||
  (Platform.OS === 'android'
    ? 'http://10.0.2.2:8000'
    : (typeof window !== 'undefined' && window.location?.hostname
        ? `${window.location.protocol}//${window.location.hostname}:8000`
        : 'http://localhost:8000'));
