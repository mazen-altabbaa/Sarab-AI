import Constants from 'expo-constants';
import { Platform } from 'react-native';

const envUrl = process.env.EXPO_PUBLIC_API_URL;
const extra = (Constants.expoConfig?.extra ?? Constants.manifest?.extra) as Record<string, unknown> | undefined;
const extraUrl =
  typeof extra?.EXPO_PUBLIC_API_URL === 'string'
    ? extra.EXPO_PUBLIC_API_URL
    : typeof extra?.apiUrl === 'string'
    ? extra.apiUrl
    : undefined;

const rawUrl =
  (typeof envUrl === 'string' && envUrl.trim().replace(/^['"]|['"]$/g, '')) ||
  (typeof extraUrl === 'string' && extraUrl.trim().replace(/^['"]|['"]$/g, '')) ||
  'http://10.252.172.15:5027';

const ANDROID_EMULATOR_LOCALHOST = '10.0.2.2';

export const getApiBaseUrl = (): string => {
  const base = Platform.OS === 'android' && rawUrl.includes('localhost') ? rawUrl.replace('localhost', ANDROID_EMULATOR_LOCALHOST) : rawUrl;

  if (__DEV__) {
    console.log('getApiBaseUrl -> platform:', Platform.OS, 'resolvedBaseUrl:', base, 'envUrl:', envUrl, 'extraUrl:', extraUrl);
  }

  return base;
};
