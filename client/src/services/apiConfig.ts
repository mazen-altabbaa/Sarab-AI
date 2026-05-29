import { Platform } from 'react-native';

const rawUrl = process.env.EXPO_PUBLIC_API_URL ?? 'http://10.252.172.15:5027';
const trimmedUrl = rawUrl.trim().replace(/^['"]|['"]$/g, '');

const ANDROID_EMULATOR_LOCALHOST = '10.0.2.2';

export const getApiBaseUrl = (): string => {
  if (Platform.OS === 'android' && trimmedUrl.includes('localhost')) {
    return trimmedUrl.replace('localhost', ANDROID_EMULATOR_LOCALHOST);
  }

  return trimmedUrl;
};
