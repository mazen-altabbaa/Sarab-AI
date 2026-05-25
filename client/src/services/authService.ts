import AsyncStorage from '@react-native-async-storage/async-storage';
import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';

export type SignupPayload = {
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  role: string;
};

export type LoginPayload = {
  email: string;
  password: string;
};

export type AuthUser = {
  id: number;
  firstName: string;
  lastName: string;
  email: string;
  role: number;
};

export type AuthResponse = {
  token: string;
  refreshToken: string;
  expiresAt: string;
  user: AuthUser;
};

export type StoredAuthData = {
  token: string | null;
  refreshToken: string | null;
  expiresAt: string | null;
  user: AuthUser | null;
};

const getApiBaseUrl = () => {
  const rawUrl = process.env.EXPO_PUBLIC_API_URL ?? 'http://25.19.119.206:5027';
  return rawUrl.trim().replace(/^['"]|['"]$/g, '');
};

const BASE_URL = getApiBaseUrl();

const STORAGE_KEYS = {
  token: 'sarab_auth_token',
  refreshToken: 'sarab_auth_refresh_token',
  expiresAt: 'sarab_auth_expires_at',
  user: 'sarab_auth_user',
};

const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
});

const attachTokenToRequest = async (config: InternalAxiosRequestConfig) => {
  const token = await AsyncStorage.getItem(STORAGE_KEYS.token);
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
};

api.interceptors.request.use(attachTokenToRequest, (error) => Promise.reject(error));

function isTokenExpired(expiresAt: string | null): boolean {
  if (!expiresAt) return true;
  const expiresTime = new Date(expiresAt).getTime();
  if (Number.isNaN(expiresTime)) return true;
  return expiresTime <= Date.now() + 60_000;
}

function normalizeAuthError(error: unknown): Error {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status;
    const data = error.response?.data;

    if (__DEV__) {
      console.log('Auth API error:', {
        baseUrl: BASE_URL,
        url: error.config?.url,
        method: error.config?.method,
        status,
        code: error.code,
        message: error.message,
        data,
      });
    }

    if (!error.response) {
      if (error.code === 'ECONNABORTED') {
        return new Error('Connection timed out. Please check the server connection and try again.');
      }

      return new Error('Cannot reach the server. Please check your internet connection and try again.');
    }

    const details = formatServerError(data);
    return new Error(details || getFallbackAuthMessage(status));
  }

  return error instanceof Error ? error : new Error('An unexpected error occurred');
}

function formatServerError(data: unknown): string | null {
  if (!data) return null;
  if (typeof data === 'string') return data;
  if (Array.isArray(data)) return data.map(String).join('\n');
  if (typeof data !== 'object') return String(data);

  const body = data as Record<string, unknown>;
  const validationErrors = body.errors;

  if (validationErrors && typeof validationErrors === 'object') {
    const messages = Object.entries(validationErrors as Record<string, unknown>)
      .flatMap(([field, value]) => {
        if (Array.isArray(value)) return value.map((item) => `${formatFieldName(field)}: ${String(item)}`);
        return `${formatFieldName(field)}: ${String(value)}`;
      })
      .filter(Boolean);

    if (messages.length > 0) return messages.join('\n');
  }

  const directMessage = body.message || body.error || body.detail || body.title;
  if (Array.isArray(directMessage)) return directMessage.map(String).join('\n');
  if (directMessage) return String(directMessage);

  return JSON.stringify(body, null, 2);
}

function formatFieldName(field: string): string {
  const cleanField = field.replace(/^\$\./, '').replace(/^model\./i, '');
  const labels: Record<string, string> = {
    email: 'Email',
    password: 'Password',
    firstName: 'First name',
    lastName: 'Last name',
    role: 'Role',
    refreshToken: 'Refresh token',
  };

  return labels[cleanField] ?? cleanField;
}

function getFallbackAuthMessage(status?: number): string {
  if (status === 400) return 'Please check the entered information and try again.';
  if (status === 401) return 'Email or password is incorrect.';
  if (status === 403) return 'You do not have permission to perform this action.';
  if (status === 409) return 'An account with this information already exists.';
  if (status && status >= 500) return 'Server error. Please try again later.';
  return 'Could not complete the request. Please try again.';
}

async function storeAuthData(data: AuthResponse): Promise<void> {
  await AsyncStorage.multiSet([
    [STORAGE_KEYS.token, data.token],
    [STORAGE_KEYS.refreshToken, data.refreshToken || ''],
    [STORAGE_KEYS.expiresAt, data.expiresAt],
    [STORAGE_KEYS.user, JSON.stringify(data.user)],
  ]);
}

export async function clearStoredAuthData(): Promise<void> {
  await AsyncStorage.multiRemove([
    STORAGE_KEYS.token,
    STORAGE_KEYS.refreshToken,
    STORAGE_KEYS.expiresAt,
    STORAGE_KEYS.user,
  ]);
}

export async function getStoredAuthData(): Promise<StoredAuthData> {
  const keys = [STORAGE_KEYS.token, STORAGE_KEYS.refreshToken, STORAGE_KEYS.expiresAt, STORAGE_KEYS.user];
  const stores = await AsyncStorage.multiGet(keys);

  const data = stores.reduce((acc, [key, value]) => {
    acc[key] = value;
    return acc;
  }, {} as Record<string, string | null>);

  return {
    token: data[STORAGE_KEYS.token] ?? null,
    refreshToken: data[STORAGE_KEYS.refreshToken] ?? null,
    expiresAt: data[STORAGE_KEYS.expiresAt] ?? null,
    user: data[STORAGE_KEYS.user] ? JSON.parse(data[STORAGE_KEYS.user] as string) : null,
  };
}

export async function signup(payload: SignupPayload): Promise<AuthResponse> {
  try {
    const response = await api.post('/api/Auth/signup', payload);
    const data = response.data as AuthResponse;
    await storeAuthData(data);
    return data;
  } catch (error) {
    throw normalizeAuthError(error);
  }
}

export async function login(payload: LoginPayload): Promise<AuthResponse> {
  try {
    const response = await api.post('/api/Auth/login', payload);
    const data = response.data as AuthResponse;
    await storeAuthData(data);
    return data;
  } catch (error) {
    throw normalizeAuthError(error);
  }
}

export async function refreshToken(): Promise<AuthResponse> {
  const refreshTokenValue = await AsyncStorage.getItem(STORAGE_KEYS.refreshToken);
  if (!refreshTokenValue) {
    throw new Error('No refresh token available');
  }

  try {
    const response = await axios.post(`${BASE_URL}/api/Auth/refresh`, {
      refreshToken: refreshTokenValue,
    });

    const data = response.data as AuthResponse;
    await storeAuthData(data);
    return data;
  } catch (error) {
    throw normalizeAuthError(error);
  }
}

export async function initializeAuthSession(): Promise<AuthResponse | null> {
  const stored = await getStoredAuthData();

  if (stored.token && stored.user && !isTokenExpired(stored.expiresAt)) {
    return {
      token: stored.token,
      refreshToken: stored.refreshToken ?? '',
      expiresAt: stored.expiresAt ?? '',
      user: stored.user,
    };
  }

  if (stored.refreshToken) {
    try {
      return await refreshToken();
    } catch (error) {
      await clearStoredAuthData();
      throw error;
    }
  }

  return null;
}

export async function logout(): Promise<void> {
  try {
    const refreshTokenValue = await AsyncStorage.getItem(STORAGE_KEYS.refreshToken);
    if (refreshTokenValue) {
      await api.post('/api/Auth/logout', { refreshToken: refreshTokenValue });
    }
  } catch (error) {
    console.warn('Logout request failed:', error);
  } finally {
    await clearStoredAuthData();
  }
}
