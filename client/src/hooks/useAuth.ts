import { useCallback, useState } from 'react';
import {
  initializeAuthSession,
  login as loginRequest,
  logout as logoutRequest,
  refreshToken as refreshRequest,
  signup as signupRequest,
} from '../services/authService';
import type { AuthResponse, LoginPayload, SignupPayload } from '../services/authService';
import { useUserStore } from '../store/useUserStore';

const getErrorMessage = (error: unknown) => {
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred';
};

const getUserName = (data: AuthResponse) => `${data.user.firstName} ${data.user.lastName}`.trim();

const syncUserStore = (data: AuthResponse) => {
  useUserStore.getState().setUser({
    token: data.token,
    name: getUserName(data),
  });
};

export function useAuth() {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<Record<string, unknown> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const initializeSession = useCallback(async () => {
    setIsInitializing(true);
    setError(null);
    try {
      const data = await initializeAuthSession();
      if (data) {
        setToken(data.token);
        setUser(data.user);
        syncUserStore(data);
      } else {
        setToken(null);
        setUser(null);
        useUserStore.getState().logout();
      }
      return data;
    } catch (error) {
      const message = getErrorMessage(error);
      setToken(null);
      setUser(null);
      setError(message);
      useUserStore.getState().logout();
      return null;
    } finally {
      setIsInitializing(false);
    }
  }, []);

  const signup = useCallback(async (payload: SignupPayload) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await signupRequest(payload);
      setToken(data.token);
      setUser(data.user);
      syncUserStore(data);
      return data;
    } catch (error) {
      const message = getErrorMessage(error);
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (payload: LoginPayload) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await loginRequest(payload);
      setToken(data.token);
      setUser(data.user);
      syncUserStore(data);
      return data;
    } catch (error) {
      const message = getErrorMessage(error);
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshToken = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await refreshRequest();
      setToken(data.token);
      setUser(data.user);
      syncUserStore(data);
      return data;
    } catch (error) {
      const message = getErrorMessage(error);
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      await logoutRequest();
      setToken(null);
      setUser(null);
      useUserStore.getState().logout();
    } catch (error) {
      const message = getErrorMessage(error);
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return {
    token,
    user,
    isAuthenticated: Boolean(token),
    isLoading,
    isInitializing,
    error,
    initializeSession,
    signup,
    login,
    refreshToken,
    logout,
    clearError,
  };
}
