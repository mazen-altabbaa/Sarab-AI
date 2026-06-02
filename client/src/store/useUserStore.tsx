import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import i18n from '../i18n';

export interface AnalysisResult {
  sampleId: number;
  message: string;
  timestamp?: string;
  results: {
    maps: {
      fullMap: string | null;
      left2right: string | null;
      right2left: string | null;
    } | null;
    trackingVideos: {
      left2right: string | null;
      right2left: string | null;
    } | null;
  };
}

interface UserState {
  token: string | null;
  userName: string | null;
  userImage: string | null;
  language: string;
  theme: 'light' | 'dark';
  isLoggedIn: boolean;

  pendingAnalysisResult: AnalysisResult | null;
  setPendingAnalysisResult: (result: AnalysisResult | null) => void;

  analysisHistory: AnalysisResult[];
  addToHistory: (result: AnalysisResult) => void;
  clearHistory: () => void;

  setUser: (data: { token?: string; name?: string; image?: string | null; language?: string }) => void;
  setLanguage: (lang: string) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  syncSettings: () => void;
  logout: () => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set, get) => ({
      token: null,
      userName: null,
      userImage: null,
      language: 'en',
      theme: 'light',
      isLoggedIn: false,
      pendingAnalysisResult: null,
      analysisHistory: [],

      setPendingAnalysisResult: (result) => set({ pendingAnalysisResult: result }),

      addToHistory: (result) => set((state) => ({
        analysisHistory: [result, ...state.analysisHistory]
      })),

      clearHistory: () => set({ analysisHistory: [] }),

      setUser: (data) => set((state) => {
        const newState = { ...state, isLoggedIn: true };
        if (data.token !== undefined) newState.token = data.token;
        if (data.name !== undefined) newState.userName = data.name;
        if (data.image !== undefined) newState.userImage = data.image;
        if (data.language) {
          newState.language = data.language;
          i18n.changeLanguage(data.language);
        }
        return newState;
      }),

      setLanguage: (lang) => {
        i18n.changeLanguage(lang);
        set({ language: lang });
      },

      setTheme: (theme) => set({ theme }),

      syncSettings: () => {
        const state = get();
        if (i18n.language !== state.language) {
          i18n.changeLanguage(state.language);
        }
      },

      logout: () => set({
        token: null,
        userName: null,
        userImage: null,
        isLoggedIn: false,
      }),
    }),
    {
      name: 'user-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        token: state.token,
        userName: state.userName,
        userImage: state.userImage,
        language: state.language,
        theme: state.theme,
        isLoggedIn: state.isLoggedIn,
        analysisHistory: state.analysisHistory,
      }),
      onRehydrateStorage: () => (state) => {
        state?.syncSettings();
      },
    }
  )
);
