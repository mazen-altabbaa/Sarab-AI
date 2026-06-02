import { ClerkProvider } from '@clerk/expo';
import { tokenCache } from '@clerk/expo/token-cache';
import * as NavigationBar from 'expo-navigation-bar';
import { Stack } from "expo-router";
import { useEffect } from 'react';
import { Platform, StatusBar, View, AppState } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { useThemeColors } from '../hooks/useThemeColors';
import '../../global.css';

const publishableKey = process.env.EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY ?? '';

if (!publishableKey) {
  // Don't throw during local development — show a clear warning so dev can continue.
  // In production you should provide a real publishable key via .env or build config.
  // eslint-disable-next-line no-console
  console.warn('EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY is not set. Clerk auth will be disabled in this session.');
}

function AuthSessionBootstrap() {
  const { initializeSession } = useAuth();

  useEffect(() => {
    initializeSession();
  }, [initializeSession]);

  return null;
}

export default function RootLayout() {
  const Colors = useThemeColors();

  useEffect(() => {
    if (Platform.OS === 'android') {
      const hideNav = () => {
        NavigationBar.setVisibilityAsync('hidden');
        NavigationBar.setBehaviorAsync('overlay-swipe');
      };
      hideNav();

      const sub = AppState.addEventListener('change', (state) => {
        if (state === 'active') hideNav();
      });
      return () => sub.remove?.();
    }
  }, []);
  // If publishableKey is missing, render layout without ClerkProvider so app can run in dev.
  const content = (
    <>
      <AuthSessionBootstrap />
      <View style={{
        height: Platform.OS === 'android' ? StatusBar.currentHeight ?? 24 : 44,
        backgroundColor: Colors.background === '#121212' ? Colors.tabBarBackground : Colors.primary,
        width: '100%'
      }} />
      <StatusBar translucent backgroundColor={Colors.primary} barStyle={Colors.background === '#121212' ? 'light-content' : 'dark-content'} />
      <Stack screenOptions={{headerShown: false}}/>
    </>
  );

  return publishableKey ? (
    <ClerkProvider publishableKey={publishableKey} tokenCache={tokenCache}>{content}</ClerkProvider>
  ) : (
    content
  );
}
