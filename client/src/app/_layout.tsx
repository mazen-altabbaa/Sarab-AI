import { ClerkProvider } from '@clerk/expo';
import { tokenCache } from '@clerk/expo/token-cache';
import * as NavigationBar from 'expo-navigation-bar';
import { Stack } from "expo-router";
import { useEffect } from 'react';
import { Platform, StatusBar, View, AppState } from 'react-native';
import { useAuth } from '../hooks/useAuth';
import { useThemeColors } from '../hooks/useThemeColors';
import "../../global.css";

const publishableKey = process.env.EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY!

if (!publishableKey) {
  throw new Error('Add your Clerk Publishable Key to the .env file')
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
      // إخفاء شريط التنقل دائماً وعدم السماح له بالظهور
      const hideNav = () => {
        NavigationBar.setVisibilityAsync('hidden');
        // حافظ على سلوك overlay-swipe (مسموح من النوع)
        NavigationBar.setBehaviorAsync('overlay-swipe');
      };
      hideNav();

      // إعادة إخفاء الشريط عندما يعود التطبيق للنشاط
      const sub = AppState.addEventListener('change', (state) => {
        if (state === 'active') hideNav();
      });
      return () => sub.remove?.();
    }
  }, []);
  return <ClerkProvider publishableKey={publishableKey} tokenCache={tokenCache}>
      <AuthSessionBootstrap />
      {/* Top status-area background so system icons are visible on light backgrounds */}
      <View style={{
        height: Platform.OS === 'android' ? StatusBar.currentHeight ?? 24 : 44,
        backgroundColor: Colors.background === '#121212' ? Colors.tabBarBackground : Colors.primary,
        width: '100%'
      }} />
      <StatusBar translucent backgroundColor={Colors.primary} barStyle={Colors.background === '#121212' ? 'light-content' : 'dark-content'} />
      <Stack screenOptions={{headerShown: false}}/>
    </ClerkProvider>
}
