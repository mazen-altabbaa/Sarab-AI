import { LinearGradient } from 'expo-linear-gradient';
import { Redirect, useRouter } from 'expo-router';
import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withTiming
} from 'react-native-reanimated';

// استيراد المكونات والثوابت من ملفاتها الجديدة
import { SarabLogo } from '../components/sarabLogo';
import { useThemeColors } from '../hooks/useThemeColors';
import { useUserStore } from '../store/useUserStore';

export default function UnifiedAuthScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const Colors = useThemeColors();
  const styles = createStyles(Colors);
  const isLoggedIn = useUserStore((state) => state.isLoggedIn);

  // --- قيم الحركة ---
  const contentOpacity = useSharedValue(0); 
  const buttonsOpacity = useSharedValue(0); 
  const buttonsTranslateY = useSharedValue(20); 

  useEffect(() => {
    // 1. ظهور الشعار
    contentOpacity.value = withTiming(1, { duration: 1500 });

    // 2. ظهور الأزرار بعد تأخير (Splash محاكاة)
    buttonsOpacity.value = withDelay(3000, withTiming(1, { duration: 1000 }));
    buttonsTranslateY.value = withDelay(3000, withTiming(0, { duration: 1000 }));
  }, []);

  // ستايل الشعار المتحرك
  const animatedLogoStyle = useAnimatedStyle(() => ({
    opacity: contentOpacity.value,
  }));

  // ستايل الأزرار المتحركة
  const animatedButtonsStyle = useAnimatedStyle(() => ({
    opacity: buttonsOpacity.value,
    transform: [{ translateY: buttonsTranslateY.value }],
  }));

  if (isLoggedIn) {
    return <Redirect href="/(tabs)" />;
  }

  return (
    <View style={styles.container}>
      {/* شريط الحالة بلون التطبيق الأساسي حتى تظهر أيقونات النظام بوضوح */}
      <StatusBar translucent backgroundColor={Colors.primary} barStyle="light-content" />
      
      <LinearGradient
        colors={Colors.gradient}
        style={styles.background}
      >
        {/* الحاويات المسؤولة عن التموضع (Positioning) */}
        <Animated.View style={[styles.logoPositioner, animatedLogoStyle]}>
          <SarabLogo />
        </Animated.View>

        <Animated.View style={[styles.buttonContainer, animatedButtonsStyle]}>
          <TouchableOpacity 
            style={styles.loginButton} 
            onPress={() => router.push('/login')}
          >
            <Text style={styles.loginButtonText}>{t('auth.login')}</Text>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.signUpButton} 
            onPress={() => router.push('/signup')}
          >
            <Text style={styles.signUpButtonText}>{t('auth.signup')}</Text>
          </TouchableOpacity>
        </Animated.View>
      </LinearGradient>
      {/* اضفه مؤقتاً في أي مكان */}
      <TouchableOpacity onPress={() => router.push('/(tabs)')}>
        <Text style={{color: 'gray', textAlign: 'center'}}>Debug: Go to Home</Text>
      </TouchableOpacity>
    </View>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { 
    flex: 1 
  },
  background: { 
    flex: 1, 
    alignItems: 'center' 
  },
  // المسؤول عن وضع الشعار في المنتصف تماماً وبالارتفاع المطلوب
  logoPositioner: {
    position: 'absolute',
    top: '32%',
    alignItems: 'center',
    width: '100%',
  },
  buttonContainer: {
    position: 'absolute',
    bottom: 80, 
    width: '100%',
    paddingHorizontal: 40,
    gap: 15,
  },
  loginButton: {
    backgroundColor: Colors.transparentWhite, 
    height: 55,
    borderRadius: 27,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.borderWhite,
  },
  loginButtonText: { 
    color: Colors.white, 
    fontSize: 18, 
    fontWeight: '600' 
  },
  signUpButton: {
    backgroundColor: Colors.white,
    height: 55,
    borderRadius: 27,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  signUpButtonText: { 
    color: Colors.secondary, 
    fontSize: 18, 
    fontWeight: '600' 
  },
});
