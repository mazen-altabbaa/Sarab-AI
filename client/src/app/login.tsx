import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ActivityIndicator, KeyboardAvoidingView, Platform, SafeAreaView, StatusBar, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { SarabInput } from '../components/sarabInput';
import { useAuth } from '../hooks/useAuth';
import { useThemeColors } from '../hooks/useThemeColors';

export default function LoginScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const Colors = useThemeColors();
  const { login, isLoading, error, clearError } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const canSubmit = Boolean(email.trim() && password.trim());
  const styles = createStyles(Colors);

  const validateEmail = (value: string) => /\S+@\S+\.\S+/.test(value.trim());

  const handleLogin = async () => {
    if (!canSubmit) {
      setValidationError(t('auth.required_login'));
      return;
    }

    if (!validateEmail(email)) {
      setValidationError(t('auth.invalid_email'));
      return;
    }

    if (password.length < 8) {
      setValidationError(t('auth.short_password'));
      return;
    }

    setValidationError(null);

    try {
      await login({ email: email.trim(), password: password.trim() });
      router.replace('/(tabs)');
    } catch (error) {
      const message = error instanceof Error ? error.message : t('auth.login_failed');
      setValidationError(message);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar translucent backgroundColor={Colors.primary} barStyle="light-content" />

      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={{ flex: 1 }}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.push('/')} style={styles.backButton}>
            <Ionicons name="chevron-back" size={28} color={Colors.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{t('auth.login_title')}</Text>
        </View>

        <View style={styles.content}>
          <Text style={styles.welcomeText}>{t('auth.welcome')}</Text>

          <SarabInput
            label={t('auth.email')}
            placeholder={t('auth.email_placeholder')}
            keyboardType="email-address"
            value={email}
            onChangeText={(text) => {
              setEmail(text);
              setValidationError(null);
              clearError();
            }}
          />

          <SarabInput
            label={t('auth.password')}
            placeholder={t('auth.password_placeholder')}
            isPassword
            secureTextEntry={!passwordVisible}
            togglePassword={() => setPasswordVisible(!passwordVisible)}
            value={password}
            onChangeText={(text) => {
              setPassword(text);
              setValidationError(null);
              clearError();
            }}
          />

          {validationError || error ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{validationError || error}</Text>
            </View>
          ) : null}

          <TouchableOpacity style={styles.forgetPassword}>
            <Text style={styles.forgetText}>{t('auth.forgot_password')}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.mainButton, (!canSubmit || isLoading) && styles.disabledButton]}
            onPress={handleLogin}
            disabled={!canSubmit || isLoading}
            activeOpacity={0.8}
            accessibilityState={{ disabled: !canSubmit || isLoading }}
          >
            {isLoading ? <ActivityIndicator color={Colors.white} /> : <Text style={styles.buttonText}>{t('auth.login')}</Text>}
          </TouchableOpacity>

          {/* Social login removed: Google button hidden per design */}

          <View style={styles.footer}>
            <Text style={styles.footerText}>{t('auth.dont_have_account')}</Text>
            <TouchableOpacity onPress={() => router.push('/signup')}>
              <Text style={styles.linkText}>{t('auth.signup')}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', height: 60, marginTop: 40 },
  backButton: { position: 'absolute', left: 20 },
  headerTitle: { fontSize: 24, fontWeight: 'bold', color: Colors.primary },
  content: { flex: 1, paddingHorizontal: 30, paddingTop: 30 },
  welcomeText: { fontSize: 32, fontWeight: 'bold', color: Colors.primary, marginBottom: 40 },
  forgetPassword: { alignSelf: 'flex-end', marginTop: -10, marginBottom: 20 },
  forgetText: { color: Colors.primary, fontSize: 14, fontWeight: '500' },
  mainButton: {
    backgroundColor: Colors.primary,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 5,
  },
  disabledButton: {
    backgroundColor: Colors.softPurple,
    opacity: 0.65,
  },
  buttonText: { color: Colors.white, fontSize: 20, fontWeight: 'bold' },
  errorBox: {
    backgroundColor: 'rgba(211, 47, 47, 0.08)',
    borderColor: 'rgba(211, 47, 47, 0.25)',
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  errorText: { color: '#d32f2f', lineHeight: 20 },
  orText: { textAlign: 'center', color: Colors.textSecondary, marginVertical: 25 },
  footer: { flexDirection: 'row', justifyContent: 'center', marginTop: 'auto', marginBottom: 30 },
  footerText: { color: Colors.textSecondary, fontSize: 15 },
  linkText: { color: Colors.primary, fontWeight: 'bold', fontSize: 15 },
});
