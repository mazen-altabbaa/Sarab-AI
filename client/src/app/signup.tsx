import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ActivityIndicator,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';

import { SarabInput } from '../components/sarabInput';
import { useAuth } from '../hooks/useAuth';
import { useThemeColors } from '../hooks/useThemeColors';

export default function SignUpScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const Colors = useThemeColors();
  const { signup, isLoading, error, clearError } = useAuth();
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const role = 'Contributor';
  const canSubmit = Boolean(firstName.trim() && lastName.trim() && email.trim() && password.trim());
  const styles = createStyles(Colors);

  const validateEmail = (value: string) => /\S+@\S+\.\S+/.test(value.trim());
  const validatePassword = (value: string) => value.trim().length >= 8;

  const handleSignUp = async () => {
    if (!canSubmit) {
      setValidationError(t('auth.required_signup'));
      return;
    }

    if (!validateEmail(email)) {
      setValidationError(t('auth.invalid_email'));
      return;
    }

    if (!validatePassword(password)) {
      setValidationError(t('auth.short_password'));
      return;
    }

    setValidationError(null);

    try {
      await signup({ firstName: firstName.trim(), lastName: lastName.trim(), email: email.trim(), password: password.trim(), role });
      router.replace('/(tabs)');
    } catch (error) {
      const message = error instanceof Error ? error.message : t('auth.signup_failed');
      setValidationError(message);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar translucent backgroundColor={Colors.primary} barStyle="light-content" />
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.push('/')} style={styles.backButton}>
          <Ionicons name="chevron-back" size={28} color={Colors.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('auth.signup_title')}</Text>
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <SarabInput
          label={t('auth.first_name')}
          placeholder={t('auth.first_name_placeholder')}
          value={firstName}
          onChangeText={(text) => {
            setFirstName(text);
            setValidationError(null);
            clearError();
          }}
        />

        <SarabInput
          label={t('auth.last_name')}
          placeholder={t('auth.last_name_placeholder')}
          value={lastName}
          onChangeText={(text) => {
            setLastName(text);
            setValidationError(null);
            clearError();
          }}
        />

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

        <Text style={styles.termsText}>
          {t('auth.terms_prefix')} {'\n'}
          <Text style={styles.linkText}>{t('auth.terms')}</Text> and <Text style={styles.linkText}>{t('auth.privacy')}</Text>
        </Text>

        <TouchableOpacity
          style={[styles.mainButton, (!canSubmit || isLoading) && styles.disabledButton]}
          onPress={handleSignUp}
          disabled={!canSubmit || isLoading}
          activeOpacity={0.8}
          accessibilityState={{ disabled: !canSubmit || isLoading }}
        >
          {isLoading ? <ActivityIndicator color={Colors.white} /> : <Text style={styles.buttonText}>{t('auth.signup')}</Text>}
        </TouchableOpacity>

        <View style={styles.footer}>
          <Text style={styles.footerText}>{t('auth.already_have_account')}</Text>
          <TouchableOpacity onPress={() => router.push('/login')}>
            <Text style={styles.linkText}>{t('auth.login')}</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 60,
    marginTop: 40,
  },
  backButton: { position: 'absolute', left: 20 },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.primary
  },
  content: {
    paddingHorizontal: 30,
    paddingTop: 20,
    paddingBottom: 40,
    flexGrow: 1,
  },
  termsText: {
    textAlign: 'center',
    color: Colors.textSecondary,
    fontSize: 13,
    marginTop: 15,
    lineHeight: 20
  },
  mainButton: {
    backgroundColor: Colors.primary,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 25,
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
  errorText: {
    color: '#d32f2f',
    lineHeight: 20,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 30
  },
  footerText: { color: Colors.textSecondary, fontSize: 15 },
  linkText: {
    color: Colors.primary,
    fontWeight: 'bold',
    fontSize: 15
  }
});
