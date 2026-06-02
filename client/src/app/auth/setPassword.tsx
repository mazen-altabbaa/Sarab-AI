import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import {
    Alert,
    Keyboard,
    KeyboardAvoidingView, Platform,
    SafeAreaView,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    TouchableWithoutFeedback,
    View
} from 'react-native';

import { CustomInput } from '../../components/ui/customInput';
import { useThemeColors } from '../../hooks/useThemeColors';

export default function SetPasswordScreen() {
  const router = useRouter();
  const Colors = useThemeColors();
  const styles = createStyles(Colors);
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const handleCreate = () => {
    if (password !== confirmPassword) {
      Alert.alert("خطأ", "كلمات المرور غير متطابقة");
      return;
    }
    console.log("Password Set!");
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={{ flex: 1 }}>
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
          <View style={{ flex: 1 }}>
            
            <View style={styles.header}>
              <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
                <Ionicons name="chevron-back" size={28} color={Colors.primary} />
              </TouchableOpacity>
              <Text style={styles.headerTitle}>Set Password</Text>
            </View>

            <ScrollView contentContainerStyle={styles.content}>
              <Text style={styles.subtitle}>Enter your new password below.</Text>

              <CustomInput 
                label="Password"
                placeholder="***************"
                isPassword
                value={password}
                onChangeText={setPassword}
              />

              <CustomInput 
                label="Confirm Password"
                placeholder="***************"
                isPassword
                value={confirmPassword}
                onChangeText={setConfirmPassword}
              />

              <TouchableOpacity style={styles.button} onPress={handleCreate}>
                <Text style={styles.buttonText}>Create New Password</Text>
              </TouchableOpacity>
            </ScrollView>

          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: { 
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', 
    height: 60, marginTop: Platform.OS === 'android' ? 40 : 10 
  },
  backButton: { position: 'absolute', left: 20 },
  headerTitle: { fontSize: 22, fontWeight: 'bold', color: Colors.primary },
  content: { paddingHorizontal: 30, paddingTop: 30 },
  subtitle: { fontSize: 16, color: Colors.textSecondary, marginBottom: 30, textAlign: 'center' },
  button: { 
    backgroundColor: Colors.primary, height: 60, borderRadius: 30, 
    justifyContent: 'center', alignItems: 'center', marginTop: 30,
    elevation: 5, shadowColor: Colors.primary, shadowOpacity: 0.3, shadowRadius: 5 
  },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
});
