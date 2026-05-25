import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Alert,
  Image,
  Keyboard,
  KeyboardAvoidingView,
  Modal,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  TouchableWithoutFeedback,
  View
} from 'react-native';

import { ProfilePicture } from '../../components/profile/profilePicture';
import { CustomInput } from '../../components/ui/customInput';
import { useThemeColors } from '../../hooks/useThemeColors';
import { useUserStore } from '../../store/useUserStore';

export default function ProfileManagerScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  const { userName, userImage, setUser } = useUserStore();

  const [profileImage, setProfileImage] = useState<string | null>(userImage);
  const [fullName, setFullName] = useState(userName || '');
  const [phoneNumber, setPhoneNumber] = useState('+963 9456 789 147');
  const [isFullImageVisible, setIsFullImageVisible] = useState(false);

  const handlePickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(t('profile.permission_title'), t('profile.permission_message'));
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setProfileImage(result.assets[0].uri);
    }
  };

  const handleUpdate = () => {
    if (fullName.trim().length < 3) {
      Alert.alert(t('common.error'), t('profile.invalid_name'));
      return;
    }

    setUser({
      name: fullName,
      image: profileImage,
    });

    Alert.alert(t('common.success'), t('profile.update_success'), [
      { text: t('common.ok'), onPress: () => router.back() }
    ]);
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={{ flex: 1 }}
      >
        <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
          <View style={{ flex: 1 }}>
            <View style={styles.header}>
              <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
                <Ionicons name="chevron-back" size={28} color={Colors.primary} />
              </TouchableOpacity>
              <Text style={styles.headerTitle}>{t('profile.edit_title')}</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
              <View style={styles.imageSection}>
                <ProfilePicture
                  imageUri={profileImage}
                  size={135}
                  showControls
                  onPick={handlePickImage}
                  onRemove={() => setProfileImage(null)}
                  onView={() => profileImage && setIsFullImageVisible(true)}
                />
              </View>

              <View style={styles.form}>
                <CustomInput
                  label={t('profile.full_name')}
                  value={fullName}
                  onChangeText={setFullName}
                  placeholder={t('profile.full_name_placeholder')}
                />

                <CustomInput
                  label={t('profile.phone')}
                  value={phoneNumber}
                  onChangeText={setPhoneNumber}
                  keyboardType="phone-pad"
                  placeholder="+963 --- --- ---"
                />
              </View>

              <TouchableOpacity style={styles.updateButton} onPress={handleUpdate} activeOpacity={0.8}>
                <Text style={styles.buttonText}>{t('profile.update')}</Text>
              </TouchableOpacity>

              <View style={{ height: 40 }} />
            </ScrollView>
          </View>
        </TouchableWithoutFeedback>
      </KeyboardAvoidingView>

      <Modal visible={isFullImageVisible} transparent animationType="fade">
        <View style={styles.modalBg}>
          <TouchableOpacity style={styles.closeBtn} onPress={() => setIsFullImageVisible(false)}>
            <Ionicons name="close" size={35} color="white" />
          </TouchableOpacity>
          {profileImage && (
            <Image source={{ uri: profileImage }} style={styles.fullImg} resizeMode="contain" />
          )}
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    height: 60, marginTop: Platform.OS === 'android' ? 40 : 10
  },
  imageSection: { alignItems: 'center', marginBottom: 30 },
  backButton: { position: 'absolute', left: 20 },
  headerTitle: { fontSize: 22, fontWeight: 'bold', color: Colors.primary },
  scrollContent: { paddingHorizontal: 30, paddingTop: 30 },
  form: { marginTop: 40 },
  updateButton: {
    backgroundColor: Colors.primary, height: 60, borderRadius: 30,
    justifyContent: 'center', alignItems: 'center', marginTop: 20, elevation: 5
  },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  modalBg: { flex: 1, backgroundColor: 'rgba(0,0,0,0.9)', justifyContent: 'center', alignItems: 'center' },
  fullImg: { width: '90%', height: '70%' },
  closeBtn: { position: 'absolute', top: 50, right: 25 },
});
