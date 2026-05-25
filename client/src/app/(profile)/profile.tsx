import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Alert,
  BackHandler,
  Image,
  Modal,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';

import { ProfileMenuItem } from '../../components/profile/profileMenuItem';
import { ProfilePicture } from '../../components/profile/profilePicture';
import { useAuth } from '../../hooks/useAuth';
import { useThemeColors } from '../../hooks/useThemeColors';
import { useUserStore } from '../../store/useUserStore';

export default function ProfileScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const Colors = useThemeColors();
  const styles = createStyles(Colors);
  const { logout } = useAuth();
  
  const { userName, userImage } = useUserStore();
  
  const [isFullImageVisible, setIsFullImageVisible] = useState(false);

  const handleBack = () => {
    router.back();
  };

  useEffect(() => {
    const backAction = () => { handleBack(); return true; };
    const backHandler = BackHandler.addEventListener('hardwareBackPress', backAction);
    return () => backHandler.remove();
  }, []);

  const handleLogout = () => {
    Alert.alert(t('profile.logout'), t('profile.logout_message'), [
      { text: t('common.cancel'), style: "cancel" },
      { 
        text: t('profile.logout'),
        style: "destructive", 
        onPress: async () => {
          await logout();
          router.replace('/login'); 
        } 
      }
    ]);
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBack} style={styles.backButton}>
          <Ionicons name="chevron-back" size={28} color={Colors.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('profile.title')}</Text>
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        
  
        <View style={styles.imageSection}>
          <ProfilePicture 
            imageUri={userImage}
            size={125}
            showControls={false} 
            onView={() => userImage && setIsFullImageVisible(true)}
          />
          <Text style={styles.userName}>{userName || t('profile.fallback_name')}</Text>
        </View>

        <View style={styles.menuContainer}>
          <ProfileMenuItem 
            title={t('profile.manager')}
            icon="person-outline" 
            onPress={() => router.push('/profilemanager')} 
          />
          <ProfileMenuItem 
            title={t('profile.settings')}
            icon="settings-outline" 
            onPress={() => router.push('/settings')} 
          />
          <ProfileMenuItem 
            title={t('profile.help')}
            icon="help-circle-outline" 
            onPress={() => router.push('/help')} 
          />
          <ProfileMenuItem 
            title={t('profile.logout')}
            icon="log-out-outline" 
            onPress={handleLogout} 
          />
        </View>
      </ScrollView>

      <Modal visible={isFullImageVisible} transparent animationType="fade">
        <View style={styles.modalBackground}>
          <TouchableOpacity style={styles.closeModal} onPress={() => setIsFullImageVisible(false)}>
            <Ionicons name="close" size={35} color="white" />
          </TouchableOpacity>
          {userImage && (
            <Image source={{ uri: userImage }} style={styles.fullImagePreview} resizeMode="contain" />
          )}
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 60,
    marginTop: 40,
  },
  backButton: { position: 'absolute', left: 20 },
  headerTitle: { fontSize: 22, fontWeight: 'bold', color: Colors.primary },
  content: { alignItems: 'center', paddingTop: 20, paddingBottom: 40 },
  imageSection: { alignItems: 'center', marginBottom: 40 },
  imageWrapper: { position: 'relative' },
  
  profileImage: { 
    width: 125, 
    height: 125, 
    borderRadius: 62.5, 
    backgroundColor: '#f3f0ff',
    borderWidth: 2,
    borderColor: '#e1d7f5',
    overflow: 'hidden', 
    justifyContent: 'center',
    alignItems: 'center',
  },
  
  absoluteCenter: {
    position: 'absolute',
    left: -10, right: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },

  fullSizeImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },

  editBadge: {
    position: 'absolute',
    right: -2,
    bottom: 5,
    backgroundColor: Colors.primary,
    width: 34,
    height: 34,
    borderRadius: 17,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: 'white',
    zIndex: 2,
  },
  deleteBadge: {
    position: 'absolute',
    left: -2,
    bottom: 5,
    backgroundColor: '#ff4d4d',
    width: 34,
    height: 34,
    borderRadius: 17,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: 'white',
    zIndex: 2,
  },
  userName: { fontSize: 24, fontWeight: 'bold', marginTop: 15, color: Colors.text },
  menuContainer: { width: '100%', paddingHorizontal: 25 },

  modalBackground: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  fullImagePreview: {
    width: '90%',
    height: '70%',
  },
  closeModal: {
    position: 'absolute',
    top: 50,
    right: 25,
  }
});
