import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Platform, SafeAreaView, StyleSheet, Switch, Text, TouchableOpacity, View } from 'react-native';
import { DarkTheme, LightTheme } from '../../constants/theme';
import { useUserStore } from '../../store/useUserStore';

export default function SettingsScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const { language, setLanguage, theme, setTheme } = useUserStore();

  const Colors = theme === 'dark' ? DarkTheme : LightTheme;
  const isDark = theme === 'dark';
  const languageOptions = ['en', 'ar', 'ja', 'de'] as const;

  const toggleLanguage = () => {
    const currentIndex = languageOptions.indexOf(language as 'en' | 'ar' | 'ja' | 'de');
    const nextIndex = (currentIndex + 1) % languageOptions.length;
    setLanguage(languageOptions[nextIndex]);
  };

  const languageLabelKey = language === 'ar'
    ? 'arabic'
    : language === 'ja'
    ? 'japanese'
    : language === 'de'
    ? 'german'
    : 'english';


  const dynamicStyles = createStyles(Colors);

  return (
    <SafeAreaView style={dynamicStyles.container}>
      <View style={dynamicStyles.header}>
        <TouchableOpacity onPress={() => router.back()} style={dynamicStyles.backButton}>
          <Ionicons name="chevron-back" size={28} color={Colors.primary} />
        </TouchableOpacity>
        <Text style={dynamicStyles.headerTitle}>{t('settings.title')}</Text>
      </View>

      <View style={dynamicStyles.content}>
        <View style={dynamicStyles.menuItem}>
          <View style={dynamicStyles.menuItemLeft}>
            <View style={dynamicStyles.iconBox}>
              <Ionicons name={isDark ? 'moon' : 'sunny-outline'} size={22} color={Colors.primary} />
            </View>
            <View>
              <Text style={dynamicStyles.menuText}>{t('settings.dark_mode')}</Text>
              <Text style={dynamicStyles.subText}>{isDark ? t('settings.on') : t('settings.off')}</Text>
            </View>
          </View>
          <Switch
            value={isDark}
            onValueChange={() => setTheme(isDark ? 'light' : 'dark')}
            trackColor={{ false: '#767577', true: Colors.primary }}
            thumbColor={Platform.OS === 'ios' ? undefined : (isDark ? '#fff' : '#f4f3f4')}
          />
        </View>

        <TouchableOpacity style={dynamicStyles.menuItem} onPress={toggleLanguage} activeOpacity={0.7}>
          <View style={dynamicStyles.menuItemLeft}>
            <View style={dynamicStyles.iconBox}>
              <Ionicons name="language-outline" size={22} color={Colors.primary} />
            </View>
            <View>
              <Text style={dynamicStyles.menuText}>{t('settings.language')}</Text>
              <Text style={dynamicStyles.subText}>
                {t(`settings.${languageLabelKey}` as const)}
              </Text>
            </View>
          </View>
          <View style={dynamicStyles.langBadge}>
            <Text style={dynamicStyles.langBadgeText}>{language.toUpperCase()}</Text>
          </View>
        </TouchableOpacity>

      </View>
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
    marginTop: Platform.OS === 'android' ? 40 : 10,
  },
  backButton: { position: 'absolute', left: 20 },
  headerTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: Colors.primary
  },
  content: { paddingHorizontal: 25, marginTop: 20 },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 15,
    borderBottomWidth: 0.5,
    borderBottomColor: Colors.placeholder + '33',
  },
  menuItemLeft: { flexDirection: 'row', alignItems: 'center' },
  menuText: {
    fontSize: 17,
    color: Colors.text,
    fontWeight: '500'
  },
  subText: {
    fontSize: 13,
    color: Colors.placeholder,
    marginTop: 2
  },
  iconBox: {
    width: 45,
    height: 45,
    borderRadius: 12,
    backgroundColor: Colors.bgLight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  langBadge: {
    backgroundColor: Colors.bgLight,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  langBadgeText: {
    color: Colors.primary,
    fontSize: 12,
    fontWeight: 'bold',
  }
});
