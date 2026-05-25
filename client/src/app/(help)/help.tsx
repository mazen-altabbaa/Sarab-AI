import { FontAwesome5, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { useThemeColors } from '../../hooks/useThemeColors';

const ContactOption = ({ item, isOpen, onPress }: { item: any, isOpen: boolean, onPress: () => void }) => {
  const { t } = useTranslation();
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  return (
    <View style={styles.optionWrapper}>
      <TouchableOpacity 
        style={styles.optionItem} 
        onPress={onPress} 
        activeOpacity={0.7}
      >
        <View style={styles.optionLeft}>
          <View style={styles.iconCircle}>
            {item.type === 'MaterialCommunityIcons' ? (
              <MaterialCommunityIcons name={item.icon} size={24} color="#fff" />
            ) : (
              <FontAwesome5 name={item.icon} size={20} color="#fff" />
            )}
          </View>
          <Text style={styles.optionText}>{t(item.titleKey)}</Text>
        </View>
        <Ionicons 
          name={isOpen ? "chevron-up" : "chevron-down"} 
          size={24} 
          color={Colors.primary} 
        />
      </TouchableOpacity>
      
      {isOpen && (
        <View style={styles.accordionContent}>
          <Text style={styles.accordionText}>
            {t('help.response_message', { channel: t(item.titleKey) })}
          </Text>
        </View>
      )}
    </View>
  );
};

export default function HelpCenterScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const Colors = useThemeColors();
  const styles = createStyles(Colors);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const contactOptions = [
    { id: '1', titleKey: 'help.customer_service', icon: 'headphones', type: 'MaterialCommunityIcons' },
    { id: '2', titleKey: 'help.website', icon: 'earth', type: 'MaterialCommunityIcons' },
    { id: '3', titleKey: 'help.whatsapp', icon: 'whatsapp', type: 'FontAwesome5' },
    { id: '4', titleKey: 'help.instagram', icon: 'instagram', type: 'MaterialCommunityIcons' },
  ];

  const toggleAccordion = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.purpleHeader}>
        <View style={styles.headerTop}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="chevron-back" size={28} color="white" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{t('help.title')}</Text>
        </View>
        <Text style={styles.headerSubtitle}>{t('help.subtitle')}</Text>
      </View>

      <ScrollView 
        contentContainerStyle={styles.content} 
        showsVerticalScrollIndicator={false}
      >
        <TouchableOpacity style={styles.contactUsButton} activeOpacity={0.8}>
          <Text style={styles.contactUsText}>{t('help.contact_us')}</Text>
        </TouchableOpacity>

        {contactOptions.map((item) => (
          <ContactOption 
            key={item.id} 
            item={item} 
            isOpen={expandedId === item.id}
            onPress={() => toggleAccordion(item.id)}
          />
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  
  purpleHeader: {
    backgroundColor: Colors.primary,
    paddingTop: Platform.OS === 'android' ? 50 : 20,
    paddingBottom: 40,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 30, 
    borderBottomRightRadius: 30,
  },
  headerTop: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', height: 60 },
  backButton: { position: 'absolute', left: 0 },
  headerTitle: { fontSize: 24, fontWeight: 'bold', color: '#fff' },
  headerSubtitle: { 
    fontSize: 16, 
    color: 'rgba(255, 255, 255, 0.9)', 
    textAlign: 'center', 
    marginTop: 5 
  },

  content: { paddingHorizontal: 25, paddingTop: 30, paddingBottom: 50 },
  
  contactUsButton: {
    backgroundColor: Colors.primary,
    borderRadius: 25,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 30,
    width: '60%',
    alignSelf: 'center',
    elevation: 4,
    shadowColor: Colors.primary,
    shadowOpacity: 0.2,
    shadowRadius: 5,
  },
  contactUsText: { color: '#fff', fontSize: 16, fontWeight: 'bold' },

  optionWrapper: {
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderPurple,
    marginBottom: 5,
  },
  optionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 15,
  },
  optionLeft: { flexDirection: 'row', alignItems: 'center' },
  iconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  optionText: { fontSize: 16, fontWeight: '600', color: Colors.text },
  
  accordionContent: {
    paddingLeft: 59, 
    paddingBottom: 15,
    paddingRight: 10,
  },
  accordionText: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
});
