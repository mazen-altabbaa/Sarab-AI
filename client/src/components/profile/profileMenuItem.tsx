import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useThemeColors } from '../../hooks/useThemeColors';

interface ProfileMenuItemProps {
  title: string;
  icon: string;
  onPress: () => void;
}

export const ProfileMenuItem = ({ title, icon, onPress }: ProfileMenuItemProps) => {
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  return (
    <TouchableOpacity style={styles.menuItem} onPress={onPress}>
      <View style={styles.menuItemLeft}>
        <View style={styles.iconBox}>
          <Ionicons name={icon as any} size={22} color={Colors.primary} />
        </View>
        <Text style={styles.menuText}>{title}</Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color={Colors.primary} />
    </TouchableOpacity>
  );
};

const createStyles = (Colors: any) => StyleSheet.create({
  menuItem: { 
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    marginBottom: 10 },
  menuItemLeft: { 
  flexDirection: 'row',
  alignItems: 'center' },
  iconBox: { width: 45,
  height: 45,
  borderRadius: 12,
  backgroundColor: Colors.bgLight,
  justifyContent: 'center',
  alignItems: 'center',
  marginRight: 15 },
  menuText: { 
  fontSize: 18,
  fontWeight: '500',
  color: Colors.text },
});
