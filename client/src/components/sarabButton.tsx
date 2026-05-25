import { FontAwesome } from '@expo/vector-icons';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useThemeColors } from '../hooks/useThemeColors';

interface SarabButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline' | 'social';
  icon?: string;
  loading?: boolean;
}

export const SarabButton = ({ title, onPress, variant = 'primary', icon, loading }: SarabButtonProps) => {
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  const buttonStyle = [
    styles.baseButton,
    variant === 'primary' && styles.primary,
    variant === 'secondary' && styles.secondary,
    variant === 'outline' && styles.outline,
    variant === 'social' && styles.social,
  ];

  const textStyle = [
    styles.baseText,
    variant === 'primary' && styles.textWhite,
    variant === 'secondary' && styles.textPurple,
    variant === 'outline' && styles.textWhite,
    variant === 'social' && styles.textMain,
  ];

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={loading}
      activeOpacity={0.8}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? '#fff' : Colors.primary} />
      ) : (
        <View style={styles.content}>
          {icon && <FontAwesome name={icon as any} size={20} color={Colors.primary} style={styles.icon} />}
          <Text style={textStyle}>{title}</Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const createStyles = (Colors: any) => StyleSheet.create({
  baseButton: {
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    flexDirection: 'row',
  },
  primary: { backgroundColor: Colors.primary, elevation: 4, shadowColor: '#000', shadowOpacity: 0.15, shadowRadius: 5 },
  secondary: { backgroundColor: Colors.card, elevation: 4 },
  outline: { backgroundColor: Colors.transparentWhite, borderWidth: 1, borderColor: Colors.borderWhite },
  social: { backgroundColor: Colors.softPurple, borderWidth: 1, borderColor: Colors.borderPurple, height: 55 },
  content: { flexDirection: 'row', alignItems: 'center' },
  icon: { marginRight: 10 },
  baseText: { fontSize: 18, fontWeight: 'bold' },
  textWhite: { color: Colors.white },
  textPurple: { color: Colors.primary },
  textMain: { color: Colors.textMain },
});
