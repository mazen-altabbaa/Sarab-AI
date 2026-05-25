import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';

interface ActionButtonsProps {
  colors: {
    primary: string;
    bgLight: string;
  };
  onCameraPress: () => void;
  onUploadPress: () => void;
  count: number;
  cameraLabel: string;
  uploadLabel: string;
  orLabel: string;
}

export const ActionButtons = ({
  colors,
  onCameraPress,
  onUploadPress,
  count,
  cameraLabel,
  uploadLabel,
  orLabel
}: ActionButtonsProps) => {
  const isDisabled = count >= 2;
  const styles = createStyles(colors);

  return (
    <View style={styles.actionContainer}>
      <TouchableOpacity
        style={[styles.inputBox, { backgroundColor: colors.bgLight }, isDisabled && styles.disabledBox]}
        onPress={onCameraPress}
        disabled={isDisabled}
      >
        <Text style={styles.inputPlaceholder}>{cameraLabel}</Text>
        <Ionicons name="camera-outline" size={24} color={colors.primary} />
      </TouchableOpacity>

      <Text style={styles.orText}>{orLabel}</Text>

      <TouchableOpacity
        style={[styles.inputBox, { backgroundColor: colors.bgLight }, isDisabled && styles.disabledBox]}
        onPress={onUploadPress}
        disabled={isDisabled}
      >
        <Text style={styles.inputPlaceholder}>{uploadLabel}</Text>
        <Ionicons name="cloud-upload-outline" size={24} color={colors.primary} />
      </TouchableOpacity>
    </View>
  );
};

const createStyles = (colors: ActionButtonsProps['colors']) => StyleSheet.create({
  actionContainer: {
    alignItems: 'center',
    gap: 10,
    marginBottom: 20,
    paddingHorizontal: 25
  },
  inputBox: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
    padding: 18,
    borderRadius: 18
  },
  disabledBox: { opacity: 0.4 },
  inputPlaceholder: {
    color: colors.primary,
    fontSize: 16,
    fontWeight: '600'
  },
  orText: {
    fontSize: 18,
    color: colors.primary,
    fontWeight: '300'
  },
});
