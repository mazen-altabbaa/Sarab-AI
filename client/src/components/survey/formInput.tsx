import React from 'react';
import { StyleSheet, TextInput } from 'react-native';
import { useThemeColors } from '../../hooks/useThemeColors';

export const FormInput = ({ style, ...props }: any) => {
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  return (
    <TextInput
      style={[styles.input, style]}
      placeholderTextColor={Colors.primary}
      {...props}
    />
  );
};

const createStyles = (Colors: any) => StyleSheet.create({
  input: { 
    backgroundColor: Colors.bgLight,
    padding: 16, 
    borderRadius: 15, 
    color: Colors.text,
    fontSize: 16
  }
});
