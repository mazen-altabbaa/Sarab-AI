import { Ionicons } from '@expo/vector-icons';
import React, { useState } from 'react';
import { StyleSheet, Text, TextInput, TextInputProps, TouchableOpacity, View } from 'react-native';
import { useThemeColors } from '../../hooks/useThemeColors';

interface CustomInputProps extends TextInputProps {
  label: string;
  isPassword?: boolean;
}

export const CustomInput = ({ label, isPassword, ...props }: CustomInputProps) => {
  const [secure, setSecure] = useState(isPassword);
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  return (
    <View style={styles.container}>
      <Text style={styles.label}>{label}</Text>
      <View style={styles.wrapper}>
        <TextInput
          {...props}
          style={styles.input}
          secureTextEntry={secure}
          placeholderTextColor={Colors.placeholder}
        />
        {isPassword && (
          <TouchableOpacity onPress={() => setSecure(!secure)}>
            <Ionicons
              name={secure ? "eye-off-outline" : "eye-outline"}
              size={22}
              color={Colors.textSecondary}
            />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const createStyles = (Colors: any) => StyleSheet.create({
  container: { marginBottom: 20 },
  label: { fontSize: 16, fontWeight: 'bold', color: Colors.text, marginBottom: 10, marginLeft: 5 },
  wrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.inputBg,
    borderRadius: 15,
    paddingHorizontal: 15,
    height: 55,
    borderWidth: 1,
    borderColor: Colors.borderPurple,
  },
  input: { flex: 1, fontSize: 16, color: Colors.text },
});
