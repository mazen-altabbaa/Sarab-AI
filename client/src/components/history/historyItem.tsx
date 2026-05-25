import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useThemeColors } from '../../hooks/useThemeColors';
import type { AnalysisResult } from '../../store/useUserStore';

interface HistoryItemProps {
  item: {
    id: string;
    title: string;
    time: string;
    date: string;
    result: AnalysisResult;
  };
  onPress: () => void;
}

export const HistoryItem = ({ item, onPress }: HistoryItemProps) => {
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  return (
    <TouchableOpacity 
      style={styles.historyItem} 
      onPress={onPress}
      activeOpacity={0.7}
    >
      <View style={styles.infoColumn}>
        <Text style={styles.itemTitle}>{item.title}</Text>
        <Text style={styles.timeText}>{item.time}</Text>
      </View>
      <View style={styles.dateColumn}>
        <Text style={styles.dateText}>{item.date}</Text>
      </View>
    </TouchableOpacity>
  );
};

const createStyles = (Colors: any) => StyleSheet.create({
  historyItem: {
    padding: 15,
    backgroundColor: Colors.bgLight,
    borderRadius: 15,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.borderPurple,
  },
  infoColumn: {
    flex: 1,
  },
  itemTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.primary,
    marginBottom: 4,
  },
  timeText: {
    fontSize: 13,
    color: Colors.primary,
    fontWeight: '500',
  },
  dateColumn: {
    alignItems: 'flex-end',
  },
  dateText: { 
    color: Colors.primary,
    fontSize: 12,
    fontWeight: 'bold'
  }
});
