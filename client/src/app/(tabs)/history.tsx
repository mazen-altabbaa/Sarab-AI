import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FlatList, StyleSheet, Text, View } from 'react-native';
import { HistoryItem } from '../../components/history/historyItem';
import { AnalysisResultModal } from '../../components/home/AnalysisResultModal';
import { useThemeColors } from '../../hooks/useThemeColors';
import { useUserStore } from '../../store/useUserStore';
import type { AnalysisResult } from '../../store/useUserStore';

export default function HistoryScreen() {
  const Colors = useThemeColors();
  const { t } = useTranslation();
  const { analysisHistory } = useUserStore();
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const styles = createStyles(Colors);

  const handleItemPress = (result: AnalysisResult) => {
    setSelectedResult(result);
    setModalVisible(true);
  };

  const getCurrentDateTime = () => {
    const now = new Date();
    const date = now.toISOString().split('T')[0];
    const time = now.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });
    return { date, time };
  };

  const historyItems = analysisHistory.map((result, index) => {
    const date = new Date(result.timestamp || Date.now());
    const dateStr = date.toISOString().split('T')[0];
    const timeStr = date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    });
    const sampleId = result.sampleId ?? index;
    return {
      id: sampleId.toString(),
      title: t('history.item_title', { id: sampleId }),
      date: dateStr,
      time: timeStr,
      result
    };
  });

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{t('history.title')}</Text>
      
      <FlatList
        data={historyItems}
        ListEmptyComponent={<Text style={styles.emptyText}>{t('history.empty')}</Text>}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        renderItem={({ item }) => (
          <HistoryItem 
            item={item} 
            onPress={() => handleItemPress(item.result)} 
          />
        )}
      />

      <AnalysisResultModal 
        visible={modalVisible} 
        onClose={() => setModalVisible(false)} 
        data={selectedResult}
      />
    </View>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: Colors.background,
    paddingTop: 50, 
    paddingHorizontal: 20,
    alignContent:'center'
  },
  title: { 
    fontSize: 24, 
    fontWeight: 'bold', 
    marginBottom: 20, 
    color: Colors.text,
    alignSelf:'center' 
  },
  listContent: {
    paddingBottom: 100, 
  },
  emptyText: {
    color: Colors.textSecondary,
    textAlign: 'center',
    marginTop: 30,
  },
});
