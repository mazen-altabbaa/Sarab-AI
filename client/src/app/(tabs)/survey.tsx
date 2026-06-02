import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { Audio } from 'expo-av'; 
import * as FileSystem from 'expo-file-system/legacy';
import * as MediaLibrary from 'expo-media-library';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Alert,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  TouchableWithoutFeedback,
  View
} from 'react-native';

import { FormInput } from '../../components/survey/formInput';
import { useThemeColors } from '../../hooks/useThemeColors';
import { useSurveyForm } from '../../hooks/useSurveyForm';
import { useVideoManager } from '../../hooks/useVideoManager';
import { surveyService } from '../../services/surveyService';
import { useUserStore, AnalysisResult } from '../../store/useUserStore';

async function saveVideoToCache(base64: string, filename: string): Promise<string> {
  try {
    const cleaned = base64.replace(/\s+/g, '');
    const dir = FileSystem.cacheDirectory ?? 'file:///tmp/';
    const path = dir + filename;
    await FileSystem.writeAsStringAsync(path, cleaned, { encoding: 'base64' as any });
    return path;
  } catch (e) {
    console.warn('saveVideoToCache error:', e);
    return '';
  }
}

export default function SurveyScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const params = useLocalSearchParams();
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  const { form, updateField, isFormValid } = useSurveyForm();
  const { videos, addVideo, removeVideo } = useVideoManager(2);

  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    if (params.videoUris) {
      try {
        const urisObj = JSON.parse(params.videoUris as string);
        if (urisObj.left2right) addVideo(urisObj.left2right);
        if (urisObj.right2left) addVideo(urisObj.right2left);
      } catch (e) {
        console.error("Error parsing videos:", e);
      }
    }
  }, [params.videoUris]);

  async function startRecording() {
    if (recording || isRecording) {
      console.warn('Recording already prepared or in progress.');
      return;
    }

    try {
      setIsRecording(true);
      const permission = await Audio.requestPermissionsAsync();
      if (permission.status !== 'granted') {
        setIsRecording(false);
        return;
      }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording: newRecording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      setRecording(newRecording);
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      setIsRecording(false);
    }
  }

  async function stopRecording() {
    if (!recording) return;
    setIsRecording(false);
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      if (uri) {
        Alert.alert(
          "تم التسجيل",
          "هل تريد تحليل الصوت لتعبئة البيانات؟",
          [
            { text: "إعادة", style: "destructive" },
            { 
              text: "تحليل", 
              onPress: async () => {
                try {
                  setLoading(true);
                  const result = await surveyService.uploadVoiceRecording(uri);
                  if (result) {
                    if (result.EyeSide || result.eyeside) updateField('EyeSide', result.EyeSide || result.eyeside);
                    if (result.Gender || result.gender) updateField('Gender', result.Gender || result.gender);
                    if (result.Age || result.age) updateField('Age', String(result.Age || result.age));
                    if (result.City || result.city) updateField('City', result.City || result.city);
                    if (result.Status || result.status) updateField('Status', result.Status || result.status);
                    if (result.Profession || result.profession) updateField('Profession', result.Profession || result.profession);
                    if (result.Notes || result.notes) updateField('Notes', result.Notes || result.notes);
                    Alert.alert("نجاح", "تمت تعبئة البيانات من التسجيل الصوتي.");
                  }
                } catch (err) {
                  Alert.alert("خطأ", "فشل تحليل الملف الصوتي.");
                } finally {
                  setLoading(false);
                }
              } 
            }
          ]
        );
      }
    } catch (err) { console.error(err); }
  }

  const handleUpload = async () => {
    if (videos.length < 2) {
      Alert.alert("بيانات ناقصة", "يجب توفر فيديوهين لإتمام عملية التحليل.");
      return;
    }
    if (!isFormValid()) {
      Alert.alert("تنبيه", "يرجى تعبئة جميع الحقول المطلوبة.");
      return;
    }

    setLoading(true); 
    try {
      const response = await surveyService.submitSurvey({
        EyeSide: form.EyeSide,
        Gender: form.Gender,
        Age: form.Age,
        City: form.City,
        Status: form.Status,
        Profession: form.Profession,
        Notes: form.Notes,
      }, videos);

      const timestamp = new Date().toISOString();
      const sampleId = response.sampleId;

      const lrRaw = response.results?.trackingVideos?.left2right ?? '';
      console.log('VIDEO DEBUG - length:', lrRaw.length, '| start:', lrRaw.substring(0, 50));

      let lrUri = response.results?.trackingVideos?.left2right ?? null;
      let rlUri = response.results?.trackingVideos?.right2left ?? null;

      if (lrUri && !lrUri.startsWith('file://')) {
        lrUri = await saveVideoToCache(lrUri, `tracked_lr_${sampleId}.mp4`) || null;
      }
      if (rlUri && !rlUri.startsWith('file://')) {
        rlUri = await saveVideoToCache(rlUri, `tracked_rl_${sampleId}.mp4`) || null;
      }

      try {
        const { status } = await MediaLibrary.requestPermissionsAsync();
        if (status === 'granted') {
          if (lrUri) {
            await MediaLibrary.saveToLibraryAsync(lrUri);
            console.log('✅ LR video saved to gallery:', lrUri);
          }
          if (rlUri) {
            await MediaLibrary.saveToLibraryAsync(rlUri);
            console.log('✅ RL video saved to gallery:', rlUri);
          }
        }
      } catch (mediaErr) {
        console.warn('MediaLibrary save error:', mediaErr);
      }

      const finalResult: AnalysisResult = {
        ...response,
        timestamp,
        results: {
          ...response.results,
          trackingVideos: lrUri || rlUri
            ? { left2right: lrUri, right2left: rlUri }
            : null,
        },
      };

      useUserStore.getState().setPendingAnalysisResult(finalResult);
      useUserStore.getState().addToHistory(finalResult);

      router.push('/');

    } catch (error: any) {
      Alert.alert("خطأ في الرفع", "تعذر إرسال البيانات للسيرفر.");
    } finally {
      setLoading(false); 
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
          <Text style={styles.title}>{t('survey.title')}</Text>

          <TouchableOpacity
            style={[styles.uploadMainBtn, (videos.length < 2 || !isFormValid() || loading) && styles.disabledBtn]}
            onPress={handleUpload}
            disabled={loading}
          >
            <Text style={styles.uploadMainText}>
              {loading ? t('survey.uploading') : t('survey.upload_samples')}
            </Text>
          </TouchableOpacity>

          <View style={styles.formContainer}>
            <Text style={styles.sectionTitle}>{t('survey.samples', { count: videos.length })}</Text>

            {videos.map((uri, index) => (
              <View key={index} style={styles.fileCard}>
                <View style={styles.fileInfo}>
                  <MaterialCommunityIcons name="video-check" size={24} color={Colors.primary} />
                  <Text style={styles.fileName} numberOfLines={1}>
                    {index === 0 ? "left2right.mp4" : "right2left.mp4"}
                  </Text>
                </View>
                <TouchableOpacity onPress={() => removeVideo(index)}>
                  <Ionicons name="trash-outline" size={20} color="#ff4d4d" />
                </TouchableOpacity>
              </View>
            ))}

            <View style={styles.divider} />

            <FormInput placeholder={t('survey.eye_side')} value={form.EyeSide} onChangeText={(v: string) => updateField('EyeSide', v)} />

            <View style={styles.row}>
              <FormInput style={{ flex: 1 }} placeholder={t('survey.gender')} value={form.Gender} onChangeText={(v: string) => updateField('Gender', v)} />
              <FormInput style={{ flex: 1 }} placeholder={t('survey.age')} value={form.Age} onChangeText={(v: string) => updateField('Age', v)} keyboardType="numeric" />
            </View>

            <FormInput placeholder={t('survey.city')} value={form.City} onChangeText={(v: string) => updateField('City', v)} />
            <FormInput placeholder={t('survey.state')} value={form.Status} onChangeText={(v: string) => updateField('Status', v)} />
            <FormInput placeholder={t('survey.profession')} value={form.Profession} onChangeText={(v: string) => updateField('Profession', v)} />

            <FormInput
              style={styles.textArea}
              placeholder={t('survey.notes')}
              value={form.Notes}
              onChangeText={(v: string) => updateField('Notes', v)}
              multiline
            />
          </View>

          <View style={styles.audioSection}>
            <TouchableOpacity
              onPressIn={startRecording}
              onPressOut={stopRecording}
              style={[
                styles.micButton,
                isRecording && styles.micActive,
                loading && styles.disabledMic,
              ]}
              disabled={loading}
            >
              <Ionicons name={isRecording ? 'mic' : 'mic-outline'} size={42} color={isRecording ? '#fff' : Colors.primary} />
            </TouchableOpacity>
            <Text style={[styles.micText, loading && { color: Colors.placeholder }]}>
              {isRecording ? t('survey.recording') : t('survey.hold_or_press')}
            </Text>
          </View>

          <View style={{ height: 40 }} />
        </ScrollView>
      </TouchableWithoutFeedback>
    </KeyboardAvoidingView>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  scrollContent: { paddingHorizontal: 25, paddingTop: 50, flexGrow: 1, paddingBottom: 50 },
  title: { fontSize: 26, fontWeight: 'bold', marginBottom: 20, color: Colors.text, textAlign: 'center' },
  sectionTitle: { fontSize: 16, fontWeight: '600', color: Colors.textSecondary, marginBottom: 10 },
  uploadMainBtn: { backgroundColor: Colors.primary, paddingVertical: 18, borderRadius: 35, alignItems: 'center', elevation: 4 },
  disabledBtn: { opacity: 0.5 },
  uploadMainText: { color: 'white', fontSize: 20, fontWeight: 'bold' },
  formContainer: { marginTop: 25, gap: 15 },
  fileCard: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: Colors.card, padding: 15, borderRadius: 15, borderWidth: 1, borderColor: Colors.borderPurple },
  fileInfo: { flexDirection: 'row', alignItems: 'center', gap: 10, flex: 1 },
  fileName: { fontSize: 14, color: Colors.text, fontWeight: '500', flex: 1 },
  row: { flexDirection: 'row', gap: 10 },
  divider: { height: 1, backgroundColor: Colors.borderPurple, marginVertical: 10 },
  textArea: { height: 120, textAlignVertical: 'top' },
  audioSection: { alignItems: 'center', marginVertical: 30 },
  micButton: {
    width: 85,
    height: 85,
    borderRadius: 45,
    backgroundColor: Colors.bgLight,
    borderWidth: 1.5,
    borderColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 2,
    shadowColor: Colors.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
  },
  micActive: {
    backgroundColor: '#ff4d4d',
    transform: [{ scale: 1.08 }],
    borderColor: '#ff4d4d',
  },
  disabledMic: {
    backgroundColor: Colors.inputBg,
    borderColor: Colors.placeholder,
    elevation: 0,
    shadowOpacity: 0,
  },
  micText: { marginTop: 10, color: Colors.textSecondary, fontSize: 14 },
});
