import { Ionicons } from '@expo/vector-icons';
import { ResizeMode, Video } from 'expo-av';
import { CameraView } from 'expo-camera';
import React, { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Colors } from '../../constants/colors';

interface CameraModalProps {
  visible: boolean;
  onClose: () => void;
  onSave: (uri: string) => void;
  videoCount: number;
}

export const CameraModal = ({ visible, onClose, onSave, videoCount }: CameraModalProps) => {
  const { t } = useTranslation();
  const cameraRef = useRef<any>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [tempVideoUri, setTempVideoUri] = useState<string | null>(null);

  const handleRecord = async () => {
    if (cameraRef.current) {
      if (isRecording) {
        cameraRef.current.stopRecording();
        setIsRecording(false);
      } else {
        setIsRecording(true);
        try {
          const video = await cameraRef.current.recordAsync();
          setTempVideoUri(video.uri);
        } catch (error) {
          setIsRecording(false);
          console.error("Recording error:", error);
        }
      }
    }
  };

  const handleConfirmSave = () => {
    if (tempVideoUri) {
      onSave(tempVideoUri);
      setTempVideoUri(null);
      onClose();
    }
  };

  return (
    <Modal visible={visible} animationType="slide">
      <View style={styles.cameraScreen}>
        {!tempVideoUri ? (
          <CameraView ref={cameraRef} style={styles.flex1} facing="back" mode="video">
            <View style={styles.cameraOverlay}>
              <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
                <Ionicons name="close" size={30} color="white" />
              </TouchableOpacity>

              <View style={styles.bottomControls}>
                <TouchableOpacity 
                  onPress={handleRecord} 
                  style={[styles.recordBtn, isRecording && styles.activeRecord]}
                >
                  <View style={isRecording ? styles.stopIcon : styles.startIcon} />
                </TouchableOpacity>
              </View>
            </View>
          </CameraView>
        ) : (
          <View style={styles.previewContainer}>
            <Text style={styles.previewTitle}>{t('camera.save_sample')}</Text>
            <Video 
              source={{ uri: tempVideoUri }} 
              style={styles.previewVideo} 
              useNativeControls 
              resizeMode={ResizeMode.CONTAIN}
              shouldPlay 
            />
            <View style={styles.previewButtons}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setTempVideoUri(null)}>
                <Text style={styles.btnText}>{t('camera.retake')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.confirmBtn} onPress={handleConfirmSave}>
                <Text style={styles.btnText}>{t('camera.confirm_video', { count: videoCount + 1 })}</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  cameraScreen: { flex: 1, backgroundColor: '#000' },
  flex1: { flex: 1 },
  cameraOverlay: { flex: 1, justifyContent: 'space-between', padding: 40 },
  closeBtn: { alignSelf: 'flex-start', backgroundColor: 'rgba(0,0,0,0.5)', padding: 8, borderRadius: 25 },
  bottomControls: { alignItems: 'center', marginBottom: 20 },
  recordBtn: { 
    width: 80, height: 80, borderRadius: 40, borderWidth: 5, 
    borderColor: 'white', justifyContent: 'center', alignItems: 'center' 
  },
  activeRecord: { borderColor: '#ff4c4c' },
  startIcon: { width: 60, height: 60, borderRadius: 30, backgroundColor: '#ff4c4c' },
  stopIcon: { width: 30, height: 30, borderRadius: 5, backgroundColor: '#ff4c4c' },
  previewContainer: { flex: 1, backgroundColor: '#121212', justifyContent: 'center', padding: 20 },
  previewTitle: { color: 'white', textAlign: 'center', fontSize: 18, marginBottom: 20, fontWeight: '600' },
  previewVideo: { width: '100%', height: '60%', borderRadius: 25 },
  previewButtons: { flexDirection: 'row', gap: 15, marginTop: 25 },
  cancelBtn: { flex: 1, backgroundColor: '#333', padding: 18, borderRadius: 15, alignItems: 'center' },
  confirmBtn: { flex: 1, backgroundColor: Colors.primary, padding: 18, borderRadius: 15, alignItems: 'center' },
  btnText: { color: 'white', fontWeight: 'bold', fontSize: 16 }
});
