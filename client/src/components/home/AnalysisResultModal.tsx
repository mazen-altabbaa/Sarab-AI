import React, { useState, useEffect } from 'react';
import {
  Modal, View, Text, Image, ScrollView,
  StyleSheet, TouchableOpacity, Dimensions,
  ActivityIndicator, BackHandler
} from 'react-native';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useVideoPlayer, VideoView } from 'expo-video';
import * as FileSystem from 'expo-file-system/legacy';
import { Colors } from '../../constants/colors';
import { useThemeColors } from '../../hooks/useThemeColors';
import type { AnalysisResult } from '../../store/useUserStore';

const { width, height } = Dimensions.get('window');

interface Props {
  visible: boolean;
  onClose: () => void;
  data: AnalysisResult | null;
  colors?: any;
  onReset?: () => void;
}

// ─── تحديد إذا كانت القيمة base64 أم file:// URI ─────────────────
function isFileUri(value: string): boolean {
  return value.startsWith('file://') || value.startsWith('content://') || value.startsWith('/');
}

// ─── حفظ base64 كملف مؤقت (fallback إذا وصل base64) ─────────────
async function saveBase64ToTemp(base64: string, filename: string): Promise<string | null> {
  try {
    const cleaned = base64.replace(/\s+/g, '');
    const dir = FileSystem.cacheDirectory ?? 'file:///tmp/';
    const path = dir + filename;
    await FileSystem.writeAsStringAsync(path, cleaned, { encoding: 'base64' as any });
    return path;
  } catch (e) {
    console.warn('saveBase64ToTemp error:', e);
    return null;
  }
}

// ─── مكوّن VideoPlayer (hook دائماً يُستدعى) ─────────────────────
const VideoPlayer = ({ uri }: { uri: string }) => {
  const player = useVideoPlayer(uri, (p) => {
    p.loop = true;
    p.muted = true;
    p.play();
  });

  return (
    <VideoView
      player={player}
      style={vidStyles.video}
      allowsPictureInPicture={false}
      contentFit="contain"
      nativeControls
    />
  );
};

// ─── مكوّن TrackingVideo: يدعم file:// URI و base64 ──────────────
const TrackingVideo = ({ value, label, sampleId, side }: {
  value: string;
  label: string;
  sampleId: number;
  side: 'lr' | 'rl';
}) => {
  const [localUri, setLocalUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);
    setLocalUri(null);

    const prepare = async () => {
      // إذا كان file:// URI جاهز — استخدمه مباشرة
      if (isFileUri(value)) {
        if (!cancelled) { setLocalUri(value); setLoading(false); }
        return;
      }
      // وإلا احفظ base64 كملف مؤقت
      const path = await saveBase64ToTemp(value, `tracked_${side}_${sampleId}.mp4`);
      if (cancelled) return;
      if (path) setLocalUri(path);
      else setError(true);
      setLoading(false);
    };

    prepare();
    return () => { cancelled = true; };
  }, [value]);

  return (
    <View style={vidStyles.wrapper}>
      <Text style={vidStyles.label}>{label}</Text>
      {loading && (
        <View style={vidStyles.placeholder}>
          <ActivityIndicator color={Colors.primary} size="large" />
          <Text style={vidStyles.loadingText}>جاري تحميل الفيديو...</Text>
        </View>
      )}
      {error && (
        <View style={vidStyles.placeholder}>
          <MaterialCommunityIcons name="video-off-outline" size={40} color="#ccc" />
          <Text style={vidStyles.errorText}>تعذّر تحميل الفيديو</Text>
        </View>
      )}
      {localUri && !loading && !error && <VideoPlayer uri={localUri} />}
    </View>
  );
};

const vidStyles = StyleSheet.create({
  wrapper: { marginBottom: 16 },
  label: { fontSize: 14, fontWeight: '600', color: '#444', marginBottom: 6 },
  video: { width: '100%', height: 220, borderRadius: 12, backgroundColor: '#000', overflow: 'hidden' },
  placeholder: {
    width: '100%', height: 220, borderRadius: 12,
    backgroundColor: '#1a1a1a', justifyContent: 'center', alignItems: 'center', gap: 10,
  },
  loadingText: { color: '#aaa', fontSize: 13 },
  errorText: { color: '#888', fontSize: 13 },
});

// ─── مكوّن HeatmapImage ───────────────────────────────────────────
const HeatmapImage = ({ base64, label }: { base64: string; label: string }) => {
  const [previewVisible, setPreviewVisible] = useState(false);

  const uri = (() => {
    const cleaned = base64.replace(/\s+/g, '');
    if (cleaned.startsWith('data:image')) return cleaned;
    if (cleaned.startsWith('/9j/') || cleaned.startsWith('/9k/')) return `data:image/jpeg;base64,${cleaned}`;
    return `data:image/png;base64,${cleaned}`;
  })();

  useEffect(() => {
    if (!previewVisible) return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      setPreviewVisible(false);
      return true;
    });
    return () => sub.remove();
  }, [previewVisible]);

  return (
    <>
      <TouchableOpacity style={imgStyles.wrapper} onPress={() => setPreviewVisible(true)} activeOpacity={0.85}>
        <Image source={{ uri }} style={imgStyles.thumb} resizeMode="contain" />
        <Text style={imgStyles.label}>{label}</Text>
        <View style={imgStyles.zoomHint}>
          <Ionicons name="expand-outline" size={14} color="#fff" />
        </View>
      </TouchableOpacity>

      <Modal visible={previewVisible} transparent animationType="fade" statusBarTranslucent>
        <View style={imgStyles.overlay}>
          <TouchableOpacity style={imgStyles.closeBtn} onPress={() => setPreviewVisible(false)}>
            <Ionicons name="close-circle" size={38} color="#fff" />
          </TouchableOpacity>
          <ScrollView maximumZoomScale={4} minimumZoomScale={1} centerContent contentContainerStyle={imgStyles.previewContent}>
            <Image source={{ uri }} style={{ width, height: height * 0.75 }} resizeMode="contain" />
          </ScrollView>
          <Text style={imgStyles.previewLabel}>{label}</Text>
        </View>
      </Modal>
    </>
  );
};

const imgStyles = StyleSheet.create({
  wrapper: { width: (width - 60) / 2, marginBottom: 15, alignItems: 'center' },
  thumb: { width: '100%', height: 150, borderRadius: 10, backgroundColor: '#e8e8e8' },
  label: { marginTop: 5, fontSize: 12, color: '#666', textAlign: 'center' },
  zoomHint: { position: 'absolute', top: 8, right: 8, backgroundColor: 'rgba(0,0,0,0.45)', borderRadius: 12, padding: 4 },
  overlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.96)', justifyContent: 'center', alignItems: 'center' },
  closeBtn: { position: 'absolute', top: 50, right: 20, zIndex: 10 },
  previewContent: { flexGrow: 1, justifyContent: 'center', alignItems: 'center' },
  previewLabel: { position: 'absolute', bottom: 40, color: '#ccc', fontSize: 13, textAlign: 'center' },
});

// ─── المكوّن الرئيسي ──────────────────────────────────────────────
export const AnalysisResultModal = ({ visible, onClose, data, onReset }: Props) => {
  const ThemeColors = useThemeColors();
  const styles = createStyles(ThemeColors);

  useEffect(() => {
    if (!visible) return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      onClose();
      return true;
    });
    return () => sub.remove();
  }, [visible, onClose]);

  if (!data) return null;

  const maps = data.results?.maps;
  const trackingVideos = data.results?.trackingVideos;

  const mapImages: { label: string; base64: string }[] = [];
  if (maps?.fullMap)    mapImages.push({ label: 'Full Heatmap',       base64: maps.fullMap });
  if (maps?.left2right) mapImages.push({ label: 'Heatmap Left→Right', base64: maps.left2right });
  if (maps?.right2left) mapImages.push({ label: 'Heatmap Right→Left', base64: maps.right2left });

  const hasTracking = !!(trackingVideos?.left2right || trackingVideos?.right2left);
  const hasAnyResult = mapImages.length > 0 || hasTracking;

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={onClose} style={styles.backBtn}>
            <Ionicons name="chevron-back" size={26} color={ThemeColors.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>نتائج التحليل البصري</Text>
          <TouchableOpacity onPress={onClose}>
            <Ionicons name="close-circle" size={28} color="#aaa" />
          </TouchableOpacity>
        </View>

        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
          <View style={styles.idBadge}>
            <MaterialCommunityIcons name="identifier" size={18} color={ThemeColors.primary} />
            <Text style={styles.idText}>Sample ID: {data.sampleId}</Text>
          </View>

          {!hasAnyResult ? (
            <View style={styles.emptyState}>
              <MaterialCommunityIcons name="alert-circle-outline" size={50} color="#ccc" />
              <Text style={styles.emptyText}>لم يتم استلام نتائج من خدمة التحليل</Text>
            </View>
          ) : (
            <>
              {mapImages.length > 0 && (
                <>
                  <Text style={styles.sectionTitle}>خرائط المسح (Heatmaps)</Text>
                  <View style={styles.imagesGrid}>
                    {mapImages.map((item, i) => (
                      <HeatmapImage key={i} base64={item.base64} label={item.label} />
                    ))}
                  </View>
                </>
              )}

              {hasTracking && (
                <>
                  <Text style={styles.sectionTitle}>فيديوهات التتبع (Tracking)</Text>
                  {trackingVideos?.left2right && (
                    <TrackingVideo
                      value={trackingVideos.left2right}
                      label="Tracking Left → Right"
                      sampleId={data.sampleId}
                      side="lr"
                    />
                  )}
                  {trackingVideos?.right2left && (
                    <TrackingVideo
                      value={trackingVideos.right2left}
                      label="Tracking Right → Left"
                      sampleId={data.sampleId}
                      side="rl"
                    />
                  )}
                  <Text style={styles.note}>
                    * فيديوهات التتبع محفوظة على الخادم ضمن العينة رقم {data.sampleId}
                  </Text>
                </>
              )}
            </>
          )}

          <View style={styles.buttonRow}>
            {onReset && (
              <TouchableOpacity style={styles.resetBtn} onPress={onReset}>
                <Text style={styles.resetBtnText}>إعادة التعيين</Text>
              </TouchableOpacity>
            )}
            <TouchableOpacity style={[styles.saveBtn, !onReset && { flex: 1 }]} onPress={onClose}>
              <Text style={styles.saveBtnText}>إغلاق</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </View>
    </Modal>
  );
};

const createStyles = (Colors: any) => StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 16, paddingVertical: 14,
    backgroundColor: Colors.card, borderBottomWidth: 1, borderBottomColor: Colors.borderPurple,
  },
  backBtn: { padding: 4 },
  headerTitle: { fontSize: 17, fontWeight: 'bold', flex: 1, textAlign: 'center', color: Colors.text },
  scrollContent: { padding: 20, paddingBottom: 50 },
  idBadge: {
    flexDirection: 'row', alignItems: 'center', gap: 6,
    backgroundColor: Colors.bgLight, paddingHorizontal: 12, paddingVertical: 8,
    borderRadius: 8, alignSelf: 'flex-start', marginBottom: 20,
  },
  idText: { fontSize: 14, color: Colors.primary, fontWeight: '600' },
  sectionTitle: { fontSize: 16, fontWeight: 'bold', marginVertical: 15, color: Colors.text },
  imagesGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
  note: { fontSize: 11, color: Colors.textSecondary, marginTop: 4, marginBottom: 16, lineHeight: 16 },
  emptyState: { alignItems: 'center', paddingVertical: 40, gap: 10 },
  emptyText: { color: Colors.textSecondary, fontSize: 14 },
  buttonRow: { flexDirection: 'row', gap: 10, marginTop: 30 },
  resetBtn: { flex: 1, backgroundColor: '#ff6b6b', padding: 15, borderRadius: 12, alignItems: 'center' },
  resetBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
  saveBtn: { flex: 1, backgroundColor: Colors.primary, padding: 15, borderRadius: 12, alignItems: 'center' },
  saveBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
});
