import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Dimensions,
  Image,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
  StyleSheet,
  Platform
} from 'react-native';
import { ActionButtons } from '../../components/home/actionButtons';
import { CameraModal } from '../../components/home/cameraModal';
import { VideoCard } from '../../components/home/videoCard';
import { AnalysisResultModal } from '../../components/home/AnalysisResultModal';
import { DarkTheme, LightTheme } from '../../constants/theme'; 
import { useVideoManager } from '../../hooks/useVideoManager';
import { useUserStore, AnalysisResult } from '../../store/useUserStore';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CONTAINER_PADDING = 25;
const VIDEO_CARD_WIDTH = SCREEN_WIDTH - (CONTAINER_PADDING * 2);

export default function HomeScreen() {
  const router = useRouter();
  const { t } = useTranslation(); 
  const { userName, userImage, theme } = useUserStore(); 

  const Colors = theme === 'dark' ? DarkTheme : LightTheme;
  const { videos, addVideo, pickVideoFile, removeVideo } = useVideoManager(2);
  const [isCameraOpen, setCameraOpen] = useState(false);
  
  const [resultVisible, setResultVisible] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [hasResults, setHasResults] = useState(false);
  const [ignoreHistory, setIgnoreHistory] = useState(false);
  const handleResetResults = () => {
    setHasResults(false);
    setAnalysisData(null);
    setResultVisible(false);
    setIgnoreHistory(true);
    useUserStore.getState().setPendingAnalysisResult(null);
  };

  const pendingResult = useUserStore(state => state.pendingAnalysisResult);
  const setPendingResult = useUserStore(state => state.setPendingAnalysisResult);
  const analysisHistory = useUserStore(state => state.analysisHistory);

  useEffect(() => {
    if (pendingResult) {
      setAnalysisData(pendingResult);
      setResultVisible(true);
      setHasResults(true);
      setIgnoreHistory(false);
      setPendingResult(null);
      return;
    }

    if (!ignoreHistory && !hasResults && analysisHistory.length > 0) {
      setHasResults(true);
    }
  }, [pendingResult, analysisHistory.length, hasResults, ignoreHistory, setPendingResult]);

  const handleStartAnalysis = () => {
    if (hasResults) {
      if (!analysisData && analysisHistory.length > 0) {
        setAnalysisData(analysisHistory[0]);
      }
      setResultVisible(true);
    } else {
      if (videos.length < 2) return;
      setIgnoreHistory(false);
      const videoData = {
        left2right: videos[0],
        right2left: videos[1]
      };
      router.push({
        pathname: '/survey',
        params: { videoUris: JSON.stringify(videoData) }
      });
    }
  };

  const dynamicStyles = createStyles(Colors);
  const isAnalysisDisabled = videos.length < 2 && !hasResults;

  return (
    <View style={dynamicStyles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 100, flexGrow: 1 }}>
        
        <View style={dynamicStyles.header}>
          <View style={dynamicStyles.profileSection}>
            <TouchableOpacity onPress={() => router.push('/profile')}>
              <View style={dynamicStyles.avatarCircle}>
                {userImage ? (
                  <Image source={{ uri: userImage }} style={dynamicStyles.avatarImage} />
                ) : (
                  <View style={dynamicStyles.placeholderAvatar}>
                    <MaterialCommunityIcons name="account-circle" size={48} color={Colors.placeholder} />
                  </View>
                )}
              </View>
            </TouchableOpacity>
            <View>
              <Text style={dynamicStyles.welcomeText}>{t('home.welcome')}</Text>
              <Text style={dynamicStyles.userName}>{userName || 'Yasser'}</Text>
            </View>
          </View>
          <TouchableOpacity onPress={() => router.push('/settings')}>
            <Ionicons name="settings-outline" size={24} color={Colors.primary} />
          </TouchableOpacity>
        </View>

        <ActionButtons 
          colors={Colors}
          onCameraPress={() => setCameraOpen(true)} 
          onUploadPress={pickVideoFile} 
          count={videos.length}
          cameraLabel={t('home.open_camera', { count: videos.length })}
          uploadLabel={t('home.select_file', { count: videos.length })}
          orLabel={t('home.or')}
        />

        <View style={dynamicStyles.displayArea}>
          {videos.length > 0 ? (
            <ScrollView 
              horizontal 
              pagingEnabled 
              showsHorizontalScrollIndicator={false}
              snapToInterval={VIDEO_CARD_WIDTH}
              decelerationRate="fast"
            >
              {videos.map((uri, index) => (
                <VideoCard 
                  key={index} 
                  uri={uri} 
                  index={index} 
                  title={index === 0 ? t('home.left_to_right') : t('home.right_to_left')}
                  onRemove={removeVideo} 
                  width={VIDEO_CARD_WIDTH} 
                />
              ))}
              {videos.length === 1 && (
                <View style={[dynamicStyles.videoCardPlaceholder, { width: VIDEO_CARD_WIDTH }]}>
                   <MaterialCommunityIcons name="video-plus-outline" size={50} color={Colors.primary} style={{opacity: 0.2}} />
                   <Text style={dynamicStyles.emptyCardText}>{t('home.waiting_second_video')}</Text>
                </View>
              )}
            </ScrollView>
          ) : (
            <View style={dynamicStyles.emptyContent}>
              <MaterialCommunityIcons name="brain" size={100} color={Colors.primary} style={{opacity: 0.3}} />
              <Text style={dynamicStyles.emptyTitle}>{t('home.add_samples')}</Text>
            </View>
          )}
        </View>

        <View style={dynamicStyles.analysisSection}>
            <TouchableOpacity 
              style={[dynamicStyles.brainIconContainer, isAnalysisDisabled && dynamicStyles.disabledBrain]} 
              onPress={handleStartAnalysis}
              activeOpacity={0.7}
              disabled={isAnalysisDisabled}
            >
              <MaterialCommunityIcons 
                name={hasResults ? "file-chart" : "brain"} 
                size={42} 
                color={isAnalysisDisabled ? Colors.placeholder : Colors.primary} 
              />
            </TouchableOpacity>
            <Text style={[dynamicStyles.analysisLabel, isAnalysisDisabled && { color: Colors.placeholder }]}>
              {hasResults ? t('home.view_results') : t('home.start_analysis')}
            </Text>
        </View>
      </ScrollView>

      <CameraModal 
        visible={isCameraOpen} 
        onClose={() => setCameraOpen(false)} 
        onSave={addVideo} 
        videoCount={videos.length}
      />

      <AnalysisResultModal 
        visible={resultVisible} 
        onClose={() => setResultVisible(false)} 
        data={analysisData}
        colors={Colors}
        onReset={handleResetResults}
      />
    </View>
  );
}

const createStyles = (Colors: any) => StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: Colors.background, 
    paddingHorizontal: 25, 
    paddingTop: Platform.OS === 'ios' ? 60 : 40, 
  },
  header: { 
    flexDirection: 'row', 
    justifyContent: 'space-between', 
    alignItems: 'center', 
    marginBottom: 30 
  },
  profileSection: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    gap: 15 
  },
  avatarCircle: { 
    width: 54, 
    height: 54, 
    borderRadius: 27, 
    backgroundColor: Colors.bgLight, 
    borderWidth: 1.5, 
    borderColor: Colors.primary, 
    overflow: 'hidden',
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarImage: {
    width: '100%',
    height: '100%',
  },
  placeholderAvatar: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  welcomeText: { 
    fontSize: 13, 
    color: Colors.primary, 
    fontWeight: '500' 
  },
  userName: { 
    fontSize: 20, 
    fontWeight: 'bold', 
    color: Colors.text 
  },
  displayArea: { 
    height: 380, 
    backgroundColor: Colors.inputBg, 
    borderRadius: 30, 
    overflow: 'hidden' 
  },
  emptyContent: { 
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center' 
  },
  emptyTitle: { 
    color: Colors.primary, 
    marginTop: 10, 
    fontWeight: '500' 
  },
  videoCardPlaceholder: { 
    height: '100%',
    justifyContent: 'center', 
    alignItems: 'center', 
    backgroundColor: Colors.background, 
    opacity: 0.8
  },
  emptyCardText: { 
    color: Colors.primary, 
    fontSize: 14, 
    fontWeight: '500', 
    opacity: 0.5 
  },
  analysisSection: { 
    marginTop: 30, 
    alignItems: 'center', 
    gap: 10 
  },
  brainIconContainer: { 
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
    shadowRadius: 4 
  },
  disabledBrain: { 
    backgroundColor: Colors.inputBg,  
    borderColor: Colors.placeholder,
    elevation: 0, 
    shadowOpacity: 0 
  },
  analysisLabel: { 
    fontSize: 16, 
    color: Colors.primary, 
    fontWeight: 'bold' 
  },
});
