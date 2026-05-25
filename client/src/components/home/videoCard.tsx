import { Ionicons } from '@expo/vector-icons';
import { ResizeMode, Video } from 'expo-av';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Colors } from '../../constants/theme';

interface VideoCardProps {
  uri: string;
  index: number;
  title: string; 
  onRemove: (index: number) => void;
  width: number;
}

export const VideoCard = ({ uri, index, title, onRemove, width }: VideoCardProps) => (
  <View style={[styles.card, { width: width }]}>
    <View style={styles.videoWrapper}>
        <Video 
          source={{ uri }} 
          style={styles.video} 
          useNativeControls 
          resizeMode={ResizeMode.COVER} 
          isLooping 
          shouldPlay
        />

        <TouchableOpacity style={styles.deleteBtn} onPress={() => onRemove(index)}>
          <Ionicons name="trash-outline" size={20} color="white" />
        </TouchableOpacity>

        <View style={styles.badge}>
          <Text style={styles.badgeText}>{title}</Text>
        </View>
    </View>
  </View>
);

const styles = StyleSheet.create({
  card: { 
    height: '100%', 
    justifyContent: 'center', 
    alignItems: 'center',
  },
  videoWrapper: {
    width: '90%', 
    height: '100%',
    borderRadius: 20, 
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  video: { 
    width: '100%', 
    height: '100%',
  },
  deleteBtn: { 
    position: 'absolute', 
    top: 20, 
    right: 20, 
    backgroundColor: 'rgba(255, 76, 76, 0.8)', 
    padding: 8, 
    borderRadius: 20,
    zIndex: 10,
  },
  badge: { 
    position: 'absolute', 
    top: 20, 
    left: 20, 
    backgroundColor: Colors.primary, 
    paddingHorizontal: 15, 
    paddingVertical: 6, 
    borderRadius: 12,
    zIndex: 10,
  },
  badgeText: { 
    color: 'white', 
    fontSize: 12, 
    fontWeight: 'bold' 
  }
});