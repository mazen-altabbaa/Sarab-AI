import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import React from 'react';
import { Image, StyleSheet, TouchableOpacity, View } from 'react-native';
import { useThemeColors } from '../../hooks/useThemeColors';

interface ProfilePictureProps {
  imageUri: string | null;
  size?: number;
  showControls?: boolean;
  onPick?: () => void;
  onRemove?: () => void;
  onView: () => void;
}

export const ProfilePicture = ({ imageUri, size = 125, showControls = true, onPick, onRemove, onView }: ProfilePictureProps) => {
  const Colors = useThemeColors();
  const styles = createStyles(Colors, size);

  return (
    <View style={styles.outerContainer}> 
      <View style={styles.imageWrapper}>
        
        <TouchableOpacity activeOpacity={0.8} onPress={onView}>
          <View style={styles.circle}>
            {imageUri ? (
              <Image source={{ uri: imageUri }} style={styles.image} />
            ) : (
              <View style={styles.absoluteCenter}>
                <MaterialCommunityIcons 
                  name="account-circle" 
                  size={165} 
                  color={Colors.placeholder} 
                  style={styles.defaultIcon}
                />
              </View>
            )}
          </View>
        </TouchableOpacity>
        
        {showControls && onPick && (
          <TouchableOpacity style={styles.editBadge} onPress={onPick}>
            <Ionicons name="pencil" size={16} color="#fff" />
          </TouchableOpacity>
        )}

        {showControls && imageUri && onRemove && (
          <TouchableOpacity style={styles.deleteBadge} onPress={onRemove}>
            <Ionicons name="trash-outline" size={16} color="#fff" />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const createStyles = (Colors: any, size: number) => StyleSheet.create({
  outerContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    marginVertical: 10,
  },
  imageWrapper: { 
    position: 'relative', 
    width: size,
    height: size,
  },
  circle: { 
    width: size,
    height: size,
    borderRadius: size / 2,
    borderWidth: 2, 
    borderColor: Colors.bgLight,
    overflow: 'hidden', 
    backgroundColor: Colors.bgLight,
  },
  image: { width: '100%', height: '100%', resizeMode: 'cover' },
  absoluteCenter: {
    position: 'absolute',
    top: -15, left: -22, right: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  defaultIcon: { includeFontPadding: false, textAlignVertical: 'center' },
  
  editBadge: {
    position: 'absolute',
    right: 5, 
    bottom: 0, 
    backgroundColor: Colors.primary,
    width: 34,
    height: 34,
    borderRadius: 17,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#fff',
    zIndex: 10,
  },
  
  deleteBadge: {
    position: 'absolute',
    left: 5, 
    bottom: 0,
    backgroundColor: '#ff4d4d',
    width: 34,
    height: 34,
    borderRadius: 17,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#fff',
    zIndex: 10,
  },
});
