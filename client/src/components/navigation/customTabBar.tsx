import { Ionicons } from '@expo/vector-icons';
import React from 'react';
import { StyleSheet, TouchableOpacity, View } from 'react-native';
import { useThemeColors } from '../../hooks/useThemeColors';

const TAB_ICONS: any = {
  'index': { active: 'home', inactive: 'home-outline' },
  'survey': { active: 'cloud-upload', inactive: 'cloud-upload-outline' },
  'history': { active: 'time', inactive: 'time-outline' },
};

export const CustomTabBar = ({ state, navigation }: any) => {
  const Colors = useThemeColors();
  const styles = createStyles(Colors);

  return (
    <View style={styles.tabBarContainer}>
      {state.routes.map((route: any, index: number) => {
        const isFocused = state.index === index;
        const currentIcon = TAB_ICONS[route.name] || TAB_ICONS['index'];

        const onPress = () => {
          const event = navigation.emit({
            type: 'tabPress',
            target: route.key,
            canPreventDefault: true,
          });

          if (!isFocused && !event.defaultPrevented) {
            navigation.navigate(route.name);
          }
        };

        return (
          <TouchableOpacity 
            key={route.key} 
            onPress={onPress} 
            style={styles.tabItem}
            activeOpacity={0.7}
          >
            <Ionicons 
              name={isFocused ? currentIcon.active : currentIcon.inactive} 
              size={26} 
              color={isFocused ? Colors.tabBarActive : Colors.tabBarInactive}
            />
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const createStyles = (Colors: any) => StyleSheet.create({
  tabBarContainer: {
    flexDirection: 'row',
    position: 'absolute',
    bottom: 0, 
    left: 0,
    right: 0,
    backgroundColor: Colors.tabBarBackground,
    height: 90, 
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: Colors.background === '#121212' ? 0.35 : 0.1,
    shadowRadius: 10,
    alignItems: 'center',
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    top: -20,
    left: 0,
    right: 0,
  },
});
