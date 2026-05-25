import { DarkTheme, LightTheme } from '../constants/theme';
import { useUserStore } from '../store/useUserStore';

export const useThemeColors = () => {
  const theme = useUserStore((state) => state.theme);
  return theme === 'dark' ? DarkTheme : LightTheme;
};

