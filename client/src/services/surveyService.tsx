import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { Platform } from 'react-native';
import { getApiBaseUrl } from './apiConfig';
 
const BASE_URL = getApiBaseUrl();
 
if (__DEV__) {
  console.log('surveyService BASE_URL:', BASE_URL);
}
 
async function getAuthHeaders() {
  try {
    const token = await AsyncStorage.getItem('sarab_auth_token');
    if (__DEV__) {
      console.log('surveyService: Fetched token status:', token ? 'Token Exists' : 'Token Mofaoood!');
    }
    return token ? { 'Authorization': `Bearer ${token}` } : {};
  } catch (error) {
    console.error('Error fetching auth token from storage:', error);
    return {};
  }
}
 
export const surveyService = {
  submitSurvey: async (formData: any, videos: string[]) => {
    try {
      const data = new FormData();
 
      Object.keys(formData).forEach(key => {
        if (formData[key] !== null && formData[key] !== undefined) {
          data.append(key, formData[key]);
        }
      });
 
      videos.forEach((uri, index) => {
        const customFileName = index === 0 ? 'left2right.mp4' : 'right2left.mp4';
        data.append('Videos', {
          uri: Platform.OS === 'ios' ? uri.replace('file://', '') : uri,
          name: customFileName,
          type: 'video/mp4',
        } as any);
      });
 
      const authHeaders = await getAuthHeaders();
 
      const response = await axios.post(
        `${BASE_URL}/api/Samples/upload/sarab-ai`,
        data,
        {
          headers: {
            ...authHeaders,
            'Content-Type': 'multipart/form-data',
            'Accept': 'application/json',
          },
          timeout: 900000,
        }
      );
 
      return response.data;
    } catch (error: any) {
      console.error('surveyService submitSurvey error', {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
      });
      throw error;
    }
  },
 
  uploadVoiceRecording: async (audioUri: string) => {
    try {
      const data = new FormData();
      const filename = audioUri.split('/').pop() || 'recording.m4a';
 
      data.append('AudioFile', {
        uri: Platform.OS === 'android' ? audioUri : audioUri.replace('file://', ''),
        name: filename,
        type: 'audio/x-m4a',
      } as any);
 
      const authHeaders = await getAuthHeaders();
      const url = `${BASE_URL}/api/ASR`;
 
      const headers: Record<string, string> = {
        'Accept': 'application/json',
      };
 
      if (authHeaders['Authorization']) {
        headers['Authorization'] = authHeaders['Authorization'];
      }
 
      if (__DEV__) {
        console.log('Sending ASR Request to:', url, 'with Headers:', headers);
      }

      const fetchResponse = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: data,
      });
 
      if (!fetchResponse.ok) {
        const text = await fetchResponse.text().catch(() => '');
        const err: any = new Error(`ASR upload failed with status ${fetchResponse.status}`);
        err.status = fetchResponse.status;
        err.body = text;
        
        console.log(`ASR Server Error Details [Status ${fetchResponse.status}]:`, text);
        throw err;
      }
 
      const result = await fetchResponse.json();
      return result;
 
    } catch (error: any) {
      console.error('surveyService uploadVoiceRecording error', {
        baseUrl: BASE_URL,
        url: `${BASE_URL}/api/ASR`,
        message: error.message,
        status: error.status,
        body: error.body,
      });
      throw error;
    }
  }
};