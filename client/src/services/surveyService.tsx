import axios from 'axios';
import { Platform } from 'react-native';
import { getApiBaseUrl } from './apiConfig';

const BASE_URL = getApiBaseUrl();

export const surveyService = {
  // 1. الواجهة الأولى: إرسال بيانات الاستبيان والفيديوهات
  submitSurvey: async (formData: any, videos: string[]) => {
    try {
      // 1. إنشاء كائن FormData لإرسال البيانات كـ Multipart
      const data = new FormData();

      // 2. إضافة بيانات الاستبيان من الكائن formData
      Object.keys(formData).forEach(key => {
        if (formData[key] !== null && formData[key] !== undefined) {
          data.append(key, formData[key]);
        }
      });

      // 3. معالجة ورفع الفيديوهات
      videos.forEach((uri, index) => {
        const customFileName = index === 0 ? 'left2right.mp4' : 'right2left.mp4';
        
        data.append('Videos', {
          uri: Platform.OS === 'ios' ? uri.replace('file://', '') : uri,
          name: customFileName,
          type: 'video/mp4', 
        } as any);
      });

      // 4. إرسال الطلب باستخدام Axios
      const response = await axios.post(
        `${BASE_URL}/api/Samples/upload/sarab-ai`,
        data,
        {
          headers: { 
            'Accept': 'application/json', 
            'Content-Type': 'multipart/form-data' 
          },
          // المهلة الزمنية 15 دقيقة
          timeout: 900000, 
          // منع Axios من تحويل الـ FormData تلقائياً لضمان سلامة الملفات الثنائية
          transformRequest: (data) => data, 
        }
      );

      // 5. إرجاع النتيجة كاملة
      // ملاحظة: الـ response.data سيحتوي الآن على (sampleId, message, results)
      // حيث أن results تحتوي على فيديوهات التتبع والخرائط الحرارية (Base64)
      return response.data;

    } catch (error: any) {
      if (error.code === 'ECONNABORTED') {
        console.error("انتهت مهلة الاتصال، المعالجة تستغرق وقتاً طويلاً.");
      }
      throw error;
    }
  },

  // 2. الواجهة الثانية: إرسال التسجيل الصوتي فقط
  uploadVoiceRecording: async (audioUri: string) => {
    try {
      const data = new FormData();
      
      // الحصول على اسم الملف من الـ URI
      const filename = audioUri.split('/').pop() || 'recording.m4a';
      
      // التغليف المتوافق مع Swagger (المفتاح هو AudioFile)
      data.append('AudioFile', {
        uri: Platform.OS === 'android' ? audioUri : audioUri.replace('file://', ''),
        name: filename,
        type: 'audio/x-m4a', // أو audio/mpeg حسب الصيغة
      } as any);

      const response = await axios.post(`${BASE_URL}/api/ASR`, data, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data; // هذا سيعيد الـ JSON الذي يحتوي على بيانات الحقول
      } catch (error: any) {
        if (error.response) {
          // سيطبع حالة الخطأ (مثلاً 500 أو 404)
          console.log("Status:", error.response.status);
          // سيطبع نص الخطأ الخام حتى لو لم يكن JSON
          console.log("Raw Error Data:", error.response.data);
        } else {
          console.log("Error Message:", error.message);
        }
      throw error;
    }
  }
};