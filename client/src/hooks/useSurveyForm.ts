import { useState } from 'react';

export const useSurveyForm = () => {
  const [form, setForm] = useState({
    EyeSide: '',
    Gender: '',
    Age: '',
    City: '',
    Status: '',
    Profession: '',
    Notes: '',
  });

  const updateField = (field: string, value: string) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const validateSurveyForm = () => {
    const { EyeSide, Gender, Age, City, Status, Profession } = form;

    if ([EyeSide, Gender, Age, City, Status, Profession].some(value => value.trim() === '')) {
      return 'يرجى تعبئة جميع الحقول المطلوبة';
    }

    const ageNumber = Number(Age.trim());
    if (!Number.isFinite(ageNumber) || ageNumber <= 0) {
      return 'العمر يجب أن يكون أكبر من صفر';
    }

    return null;
  };

  const isFormValid = () => validateSurveyForm() === null;

  return { form, updateField, isFormValid, validateSurveyForm };
};