export { LanguageProvider, useLanguage, type Locale } from './context';
export { getTranslation, translations } from './translations';

import { useLanguage } from './context';
import { getTranslation } from './translations';
import { useCallback } from 'react';

export function useTranslation() {
  const { locale, setLocale } = useLanguage();

  const t = useCallback((key: string): string => {
    return getTranslation(locale, key);
  }, [locale]);

  return { t, locale, setLocale };
}
