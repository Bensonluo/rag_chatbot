'use client';

import { createContext, useContext, useSyncExternalStore, useEffect, useCallback, type ReactNode } from 'react';

export type Locale = 'zh' | 'en';

interface LanguageContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
}

const LanguageContext = createContext<LanguageContextType>({
  locale: 'zh',
  setLocale: () => {},
});

const STORAGE_KEY = 'portfolio-locale';
const LOCALE_EVENT = 'portfolio-locale-change';
let volatileLocale: Locale | null = null;

function detectSystemLocale(): Locale {
  const language = navigator.languages?.[0] || navigator.language || '';
  return language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
}

function readLocale(): Locale {
  if (volatileLocale) return volatileLocale;
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored === 'en' || stored === 'zh' ? stored : detectSystemLocale();
  } catch {
    return detectSystemLocale();
  }
}

function subscribeLocale(onChange: () => void) {
  window.addEventListener('storage', onChange);
  window.addEventListener(LOCALE_EVENT, onChange);
  return () => {
    window.removeEventListener('storage', onChange);
    window.removeEventListener(LOCALE_EVENT, onChange);
  };
}

const serverLocale = (): Locale => 'zh';

export function LanguageProvider({ children }: { children: ReactNode }) {
  const locale = useSyncExternalStore(subscribeLocale, readLocale, serverLocale);

  useEffect(() => {
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
  }, [locale]);

  const setLocale = useCallback((newLocale: Locale) => {
    try {
      localStorage.setItem(STORAGE_KEY, newLocale);
      volatileLocale = null;
    } catch {
      // Keep the current tab usable when browser storage is blocked.
      volatileLocale = newLocale;
    }
    window.dispatchEvent(new Event(LOCALE_EVENT));
  }, []);

  return (
    <LanguageContext.Provider value={{ locale, setLocale }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  return context;
}
