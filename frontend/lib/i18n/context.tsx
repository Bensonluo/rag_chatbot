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

/** Pre-paint locale bootstrap, inlined into the landing page. The static
 * export ships both locale trees; this runs during parse — before first
 * paint — to pin <html data-locale> so CSS shows the visitor's tree from
 * the very first paint: no post-hydration language swap, no layout shift
 * (the single-tree design cost CLS 0.207 on non-zh first visits). */
export const LOCALE_BOOTSTRAP = [
  '(function(){try{',
  `var s=localStorage.getItem('${STORAGE_KEY}');`,
  "var l=(s==='en'||s==='zh')?s:",
  "((navigator.languages&&navigator.languages[0])||navigator.language||'')",
  ".toLowerCase().indexOf('zh')===0?'zh':'en';",
  'var d=document.documentElement;',
  "d.dataset.locale=l;d.lang=l==='zh'?'zh-CN':'en';",
  '}catch(e){}})();',
].join('');

export function LanguageProvider({ children }: { children: ReactNode }) {
  const locale = useSyncExternalStore(subscribeLocale, readLocale, serverLocale);

  useEffect(() => {
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
    // Keep the CSS locale gate (both-trees landing) in lockstep with the
    // context — covers toggles even when browser storage is blocked.
    document.documentElement.dataset.locale = locale;
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
