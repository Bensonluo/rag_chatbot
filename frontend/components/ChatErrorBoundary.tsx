'use client';

/** Contains a crash of the live-chat island to the chat panel itself.
 * React unmounts the entire tree on an uncaught render error, so without
 * this boundary a single throwing render inside the chat component would
 * blank the whole storefront. Error boundaries are still a class-only
 * feature (see the React docs on Component); the fallback UI below is a
 * function component so it can follow the visitor's locale. */

import { Component, type ReactNode } from 'react';
import { useLanguage } from '@/lib/i18n/context';

const FALLBACK = {
  zh: {
    note: '在线客服暂时不可用，页面其余部分不受影响。',
    retry: '重试',
  },
  en: {
    note: 'The live chat hit an error — the rest of the page is unaffected.',
    retry: 'Try again',
  },
} as const;

function ChatFallback({ onRetry }: { onRetry: () => void }) {
  const { locale } = useLanguage();
  const copy = FALLBACK[locale === 'zh' ? 'zh' : 'en'];
  return (
    <div
      role="alert"
      className="flex min-h-[220px] flex-col items-center justify-center gap-3 p-6 text-center text-sm"
    >
      <p>{copy.note}</p>
      <button
        type="button"
        onClick={onRetry}
        className="rounded-full border px-4 py-1.5"
      >
        {copy.retry}
      </button>
    </div>
  );
}

interface ChatErrorBoundaryState {
  failed: boolean;
}

export class ChatErrorBoundary extends Component<
  { children: ReactNode },
  ChatErrorBoundaryState
> {
  state: ChatErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ChatErrorBoundaryState {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <ChatFallback
          onRetry={() => this.setState({ failed: false })}
        />
      );
    }
    return this.props.children;
  }
}
