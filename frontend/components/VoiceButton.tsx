'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Mic, MicOff, Radio, Square, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTranslation } from '@/lib/i18n';
import { useHydrated } from '@/lib/use-hydrated';

// Web Speech API type declarations
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

interface VoiceButtonBaseProps {
  sessionId?: number;
  stream?: boolean;
  disabled?: boolean;
  language?: string;
  onTranscription?: (text: string, confidence?: number) => void;
  onChatResponse?: (content: string, sessionId: number) => void;
  onResponseChunk?: (chunk: string) => void;
  onResponseComplete?: (response: string) => void;
  onError?: (message: string) => void;
  className?: string;
}

/**
 * Map a Web Speech API error code to an i18n message key.
 *
 * There is deliberately NO pre-flight reachability probe here. The old
 * gate fetched google.com/generate_204 with mode:'cors' and disabled the
 * mic button whenever that fetch threw — which also happens when the
 * endpoint simply doesn't return CORS headers, permanently breaking the
 * button for users who CAN reach Google (observed 2026-09-30). The
 * recognition engine is the source of truth: it fires onerror with
 * 'network' within a second when its speech backend is unreachable.
 */
function speechErrorToMessageKey(error: string): string | null {
  switch (error) {
    case 'network':
      return 'voice.needsGoogle';
    case 'not-allowed':
    case 'service-not-allowed':
    case 'audio-capture':
      return 'voice.micDenied';
    case 'language-not-supported':
      return 'voice.browserNotSupported';
    // no-speech / aborted: the user paused or said nothing — not an error.
    default:
      return null;
  }
}

// 手动模式按钮 - 按住说话，松开结束
interface ManualVoiceButtonProps extends VoiceButtonBaseProps {}

export function ManualVoiceButton({
  sessionId = 1,
  stream = true,
  disabled = false,
  language = 'zh-CN',
  onTranscription,
  onChatResponse,
  onResponseChunk,
  onResponseComplete,
  onError,
  className,
}: ManualVoiceButtonProps) {
  const { t } = useTranslation();
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const hydrated = useHydrated();
  const webSpeechSupported = hydrated && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  const [transcript, setTranscript] = useState('');
  const [isProcessingResult, setIsProcessingResult] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const displayError = error ?? (hydrated && !webSpeechSupported ? t('voice.browserNotSupported') : null);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const isProcessingResultRef = useRef(false);
  const processingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const accumulatedTranscriptRef = useRef<string>('');
  const currentInterimRef = useRef<string>('');

  useEffect(() => () => {
    if (processingTimerRef.current) clearTimeout(processingTimerRef.current);
    const recognition = recognitionRef.current;
    recognitionRef.current = null;
    releaseRecognition(recognition, true);
  }, []);

  const callbacksRef = useRef({
    onTranscription,
    onChatResponse,
    onResponseChunk,
    onResponseComplete,
    onError,
  });

  useEffect(() => {
    callbacksRef.current = {
      onTranscription,
      onChatResponse,
      onResponseChunk,
      onResponseComplete,
      onError,
    };
  });

  const failRecognition = useCallback((recognition: SpeechRecognition | null, message: string) => {
    if (recognitionRef.current !== recognition) return;
    recognitionRef.current = null;
    releaseRecognition(recognition, true);
    setIsRecording(false);
    setIsSpeaking(false);
    setError(message);
    callbacksRef.current.onError?.(message);
  }, []);

  const handleTranscriptionResult = useCallback((text: string, confidence: number) => {
    if (!text.trim() || isProcessingResultRef.current) return;

    isProcessingResultRef.current = true;
    setIsProcessingResult(true);

    // 只负责语音识别，API调用由父组件处理
    callbacksRef.current.onTranscription?.(text, confidence);

    // 短暂延迟后重置状态，让父组件有时间处理
    processingTimerRef.current = setTimeout(() => {
      processingTimerRef.current = null;
      isProcessingResultRef.current = false;
      setIsProcessingResult(false);
    }, 100);
  }, []);

  const initRecognition = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return null;

    const recognition = new SpeechRecognition();
    recognition.continuous = true; // 持续监听，不自动切断
    recognition.interimResults = true;
    recognition.lang = language;
    recognition.maxAlternatives = 1;

    // 重置累积的文字
    accumulatedTranscriptRef.current = '';
    currentInterimRef.current = '';

    recognition.onstart = () => {
      if (recognitionRef.current !== recognition) return;
      setIsSpeaking(true);
      setIsRecording(true);
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      if (recognitionRef.current !== recognition) return;
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          accumulatedTranscriptRef.current += result[0].transcript;
        } else {
          interimTranscript += result[0].transcript;
        }
      }

      // 保存当前临时文字
      currentInterimRef.current = interimTranscript;

      // 显示累积的文字 + 当前临时文字
      const displayText = accumulatedTranscriptRef.current + interimTranscript;
      setTranscript(displayText);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (recognitionRef.current !== recognition) return;
      const messageKey = speechErrorToMessageKey(event.error);
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setPermissionDenied(true);
      }
      if (messageKey) {
        failRecognition(recognition, t(messageKey));
      }
    };

    recognition.onend = () => {
      if (recognitionRef.current !== recognition) return;
      recognitionRef.current = null;
      setIsSpeaking(false);
      setIsRecording(false);
      // 不在这里发送，让用户手动控制
    };

    return recognition;
  }, [language, t, failRecognition]);

  const voiceDisabled = !isRecording && (disabled || !webSpeechSupported || permissionDenied || isProcessingResult);

  const toggleRecording = useCallback((e?: React.MouseEvent) => {
    e?.preventDefault();
    if (recognitionRef.current) {
      // 停止录音，发送累积的文字 + 当前临时文字
      const textToSend = (accumulatedTranscriptRef.current + currentInterimRef.current).trim();
      const recognition = recognitionRef.current;
      recognitionRef.current = null;
      releaseRecognition(recognition, false);
      setIsRecording(false);
      setIsSpeaking(false);
      setTranscript('');

      // 发送识别结果
      if (textToSend) {
        handleTranscriptionResult(textToSend, 0.9);
      }
      accumulatedTranscriptRef.current = '';
      currentInterimRef.current = '';
    } else {
      // 开始录音
      if (disabled || !webSpeechSupported || permissionDenied || isProcessingResultRef.current) return;
      accumulatedTranscriptRef.current = '';
      currentInterimRef.current = '';
      try {
        recognitionRef.current = initRecognition();
        setIsRecording(true);
        setError(null);
        recognitionRef.current?.start();
      } catch {
        failRecognition(recognitionRef.current, t('voice.recognitionFailed'));
      }
    }
  }, [disabled, webSpeechSupported, permissionDenied, initRecognition, handleTranscriptionResult, failRecognition, t]);

  const getIcon = () => {
    if (!webSpeechSupported || permissionDenied) return <MicOff className="h-4 w-4" />;
    if (isProcessingResult) return <Loader2 className="h-4 w-4 animate-spin" />;
    return isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />;
  };

  return (
    <div className={className}>
      <Button
        type="button"
        variant={isRecording ? 'destructive' : 'default'}
        size="icon"
        disabled={voiceDisabled}
        onClick={toggleRecording}
        className={`relative ${isRecording ? 'animate-pulse' : ''}`}
        title={isRecording ? t('voice.clickToStop') : t('voice.clickToRecord')}
      >
        {getIcon()}
        {isRecording && (
          <motion.div
            className="absolute inset-0 rounded-md bg-red-500/20"
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 0.8, repeat: Infinity }}
          />
        )}
      </Button>

      {transcript && isRecording && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 rounded-md bg-muted p-2 text-xs max-w-[200px]"
        >
          {transcript}
        </motion.div>
      )}

      {displayError && (
        <p className="mt-2 text-xs text-red-500">{displayError}</p>
      )}
    </div>
  );
}

// 自动模式按钮 - 点击开启自动监听
interface AutoVoiceButtonProps extends VoiceButtonBaseProps {}

export function AutoVoiceButton({
  sessionId = 1,
  stream = true,
  disabled = false,
  language = 'zh-CN',
  onTranscription,
  onChatResponse,
  onResponseChunk,
  onResponseComplete,
  onError,
  className,
}: AutoVoiceButtonProps) {
  const { t } = useTranslation();
  const [isAutoMode, setIsAutoMode] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const hydrated = useHydrated();
  const webSpeechSupported = hydrated && Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  const [transcript, setTranscript] = useState('');
  const [isProcessingResult, setIsProcessingResult] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const displayError = error ?? (hydrated && !webSpeechSupported ? t('voice.browserNotSupported') : null);

  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const isProcessingResultRef = useRef(false);
  const processingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isAutoVadActiveRef = useRef(false);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const restartTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastTranscriptRef = useRef<string>('');

  useEffect(() => () => {
    isAutoVadActiveRef.current = false;
    if (processingTimerRef.current) clearTimeout(processingTimerRef.current);
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    if (restartTimerRef.current) clearTimeout(restartTimerRef.current);
    const recognition = recognitionRef.current;
    recognitionRef.current = null;
    releaseRecognition(recognition, true);
  }, []);

  const callbacksRef = useRef({
    onTranscription,
    onChatResponse,
    onResponseChunk,
    onResponseComplete,
    onError,
  });

  useEffect(() => {
    callbacksRef.current = { onTranscription, onChatResponse, onResponseChunk, onResponseComplete, onError };
  });

  const failRecognition = useCallback((recognition: SpeechRecognition | null, message: string) => {
    if (recognitionRef.current !== recognition) return;
    recognitionRef.current = null;
    isAutoVadActiveRef.current = false;
    if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    if (restartTimerRef.current) clearTimeout(restartTimerRef.current);
    releaseRecognition(recognition, true);
    lastTranscriptRef.current = '';
    setTranscript('');
    setIsAutoMode(false);
    setIsSpeaking(false);
    setError(message);
    callbacksRef.current.onError?.(message);
  }, []);

  const handleTranscriptionResult = useCallback((text: string, confidence: number) => {
    if (!text.trim() || isProcessingResultRef.current) return;

    isProcessingResultRef.current = true;
    setIsProcessingResult(true);

    // 只负责语音识别，API调用由父组件处理
    callbacksRef.current.onTranscription?.(text, confidence);

    // 短暂延迟后重置状态，让父组件有时间处理
    processingTimerRef.current = setTimeout(() => {
      processingTimerRef.current = null;
      isProcessingResultRef.current = false;
      setIsProcessingResult(false);
    }, 100);
  }, []);

  const initRecognition = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return null;

    const recognition = new SpeechRecognition();
    recognition.continuous = true; // 持续监听
    recognition.interimResults = true;
    recognition.lang = language;
    recognition.maxAlternatives = 1;

    // 清除静默计时器
    const clearSilenceTimer = () => {
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
    };

    // 启动静默计时器 (2.5秒无新语音则处理结果)
    const startSilenceTimer = () => {
      clearSilenceTimer();
      silenceTimerRef.current = setTimeout(() => {
        const text = lastTranscriptRef.current.trim();
        if (text && !isProcessingResultRef.current) {
          handleTranscriptionResult(text, 0.9);
          lastTranscriptRef.current = '';
          setTranscript('');
        }
      }, 2000);
    };

    recognition.onstart = () => {
      if (recognitionRef.current !== recognition) return;
      setIsSpeaking(true);
      lastTranscriptRef.current = '';
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      if (recognitionRef.current !== recognition) return;
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          lastTranscriptRef.current += result[0].transcript;
        } else {
          interimTranscript += result[0].transcript;
        }
      }

      const displayText = lastTranscriptRef.current || interimTranscript;
      setTranscript(displayText);

      // 每次有新语音时重置静默计时器
      if (displayText.trim()) {
        startSilenceTimer();
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (recognitionRef.current !== recognition) return;
      clearSilenceTimer();
      const messageKey = speechErrorToMessageKey(event.error);
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setPermissionDenied(true);
      }
      if (messageKey) {
        failRecognition(recognition, t(messageKey));
      }
    };

    recognition.onend = () => {
      if (recognitionRef.current !== recognition) return;
      setIsSpeaking(false);
      clearSilenceTimer();

      // 处理剩余的文字
      const text = lastTranscriptRef.current.trim();
      if (text && !isProcessingResultRef.current) {
        handleTranscriptionResult(text, 0.9);
        lastTranscriptRef.current = '';
      }

      setTranscript('');

      // 自动重启
      if (isAutoVadActiveRef.current) {
        if (restartTimerRef.current) clearTimeout(restartTimerRef.current);
        restartTimerRef.current = setTimeout(() => {
          restartTimerRef.current = null;
          if (isAutoVadActiveRef.current && recognitionRef.current === recognition) {
            try {
              recognition.start();
            } catch {
              failRecognition(recognition, t('voice.recognitionFailed'));
            }
          }
        }, 500);
      }
    };

    return recognition;
  }, [language, handleTranscriptionResult, t, failRecognition]);

  const voiceDisabled = !isAutoMode && (disabled || !webSpeechSupported || permissionDenied || isProcessingResult);

  const toggleAutoMode = useCallback((e?: React.MouseEvent) => {
    e?.preventDefault();
    if (isAutoVadActiveRef.current) {
      // 停止
      isAutoVadActiveRef.current = false;
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      if (restartTimerRef.current) {
        clearTimeout(restartTimerRef.current);
        restartTimerRef.current = null;
      }
      const text = lastTranscriptRef.current.trim();
      const recognition = recognitionRef.current;
      recognitionRef.current = null;
      releaseRecognition(recognition, false);
      if (text) handleTranscriptionResult(text, 0.9);
      lastTranscriptRef.current = '';
      setTranscript('');
      setIsSpeaking(false);
      setIsAutoMode(false);
    } else {
      // 启动
      if (disabled || !webSpeechSupported || permissionDenied || isProcessingResultRef.current) return;
      isAutoVadActiveRef.current = true;
      lastTranscriptRef.current = '';
      setIsAutoMode(true);
      setError(null);
      try {
        recognitionRef.current = initRecognition();
        recognitionRef.current?.start();
      } catch {
        failRecognition(recognitionRef.current, t('voice.recognitionFailed'));
      }
    }
  }, [disabled, webSpeechSupported, permissionDenied, initRecognition, handleTranscriptionResult, failRecognition, t]);

  const getIcon = () => {
    if (!webSpeechSupported || permissionDenied) return <MicOff className="h-4 w-4" />;
    if (isProcessingResult) return <Loader2 className="h-4 w-4 animate-spin" />;

    if (isAutoMode) {
      return isSpeaking ? (
        <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 0.5, repeat: Infinity }}>
          <Radio className="h-4 w-4 text-red-500" />
        </motion.div>
      ) : (
        <Radio className="h-4 w-4 text-green-500" />
      );
    }

    return <Radio className="h-4 w-4" />;
  };

  return (
    <div className={className}>
      <Button
        type="button"
        variant={isAutoMode ? 'secondary' : 'outline'}
        size="default"
        disabled={voiceDisabled}
        onClick={toggleAutoMode}
        className={`relative ${isSpeaking ? 'animate-pulse' : ''}`}
      >
        {getIcon()}
        <span className="ml-2">{isAutoMode ? (isSpeaking ? t('voice.recognizing') : t('voice.autoMode')) : t('voice.autoMode')}</span>
        {isAutoMode && !isSpeaking && (
          <motion.div
            className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-green-500"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        )}
      </Button>

      {transcript && isSpeaking && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-2 rounded-md bg-muted p-2 text-xs"
        >
          {transcript}
        </motion.div>
      )}

      {displayError && (
        <p className="mt-2 text-xs text-red-500">{displayError}</p>
      )}
    </div>
  );
}

function releaseRecognition(recognition: SpeechRecognition | null, abort: boolean) {
  if (!recognition) return;
  recognition.onstart = null;
  recognition.onresult = null;
  recognition.onerror = null;
  recognition.onend = null;
  try {
    if (abort) recognition.abort();
    else recognition.stop();
  } catch {
    // The browser may already have ended this recognition session.
  }
}

// 兼容旧组件 - 保留但标记为废弃
export type VoiceButtonMode = 'manual' | 'auto';

interface VoiceButtonProps extends VoiceButtonBaseProps {
  mode?: VoiceButtonMode;
}

export function VoiceButton({ mode = 'manual', ...props }: VoiceButtonProps) {
  if (mode === 'auto') {
    return <AutoVoiceButton {...props} />;
  }
  return <ManualVoiceButton {...props} />;
}
