// API configuration for RAG Chatbot.
// NEXT_PUBLIC_API_URL is inlined at build time. An explicit string (including
// "") means relative same-origin requests; only local .env.local overrides it.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? '';

// Clean up the base URL (remove trailing slash if present)
// This prevents double-slash URLs like "//api/v1/chat"
const cleanBaseUrl = API_BASE_URL.replace(/\/+$/, '');

// Convert HTTP URL to WebSocket URL
const getWebSocketUrl = (httpUrl: string): string => {
  return httpUrl
    .replace(/^http:\/\//, 'ws://')
    .replace(/^https:\/\//, 'wss://');
};

export const API_ENDPOINTS = {
  // Authentication (optional in demo mode)
  LOGIN: `${cleanBaseUrl}/api/v1/auth/login`,
  REGISTER: `${cleanBaseUrl}/api/v1/auth/register`,
  REFRESH: `${cleanBaseUrl}/api/v1/auth/refres`,

  // Chat
  CHAT: `${cleanBaseUrl}/api/v1/chat`,
  CHAT_STREAM: `${cleanBaseUrl}/api/v1/chat/stream`,
  CHAT_HISTORY: `${cleanBaseUrl}/api/v1/chat/history`,

  // Sessions
  SESSIONS: `${cleanBaseUrl}/api/v1/sessions`,
  SESSION_DETAIL: (id: number) => `${cleanBaseUrl}/api/v1/sessions/${id}`,
  SESSION_MESSAGES: (id: number) => `${cleanBaseUrl}/api/v1/sessions/${id}/messages`,

  // Documents
  UPLOAD_DOCUMENT: `${cleanBaseUrl}/api/v1/documents/upload`,
  SEARCH_DOCUMENTS: `${cleanBaseUrl}/api/v1/documents/search`,

  // Health
  HEALTH: `${cleanBaseUrl}/health`,

  // Feedback
  FEEDBACK: `${cleanBaseUrl}/api/v1/feedback`,
  FEEDBACK_STATS: `${cleanBaseUrl}/api/v1/feedback/stats`,

  // Voice WebSocket
  VOICE_WS: `${getWebSocketUrl(cleanBaseUrl)}/api/v1/voice/ws`,
  VOICE_HEALTH: `${cleanBaseUrl}/api/v1/voice/health`,
} as const;

export type ChatStreamErrorKind =
  | 'http' | 'network' | 'interrupted' | 'server_error' | 'invalid_response' | 'aborted';

// Only fixed diagnostics cross the transport/UI boundary; never response bodies.
export class ChatStreamError extends Error {
  constructor(readonly kind: ChatStreamErrorKind, readonly status?: number) {
    const messages: Record<ChatStreamErrorKind, string> = {
      http: `Chat request failed (HTTP ${status ?? 'unknown'}).`,
      network: 'Unable to connect to the chat service.',
      interrupted: 'Chat stream ended before completion.',
      server_error: 'Stream aborted by the server.',
      invalid_response: 'Chat service returned an invalid response.',
      aborted: 'Chat request was cancelled.',
    };
    super(messages[kind]);
    this.name = 'ChatStreamError';
  }
}

function parseTraceEvent(value: unknown): TraceEvent | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const event = value as Record<string, unknown>;
  if (typeof event.stage !== 'string' || !event.stage.trim()) return null;
  if (typeof event.status !== 'string' || !['start', 'detail', 'end', 'error'].includes(event.status)) return null;
  if (event.ms != null && (typeof event.ms !== 'number' || !Number.isFinite(event.ms) || event.ms < 0)) return null;
  if (event.detail !== undefined && (!event.detail || typeof event.detail !== 'object' || Array.isArray(event.detail))) return null;
  return {
    stage: event.stage,
    status: event.status as TraceEvent['status'],
    ...(typeof event.ms === 'number' ? { ms: event.ms } : {}),
    ...(event.detail !== undefined ? { detail: event.detail as Record<string, unknown> } : {}),
  };
}

function positiveSafeId(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

function minorUnits(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function parseCommerceApproval(value: unknown): CommerceApproval | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const card = value as Record<string, unknown>;
  if (typeof card.purchase_id !== 'string' || !card.purchase_id) return null;
  if (card.stage !== 'quoted' && card.stage !== 'awaiting_payment') return null;
  if (card.action !== 'confirm_order' && card.action !== 'confirm_payment') return null;
  if (card.stage === 'quoted' && card.action !== 'confirm_order') return null;
  if (card.stage === 'awaiting_payment' && card.action !== 'confirm_payment') return null;
  if (typeof card.action_label !== 'string' || !card.action_label) return null;
  if (!Array.isArray(card.lines) || card.lines.length < 1 || card.lines.length > 50) return null;
  const lines: CommerceApprovalLine[] = [];
  for (const row of card.lines) {
    if (!row || typeof row !== 'object' || Array.isArray(row)) return null;
    const line = row as Record<string, unknown>;
    if (
      typeof line.sku_id !== 'string' || !line.sku_id ||
      typeof line.name !== 'string' ||
      !minorUnits(line.quantity) || line.quantity < 1 ||
      !minorUnits(line.unit_price_minor) || !minorUnits(line.subtotal_minor)
    ) return null;
    lines.push({ sku_id: line.sku_id, name: line.name, quantity: line.quantity, unit_price_minor: line.unit_price_minor, subtotal_minor: line.subtotal_minor });
  }
  if (
    !minorUnits(card.total_minor) || card.currency !== 'CNY' ||
    typeof card.quote_hash !== 'string' || !card.quote_hash ||
    typeof card.expires_at !== 'string' || !Number.isFinite(Date.parse(card.expires_at)) ||
    !minorUnits(card.purchase_version) || !minorUnits(card.generation)
  ) return null;
  return {
    purchase_id: card.purchase_id, stage: card.stage, action: card.action, action_label: card.action_label,
    lines, total_minor: card.total_minor, currency: 'CNY', quote_hash: card.quote_hash,
    expires_at: card.expires_at, purchase_version: card.purchase_version, generation: card.generation,
  };
}

function parseCommerceWatch(value: unknown): CommerceWatch | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const card = value as Record<string, unknown>;
  if (typeof card.watch_id !== 'string' || !card.watch_id) return null;
  if (typeof card.sku_id !== 'string' || !card.sku_id) return null;
  if (typeof card.sku_name !== 'string') return null;
  if (!minorUnits(card.target_price_minor) || card.target_price_minor < 1) return null;
  if (!minorUnits(card.current_price_minor)) return null;
  if (card.currency !== 'CNY' || card.condition !== 'at_or_below' || card.action !== 'ask_order') return null;
  if (typeof card.action_label !== 'string' || !card.action_label) return null;
  if (typeof card.triggered_at !== 'string' || !Number.isFinite(Date.parse(card.triggered_at))) return null;
  if (typeof card.expires_at !== 'string' || !Number.isFinite(Date.parse(card.expires_at))) return null;
  return {
    watch_id: card.watch_id, sku_id: card.sku_id, sku_name: card.sku_name,
    target_price_minor: card.target_price_minor, current_price_minor: card.current_price_minor,
    currency: 'CNY', condition: 'at_or_below', action: 'ask_order', action_label: card.action_label,
    triggered_at: card.triggered_at, expires_at: card.expires_at,
  };
}

function parseCompletion(value: unknown, sessionId: number, sessionKey?: string): ChatCompletion | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const receipt = value as Record<string, unknown>;
  if (!positiveSafeId(receipt.message_id) || receipt.session_id !== sessionId || receipt.status !== 'completed') return null;
  if (sessionKey !== undefined && receipt.session_key !== sessionKey) return null;
  // The structured cards ride the same completion receipt as the
  // CommerceView `ui` payload; an invalid or absent card never invalidates
  // the receipt itself.
  const ui = receipt.ui;
  const approval = ui && typeof ui === 'object' && !Array.isArray(ui)
    ? parseCommerceApproval((ui as Record<string, unknown>).approval)
    : null;
  const watch = ui && typeof ui === 'object' && !Array.isArray(ui)
    ? parseCommerceWatch((ui as Record<string, unknown>).watch)
    : null;
  return { message_id: receipt.message_id, session_id: sessionId, status: 'completed', ...(sessionKey ? { session_key: sessionKey } : {}), ...(approval ? { approval } : {}), ...(watch ? { watch } : {}) };
}

interface SseFrame { event: string; data: string }

// Line parsing preserves split CRLF delimiters and ignores comment lines only.
// Fetch POST has no automatic EventSource reconnect: replay may repeat an action.
class SseFrameParser {
  private line = '';
  private skipLF = false;
  private event = '';
  private data: string[] = [];

  *feed(text: string): Generator<SseFrame> {
    for (const char of text) {
      if (this.skipLF) {
        this.skipLF = false;
        if (char === '\n') continue;
      }
      if (char !== '\r' && char !== '\n') {
        this.line += char;
        continue;
      }
      this.skipLF = char === '\r';
      const line = this.line;
      this.line = '';
      if (!line) {
        if (this.data.length) {
          const frame = { event: this.event || 'message', data: this.data.join('\n') };
          this.event = '';
          this.data = [];
          yield frame;
        }
        this.event = '';
        this.data = [];
        continue;
      }
      if (line.startsWith(':')) continue;
      const colon = line.indexOf(':');
      const field = colon < 0 ? line : line.slice(0, colon);
      const value = colon < 0 ? '' : line.slice(colon + 1).replace(/^ /, '');
      if (field === 'event') this.event = value;
      if (field === 'data') this.data.push(value);
    }
  }
}

export interface ChatStreamTransport {
  endpoint: string;
  headers: Record<string, string>;
  credentials: RequestCredentials;
  expectedSessionKey: string;
}

// API client helper functions
export class ApiClient {
  private static getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // Add auth token if available (demo mode: token is optional)
    const token = localStorage.getItem('access_token');
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
  }

  // Generic request method
  private static async request<T>(
    url: string,
    options: RequestInit = {}
  ): Promise<T> {
    let response: Response;
    try {
      response = await fetch(url, {
        ...options,
        headers: { ...this.getHeaders(), ...options.headers },
      });
    } catch {
      throw new ChatStreamError(options.signal?.aborted ? 'aborted' : 'network');
    }
    if (!response.ok) {
      try { await response.body?.cancel(); } catch { /* preserve the HTTP status */ }
      throw new ChatStreamError('http', response.status);
    }
    if (response.status === 204 && options.method === 'DELETE') return undefined as T;
    try {
      return await response.json() as T;
    } catch {
      throw new ChatStreamError('invalid_response');
    }
  }

  // POST request
  static async post<T>(
    url: string,
    data: Record<string, unknown>
  ): Promise<T> {
    return this.request<T>(url, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // GET request
  static async get<T>(url: string): Promise<T> {
    return this.request<T>(url, {
      method: 'GET',
    });
  }

  // DELETE request
  static async delete<T>(url: string): Promise<T> {
    return this.request<T>(url, {
      method: 'DELETE',
    });
  }

  // Non-streaming chat
  static async chat(message: string, sessionId: number = 1): Promise<ChatResponse> {
    return this.post<ChatResponse>(API_ENDPOINTS.CHAT, {
      message,
      session_id: sessionId,
    });
  }

  // Streaming chat using Server-Sent Events.
  // The backend multiplexes named `event: trace` pipeline events onto the
  // same stream as content tokens — content frames are yielded as before,
  // trace frames are forwarded to `onTraceEvent` when provided (and always
  // excluded from the yielded content).
  static async *streamChat(
    message: string,
    sessionId: number = 1,
    onTraceEvent?: (event: TraceEvent) => void,
    onCitations?: (citations: SourceCitation[]) => void,
    signal?: AbortSignal,
    onCompletion?: (receipt: ChatCompletion) => void,
    transport?: ChatStreamTransport,
  ): AsyncGenerator<string, void, unknown> {
    let response: Response;
    try {
      if (signal?.aborted) throw new ChatStreamError('aborted');
      response = await fetch(transport?.endpoint ?? API_ENDPOINTS.CHAT_STREAM, {
        method: 'POST',
        headers: { ...(transport?.headers ?? this.getHeaders()), Accept: 'text/event-stream' },
        body: JSON.stringify({ message, session_id: sessionId, include_citations: Boolean(onCitations), include_completion: Boolean(onCompletion), ...(transport ? { expected_session_key: transport.expectedSessionKey } : {}) }),
        ...(transport ? { credentials: transport.credentials } : {}),
        signal,
        cache: 'no-store',
      });
    } catch (error) {
      if (error instanceof ChatStreamError) throw error;
      if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        throw new ChatStreamError('aborted');
      }
      throw new ChatStreamError('network');
    }

    const contentType = response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase();
    if (!response.ok || contentType !== 'text/event-stream') {
      try { await response.body?.cancel(); } catch { /* preserve the request failure */ }
      throw response.ok
        ? new ChatStreamError('invalid_response')
        : new ChatStreamError('http', response.status);
    }
    const reader = response.body?.getReader();
    if (!reader) throw new ChatStreamError('invalid_response');
    const decoder = new TextDecoder();
    const parser = new SseFrameParser();
    let completion: ChatCompletion | null = null;
    let invalidCompletion = false;
    let hasAnswer = false;
    try {
      while (true) {
        if (signal?.aborted) throw new ChatStreamError('aborted');
        const { done, value } = await reader.read();
        if (signal?.aborted) throw new ChatStreamError('aborted');
        const decoded = done ? decoder.decode() : decoder.decode(value, { stream: true });
        for (const frame of parser.feed(decoded)) {
          if (signal?.aborted) throw new ChatStreamError('aborted');
          if (frame.event === 'trace') {
            try {
              const event = parseTraceEvent(JSON.parse(frame.data));
              if (event) onTraceEvent?.(event);
            } catch { /* diagnostic metadata only */ }
            continue;
          }
          if (frame.event === 'citations') {
            try {
              const payload: unknown = JSON.parse(frame.data);
              if (payload && typeof payload === 'object' && 'citations' in payload) {
                onCitations?.(parseSourceCitations(payload.citations));
              }
            } catch { /* invalid metadata never becomes answer text */ }
            continue;
          }
          if (frame.event === 'completion') {
            if (onCompletion) {
              let candidate: ChatCompletion | null = null;
              try { candidate = parseCompletion(JSON.parse(frame.data), sessionId, transport?.expectedSessionKey); } catch { /* unusable receipt */ }
              if (!hasAnswer || !candidate || (completion && completion.message_id !== candidate.message_id)) {
                invalidCompletion = true;
              } else {
                completion = candidate;
              }
            }
            continue;
          }
          if (frame.event !== 'message') continue;
          if (frame.data === '[DONE]') {
            if (hasAnswer && completion && !invalidCompletion) onCompletion?.(completion);
            return;
          }
          if (frame.data === '[STREAM_ERROR]') throw new ChatStreamError('server_error');
          if (frame.data) {
            if (completion) invalidCompletion = true;
            if (frame.data.trim()) hasAnswer = true;
            yield frame.data;
          }
        }
        if (done) throw new ChatStreamError('interrupted');
      }
    } catch (error) {
      if (error instanceof ChatStreamError) throw error;
      if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) {
        throw new ChatStreamError('aborted');
      }
      throw new ChatStreamError('interrupted');
    } finally {
      try { await reader.cancel(); } catch { /* cancellation must not replace the outcome */ }
      try { reader.releaseLock(); } catch { /* cleanup must not replace the outcome */ }
    }
  }

  // History uses the same shared demo session as chat. In strict mode, the
  // backend checks ownership using the normal Authorization header.
  static async getChatHistory(sessionId: number, limit: number = 50): Promise<ChatHistoryResponse> {
    const history = await this.get<ChatHistoryResponse>(
      `${API_ENDPOINTS.CHAT_HISTORY}?session_id=${sessionId}&limit=${limit}`
    );
    if (!history || history.session_id !== sessionId || !Array.isArray(history.messages) ||
      history.messages.some(message => !message || typeof message !== 'object' ||
        !['user', 'assistant', 'system'].includes(message.role) || typeof message.content !== 'string')) {
      throw new ChatStreamError('invalid_response');
    }
    const counts = new Map<number, number>();
    for (const message of history.messages) {
      if (positiveSafeId(message.message_id)) counts.set(message.message_id, (counts.get(message.message_id) ?? 0) + 1);
    }
    return {
      session_id: sessionId,
      messages: history.messages.map(message => {
        const id = positiveSafeId(message.message_id) && counts.get(message.message_id) === 1 ? message.message_id : undefined;
        const status = typeof message.status === 'string' && ['pending', 'processing', 'completed', 'failed'].includes(message.status)
          ? message.status : undefined;
        const rating = id && (message.user_rating === -1 || message.user_rating === 0 || message.user_rating === 1)
          ? message.user_rating : undefined;
        return { ...message, message_id: id, status, user_rating: rating, citations: parseSourceCitations(message.citations) };
      }),
    };
  }

  static async submitFeedback(messageId: number, rating: -1 | 0 | 1): Promise<FeedbackResponse> {
    if (!positiveSafeId(messageId) || ![-1, 0, 1].includes(rating)) throw new ChatStreamError('invalid_response');
    const result = await this.post<FeedbackResponse>(API_ENDPOINTS.FEEDBACK, { message_id: messageId, rating });
    if (!result || result.success !== true || result.message_id !== messageId || result.rating !== rating) {
      throw new ChatStreamError('invalid_response');
    }
    return { success: true, message_id: messageId, rating };
  }

  // Health check
  static async healthCheck(): Promise<{ status: string; version?: string }> {
    return this.get<{ status: string; version?: string }>(API_ENDPOINTS.HEALTH);
  }

  // Get sessions list
  static async getSessions(limit: number = 10): Promise<SessionsResponse> {
    return this.get<SessionsResponse>(`${API_ENDPOINTS.SESSIONS}?limit=${limit}`);
  }

  // Get session messages
  static async getSessionMessages(sessionId: number, limit: number = 50): Promise<MessagesResponse> {
    return this.get<MessagesResponse>(`${API_ENDPOINTS.SESSION_MESSAGES(sessionId)}?limit=${limit}`);
  }

  // Create new session
  static async createSession(title: string = 'New Chat'): Promise<SessionResponse> {
    return this.post<SessionResponse>(API_ENDPOINTS.SESSIONS, { title });
  }

  // Delete session
  static async deleteSession(sessionId: number): Promise<void> {
    return this.delete<void>(API_ENDPOINTS.SESSION_DETAIL(sessionId));
  }
}

// API Types matching backend response format

// Retrieved reference snapshots, not a claim-by-claim verification result.
// Graph provenance arrays are independent sets; do not zip them into mappings.
export interface SourceCitation {
  source_type: 'document' | 'graph' | 'faq';
  document_id?: string | null;
  version_id?: string | null;
  chunk_id?: string | null;
  faq_id?: string | null;
  title?: string | null;
  excerpt: string;
  page_start?: number | null;
  page_end?: number | null;
  start_pos?: number | null;
  end_pos?: number | null;
  section_path: string[];
  source_document_ids: string[];
  source_version_ids: string[];
  source_chunk_ids: string[];
}

export function parseSourceCitations(value: unknown): SourceCitation[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is SourceCitation => {
    if (!item || typeof item !== 'object') return false;
    const citation = item as Record<string, unknown>;
    if (!['document', 'graph', 'faq'].includes(String(citation.source_type))) return false;
    if (typeof citation.excerpt !== 'string') return false;
    const textFields = ['document_id', 'version_id', 'chunk_id', 'faq_id', 'title'];
    if (textFields.some((key) => citation[key] != null && typeof citation[key] !== 'string')) return false;
    const positionFields = ['page_start', 'page_end', 'start_pos', 'end_pos'];
    if (positionFields.some((key) => citation[key] != null &&
      (typeof citation[key] !== 'number' || !Number.isInteger(citation[key]) || (citation[key] as number) < 0))) return false;
    return ['section_path', 'source_document_ids', 'source_version_ids', 'source_chunk_ids'].every(
      (key) => Array.isArray(citation[key]) && (citation[key] as unknown[]).every((entry) => typeof entry === 'string')
    );
  });
}

// A live pipeline trace event multiplexed onto the chat SSE stream
// (`event: trace` frames). Emitted by dialogue-graph stages as they run.
export interface TraceEvent {
  stage: string;
  status: 'start' | 'detail' | 'end' | 'error';
  ms?: number;
  detail?: Record<string, unknown>;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  citations?: SourceCitation[];
  message_id?: number;
  status?: 'pending' | 'processing' | 'completed' | 'failed';
  user_rating?: -1 | 0 | 1;
}

/** One orderable line of the server's structured confirmation card. */
export interface CommerceApprovalLine {
  sku_id: string;
  name: string;
  quantity: number;
  unit_price_minor: number;
  subtotal_minor: number;
}

/**
 * Structured purchase confirmation card delivered with a completion receipt.
 * Every field is copied server-side from the durable quote confirmation;
 * when the model's reply text disagrees with the card, the card wins.
 */
export interface CommerceApproval {
  purchase_id: string;
  stage: 'quoted' | 'awaiting_payment';
  action: 'confirm_order' | 'confirm_payment';
  action_label: string;
  lines: CommerceApprovalLine[];
  total_minor: number;
  currency: 'CNY';
  quote_hash: string;
  expires_at: string;
  purchase_version: number;
  generation: number;
}

/**
 * Structured price-watch ask card delivered with a completion receipt
 * ("pre-authorized condition + trigger + ask"). Every price and name is
 * copied server-side from the durable watch rule plus the provider price
 * fact that satisfied it; the card only ASKS whether to order now and
 * ordering still travels the ordinary confirmation flow.
 */
export interface CommerceWatch {
  watch_id: string;
  sku_id: string;
  sku_name: string;
  target_price_minor: number;
  current_price_minor: number;
  currency: 'CNY';
  condition: 'at_or_below';
  action: 'ask_order';
  action_label: string;
  triggered_at: string;
  expires_at: string;
}

export interface ChatCompletion {
  message_id: number;
  session_id: number;
  status: 'completed';
  session_key?: string;
  approval?: CommerceApproval;
  watch?: CommerceWatch;
}

export interface FeedbackResponse {
  success: true;
  message_id: number;
  rating: -1 | 0 | 1;
}

export interface ChatHistoryResponse {
  messages: ChatMessage[];
  session_id: number;
}

export interface ChatRequest {
  message: string;
  session_id: number;  // Required, defaults to 1 (demo session)
}

export interface ChatResponse {
  content: string;
  session_id: number;
  intent: string;
  message_id?: number | null;
  status?: 'completed' | null;
  sources?: string[];
  citations?: SourceCitation[];
  metadata?: {
    confidence?: number;
    tokens_used?: number;
  };
}

export interface Session {
  id: number;
  title: string;
  memory_type: string;
  context_window: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface SessionsResponse {
  sessions: Session[];
  total: number;
}

export interface MessagesResponse {
  messages: ChatMessage[];
  total: number;
  session_id: number;
}

export interface SessionResponse {
  session: Session;
}

export interface AuthRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: number;
  email: string;
  full_name: string;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
}

// ==================== Voice API Types ====================

export type VoiceMode = 'idle' | 'manual_recording' | 'auto';

export interface VoiceState {
  mode: VoiceMode;
  isRecording: boolean;
  isConnected: boolean;
  error: string | null;
}

// WebSocket message types (client -> server)
export type VoiceWSMessageType =
  | 'start_recording'
  | 'stop_recording'
  | 'enable_auto'
  | 'disable_auto'
  | 'ping';

export interface VoiceWSMessage {
  type: VoiceWSMessageType;
  data?: {
    session_id?: number;
    user_id?: number;
    stream?: boolean;
  };
}

// WebSocket event types (server -> client)
export type VoiceWSEventType =
  | 'recording_started'
  | 'recording_stopped'
  | 'auto_mode_enabled'
  | 'auto_mode_disabled'
  | 'vad_speech_start'
  | 'vad_speech_end'
  | 'transcription_result'
  | 'chat_response'
  | 'response_chunk'
  | 'response_complete'
  | 'error'
  | 'pong';

export interface VoiceWSEvent {
  type: VoiceWSEventType;
  [key: string]: unknown;
}

export interface TranscriptionResultEvent extends VoiceWSEvent {
  type: 'transcription_result';
  text: string;
  confidence?: number;
}

export interface ChatResponseEvent extends VoiceWSEvent {
  type: 'chat_response';
  content: string;
  session_id: number;
}

export interface ResponseChunkEvent extends VoiceWSEvent {
  type: 'response_chunk';
  chunk: string;
}

export interface ResponseCompleteEvent extends VoiceWSEvent {
  type: 'response_complete';
  response: string;
}

export interface VadSpeechStartEvent extends VoiceWSEvent {
  type: 'vad_speech_start';
  timestamp: number;
}

export interface VadSpeechEndEvent extends VoiceWSEvent {
  type: 'vad_speech_end';
  timestamp: number;
}

export interface ErrorEvent extends VoiceWSEvent {
  type: 'error';
  message: string;
  code?: string;
}

export interface VoiceHealthResponse {
  status: string;
  mode: VoiceMode | null;
  is_running: boolean;
  active_connections: number;
}

// Voice API event handlers
export interface VoiceEventHandlers {
  onRecordingStarted?: () => void;
  onRecordingStopped?: () => void;
  onAutoModeEnabled?: () => void;
  onAutoModeDisabled?: () => void;
  onVadSpeechStart?: (timestamp: number) => void;
  onVadSpeechEnd?: (timestamp: number) => void;
  onTranscription?: (text: string, confidence?: number) => void;
  onChatResponse?: (content: string, sessionId: number) => void;
  onResponseChunk?: (chunk: string) => void;
  onResponseComplete?: (response: string) => void;
  onError?: (message: string, code?: string) => void;
  onConnectionChange?: (connected: boolean) => void;
}

// ==================== Voice API Client ====================

export class VoiceApiClient {
  private ws: WebSocket | null = null;
  private handlers: VoiceEventHandlers = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 1000;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private sessionId: number = 1;
  private stream: boolean = true;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private rejectPendingConnection: ((error: Error) => void) | null = null;
  private stableConnectionTimer: NodeJS.Timeout | null = null;


  constructor(handlers: VoiceEventHandlers = {}) {
    this.handlers = handlers;
  }

  // Update event handlers
  setHandlers(handlers: Partial<VoiceEventHandlers>): void {
    this.handlers = { ...this.handlers, ...handlers };
  }

  // Explicit connects replace the previous session; automatic retries retain their budget.
  async connect(sessionId: number = 1, stream: boolean = true): Promise<void> {
    this.closeConnection('Connection replaced');
    this.sessionId = sessionId;
    this.stream = stream;
    this.reconnectAttempts = 0;
    return this.openConnection();
  }

  private openConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const token = localStorage.getItem('access_token');
        const wsUrl = token
          ? `${API_ENDPOINTS.VOICE_WS}?token=${encodeURIComponent(token)}`
          : API_ENDPOINTS.VOICE_WS;
        const socket = new WebSocket(wsUrl);
        this.ws = socket;
        this.rejectPendingConnection = reject;

        socket.onopen = () => {
          if (this.ws !== socket) return;
          this.rejectPendingConnection = null;
          this.startHeartbeat();
          // A handshake alone does not mean the connection is healthy.
          this.stableConnectionTimer = setTimeout(() => {
            this.stableConnectionTimer = null;
            if (this.ws === socket && socket.readyState === WebSocket.OPEN) {
              this.reconnectAttempts = 0;
            }
          }, 10000);
          resolve();
          this.handlers.onConnectionChange?.(true);
        };

        socket.onclose = (event) => {
          if (this.ws !== socket) return;
          this.ws = null;
          this.stopHeartbeat();
          this.stopStabilityTimer();
          this.rejectPendingConnection = null;
          reject(new Error(`WebSocket closed before connection completed (${event.code})`));
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
          this.handlers.onConnectionChange?.(false);
        };

        socket.onerror = () => {
          if (this.ws !== socket) return;
          this.rejectPendingConnection = null;
          reject(new Error('WebSocket connection failed'));
          this.handlers.onError?.('WebSocket connection error');
        };

        socket.onmessage = (event) => {
          if (this.ws === socket) this.handleMessage(event.data);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  // Handle incoming WebSocket message
  private handleMessage(data: string): void {
    try {
      const message: VoiceWSEvent = JSON.parse(data);
      console.log('[VoiceAPI] Received:', message.type);

      switch (message.type) {
        case 'recording_started':
          this.handlers.onRecordingStarted?.();
          break;

        case 'recording_stopped':
          this.handlers.onRecordingStopped?.();
          break;

        case 'auto_mode_enabled':
          this.handlers.onAutoModeEnabled?.();
          break;

        case 'auto_mode_disabled':
          this.handlers.onAutoModeDisabled?.();
          break;

        case 'vad_speech_start':
          this.handlers.onVadSpeechStart?.((message as VadSpeechStartEvent).timestamp);
          break;

        case 'vad_speech_end':
          this.handlers.onVadSpeechEnd?.((message as VadSpeechEndEvent).timestamp);
          break;

        case 'transcription_result': {
          const evt = message as TranscriptionResultEvent;
          this.handlers.onTranscription?.(evt.text, evt.confidence);
          break;
        }

        case 'chat_response': {
          const evt = message as ChatResponseEvent;
          this.handlers.onChatResponse?.(evt.content, evt.session_id);
          break;
        }

        case 'response_chunk': {
          const evt = message as ResponseChunkEvent;
          this.handlers.onResponseChunk?.(evt.chunk);
          break;
        }

        case 'response_complete': {
          const evt = message as ResponseCompleteEvent;
          this.handlers.onResponseComplete?.(evt.response);
          break;
        }

        case 'error': {
          const evt = message as ErrorEvent;
          this.handlers.onError?.(evt.message, evt.code);
          break;
        }

        case 'pong':
          // Heartbeat response - no action needed
          break;

        default:
          console.warn('[VoiceAPI] Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('[VoiceAPI] Failed to parse message:', error);
    }
  }

  // Send message to WebSocket
  private send(message: VoiceWSMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('[VoiceAPI] WebSocket not connected');
      this.handlers.onError?.('WebSocket not connected');
    }
  }

  // Send binary audio chunk to WebSocket
  sendAudioChunk(chunk: ArrayBuffer): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(chunk);
    } else {
      const state = this.ws === null 
        ? 'not initialized' 
        : `state=${this.ws.readyState}`;
      console.warn(`[VoiceAPI] Cannot send audio chunk: WebSocket ${state}`);
    }
  }

  // Get diagnostic information
  getDiagnostics(): { 
    connected: boolean; 
    readyState: number | null;
    reconnectAttempts: number;
    sessionId: number;
  } {
    return {
      connected: this.ws?.readyState === WebSocket.OPEN,
      readyState: this.ws?.readyState ?? null,
      reconnectAttempts: this.reconnectAttempts,
      sessionId: this.sessionId,
    };
  }

  // Start manual recording
  startRecording(): void {
    if (!this.isConnected()) {
      const diag = this.getDiagnostics();
      this.handlers.onError?.(`Cannot start recording: WebSocket not connected (${diag.readyState})`);
      return;
    }
    this.send({ type: 'start_recording' });
  }

  // Stop manual recording and process
  stopRecording(sessionId?: number): void {
    this.send({
      type: 'stop_recording',
      data: {
        session_id: sessionId ?? this.sessionId,
        stream: this.stream,
      },
    });
  }

  // Enable auto mode (VAD)
  enableAutoMode(sessionId?: number): void {
    this.send({
      type: 'enable_auto',
      data: {
        session_id: sessionId ?? this.sessionId,
        stream: this.stream,
      },
    });
  }

  // Disable auto mode
  disableAutoMode(): void {
    this.send({ type: 'disable_auto' });
  }

  // Start heartbeat
  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 25000); // Send ping every 25 seconds
  }

  // Stop heartbeat
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private stopStabilityTimer(): void {
    if (this.stableConnectionTimer !== null) {
      clearTimeout(this.stableConnectionTimer);
      this.stableConnectionTimer = null;
    }
  }

  // The timer is owned by the client and is cancelled on explicit connect/disconnect.
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.openConnection().catch((error) => {
        console.error('[VoiceAPI] Reconnect failed:', error);
      });
    }, delay);
  }

  private closeConnection(reason: string): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopHeartbeat();
    this.stopStabilityTimer();
    const socket = this.ws;
    // Invalidate ownership before close; delayed callbacks must not affect a new session.
    this.ws = null;
    this.rejectPendingConnection?.(new Error(reason));
    this.rejectPendingConnection = null;
    if (socket) {
      socket.close(1000, reason);
      this.handlers.onConnectionChange?.(false);
    }
  }

  // Disconnect WebSocket and cancel both connecting and scheduled attempts.
  disconnect(): void {
    this.closeConnection('Client disconnect');
  }

  // Check if connected
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // Health check (HTTP)
  static async healthCheck(): Promise<VoiceHealthResponse> {
    const response = await fetch(API_ENDPOINTS.VOICE_HEALTH, {
      headers: ApiClient['getHeaders'](),
    });

    if (!response.ok) {
      throw new Error(`Voice health check failed: ${response.status}`);
    }

    return response.json();
  }
}
