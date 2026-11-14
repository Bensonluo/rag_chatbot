import { API_ENDPOINTS, ApiClient, ChatStreamError, parseSourceCitations, type ChatCompletion, type ChatHistoryResponse, type ChatMessage, type SourceCitation, type TraceEvent } from './api';

export const GUEST_EXPIRED = 'rag-guest-expired';
const BASE = `${API_ENDPOINTS.CHAT.slice(0, -'/chat'.length)}/guest`;
const HEADERS = { 'Content-Type': 'application/json', 'X-Rag-Guest': '1' };
export interface GuestScope { session_id: number; session_key: string }
export interface GuestIdentity { visitor_id: string; expires_at: string; session: { id: number; session_key: string } }
export type GuestMessage = ChatMessage;
export interface GuestHistory extends ChatHistoryResponse { session_key: string; messages: GuestMessage[] }
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const id = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v > 0;
const uuid = (v: unknown): v is string => typeof v === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v);
const invalid = (): never => { throw new ChatStreamError('invalid_response'); };
const validScope = (v: GuestScope) => id(v.session_id) && uuid(v.session_key);
export const guestScope = (v: GuestIdentity): GuestScope => ({ session_id: v.session.id, session_key: v.session.session_key });
let bootstrapFlight: Promise<GuestIdentity> | null = null;

function expired(scope: GuestScope) { if (typeof window !== 'undefined') window.dispatchEvent(new CustomEvent(GUEST_EXPIRED, { detail: scope })); }
function withSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new ChatStreamError('aborted')); return; }
    const abort = () => reject(new ChatStreamError('aborted'));
    signal.addEventListener('abort', abort, { once: true });
    promise.then(value => { signal.removeEventListener('abort', abort); if (signal.aborted) reject(new ChatStreamError('aborted')); else resolve(value); }, error => { signal.removeEventListener('abort', abort); reject(error); });
  });
}

export class GuestApi {
  private static async request(path: string, init: RequestInit = {}, signal?: AbortSignal, scope?: GuestScope): Promise<unknown> {
    let response: Response;
    try { response = await fetch(`${BASE}${path}`, { ...init, credentials: 'include', cache: 'no-store', signal, headers: HEADERS }); }
    catch { throw new ChatStreamError(signal?.aborted ? 'aborted' : 'network'); }
    if (signal?.aborted) { void response.body?.cancel().catch(() => {}); throw new ChatStreamError('aborted'); }
    if (!response.ok) { void response.body?.cancel().catch(() => {}); if (response.status === 401 && scope) expired(scope); throw new ChatStreamError('http', response.status); }
    if (response.status === 204) return null;
    try { const result: unknown = await response.json(); if (signal?.aborted) throw new ChatStreamError('aborted'); return result; }
    catch (error) { if (error instanceof ChatStreamError) throw error; return invalid(); }
  }
  static bootstrap(signal?: AbortSignal): Promise<GuestIdentity> {
    // Share only the in-flight allocation. One component's unmount must not cancel
    // another subscriber or retain a principal after cookies have changed.
    if (!bootstrapFlight) {
      const flight = this.request('/bootstrap', { method: 'POST', body: '{}' }).then(value => {
        if (!record(value) || !uuid(value.visitor_id) || typeof value.expires_at !== 'string' || !Number.isFinite(Date.parse(value.expires_at)) || !record(value.session) || !id(value.session.id) || !uuid(value.session.session_key)) return invalid();
        return { visitor_id: value.visitor_id, expires_at: value.expires_at, session: { id: value.session.id, session_key: value.session.session_key } };
      });
      bootstrapFlight = flight;
      void flight.then(() => { if (bootstrapFlight === flight) bootstrapFlight = null; }, () => { if (bootstrapFlight === flight) bootstrapFlight = null; });
    }
    return withSignal(bootstrapFlight, signal);
  }
  static async healthCheck(): Promise<{ status: string }> {
    let response: Response;
    try { response = await fetch(API_ENDPOINTS.HEALTH, { cache: 'no-store', credentials: 'omit' }); } catch { throw new ChatStreamError('network'); }
    if (!response.ok) throw new ChatStreamError('http', response.status);
    return { status: 'healthy' };
  }
  static async getChatHistory(scope: GuestScope, limit = 50, signal?: AbortSignal): Promise<GuestHistory> {
    if (!validScope(scope)) return invalid();
    const query = new URLSearchParams({ session_id: String(scope.session_id), expected_session_key: scope.session_key, limit: String(limit) });
    const value = await this.request(`/history?${query}`, {}, signal, scope);
    if (!record(value) || value.session_id !== scope.session_id || value.session_key !== scope.session_key || !Array.isArray(value.messages)) return invalid();
    const counts = new Map<number, number>();
    for (const row of value.messages) if (record(row) && id(row.message_id)) counts.set(row.message_id, (counts.get(row.message_id) ?? 0) + 1);
    // Shopping turns arrive as ordinary assistant text. Any legacy structured UI
    // payload on a history row is ignored, never parsed into purchase controls.
    const messages = value.messages.map((row): GuestMessage => {
      if (!record(row) || typeof row.role !== 'string' || !['user', 'assistant', 'system'].includes(row.role) || typeof row.content !== 'string') return invalid();
      const messageId = id(row.message_id) && counts.get(row.message_id) === 1 ? row.message_id : undefined;
      const status = typeof row.status === 'string' && ['pending', 'processing', 'completed', 'failed'].includes(row.status) ? row.status as GuestMessage['status'] : undefined;
      return { role: row.role as GuestMessage['role'], content: row.content, message_id: messageId, status, ...(typeof row.timestamp === 'string' ? { timestamp: row.timestamp } : {}), ...(messageId && (row.user_rating === 1 || row.user_rating === 0 || row.user_rating === -1) ? { user_rating: row.user_rating } : {}), citations: parseSourceCitations(row.citations) };
    });
    return { session_id: scope.session_id, session_key: scope.session_key, messages };
  }
  static async *streamChat(message: string, scope: GuestScope, onTrace?: (event: TraceEvent) => void, onCitations?: (items: SourceCitation[]) => void, signal?: AbortSignal, onCompletion?: (receipt: ChatCompletion) => void): AsyncGenerator<string> {
    if (!validScope(scope)) return invalid();
    try { yield* ApiClient.streamChat(message, scope.session_id, onTrace, onCitations, signal, onCompletion, { endpoint: `${BASE}/chat/stream`, headers: HEADERS, credentials: 'include', expectedSessionKey: scope.session_key }); }
    catch (error) { if (error instanceof ChatStreamError && error.status === 401) expired(scope); throw error; }
  }
  static async submitFeedback(messageId: number, rating: -1 | 0 | 1, scope: GuestScope, signal?: AbortSignal): Promise<void> {
    if (!id(messageId) || !validScope(scope) || ![-1, 0, 1].includes(rating)) return invalid();
    const value = await this.request('/feedback', { method: 'POST', body: JSON.stringify({ message_id: messageId, rating, expected_session_key: scope.session_key }) }, signal, scope);
    if (!record(value) || value.success !== true || value.message_id !== messageId || value.rating !== rating || value.session_id !== scope.session_id || value.session_key !== scope.session_key) return invalid();
  }
  static async clearHistory(scope: GuestScope, signal?: AbortSignal): Promise<void> {
    if (!validScope(scope)) return invalid();
    const query = new URLSearchParams({ session_id: String(scope.session_id), expected_session_key: scope.session_key });
    await this.request(`/history?${query}`, { method: 'DELETE' }, signal, scope);
  }
}
