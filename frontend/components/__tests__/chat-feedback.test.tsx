/** Actual React + ApiClient + native byte streams; only fetch is controlled. */
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ReadableStream as NativeReadableStream } from 'node:stream/web';
import { TextDecoder, TextEncoder } from 'node:util';
import { ChatInterfaceUpdated } from '@/components/ChatInterfaceUpdated';
import { LanguageProvider } from '@/lib/i18n/context';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true, TextDecoder, TextEncoder });

function json(value: unknown, status = 200): Response {
  return { ok: status >= 200 && status < 300, status, statusText: 'PRIVATE RESPONSE DETAIL', json: async () => value } as Response;
}

function stream() {
  let controller: ReadableStreamDefaultController<Uint8Array>;
  let closed = false;
  const body = new NativeReadableStream<Uint8Array>({ start(value) { controller = value; }, cancel() { closed = true; } });
  return {
    response: { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body } as unknown as Response,
    send(text: string) { if (!closed) controller.enqueue(new TextEncoder().encode(text)); },
    close() { if (!closed) { closed = true; controller.close(); } },
    abort() { if (!closed) { closed = true; controller.error(new DOMException('PRIVATE ABORT DETAIL', 'AbortError')); } },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(complete => { resolve = complete; });
  return { promise, resolve };
}

let root: Root | undefined;
let container: HTMLDivElement;
let history: unknown[];
let fetchMock: jest.Mock;
let chatReply: (init?: RequestInit) => Response;
let feedbackReply: (init?: RequestInit) => Response | Promise<Response>;
let streams: ReturnType<typeof stream>[];
const originalFetch = globalThis.fetch;

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem('portfolio-locale', 'en');
  localStorage.setItem('access_token', 'isolated-feedback-token');
  history = [];
  streams = [];
  container = document.createElement('div');
  document.body.appendChild(container);
  chatReply = () => { throw new Error('Chat transport not configured'); };
  feedbackReply = init => json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', ...JSON.parse(String(init?.body)) });
  fetchMock = jest.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(url), 'http://localhost').pathname;
    if (path === '/health') return json({ status: 'healthy' });
    if (path === '/api/v1/guest/bootstrap') return json({ visitor_id: '22222222-2222-4222-8222-222222222222', expires_at: '2099-01-01T00:00:00Z', session: { id: 317, session_key: '11111111-1111-4111-8111-111111111111' } });
    if (path === '/api/v1/guest/history') return json({ session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', messages: history });
    if (path === '/api/v1/guest/chat/stream') return chatReply(init);
    if (path === '/api/v1/guest/feedback') return feedbackReply(init);
    throw new Error(`Unexpected request: ${path}`);
  });
  globalThis.fetch = fetchMock;
});

afterEach(async () => {
  await act(async () => { root?.unmount(); root = undefined; });
  await act(async () => streams.forEach(value => value.close()));
  container.remove();
  globalThis.fetch = originalFetch;
});

async function mount() {
  await act(async () => {
    root = createRoot(container);
    root.render(<LanguageProvider><ChatInterfaceUpdated /></LanguageProvider>);
  });
}

function newStream() {
  const value = stream();
  streams.push(value);
  chatReply = init => {
    init?.signal?.addEventListener('abort', () => value.abort(), { once: true });
    return value.response;
  };
  return value;
}

async function send(question = 'Question') {
  const input = container.querySelector<HTMLInputElement>('input[type="text"]')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(input, question);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await act(async () => container.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')?.click());
}

function receipt(messageId: unknown, sessionId = 317, status = 'completed') {
  return 'event: completion\ndata: '+JSON.stringify({ message_id: messageId, session_id: sessionId, session_key: '11111111-1111-4111-8111-111111111111', status })+'\n\n';
}

async function complete(value: ReturnType<typeof stream>, messageId: number, answer = 'Same answer') {
  await act(async () => { value.send('data: '+answer+'\n\n'+receipt(messageId)+'data: [DONE]\n\n'); value.close(); });
}

function groups() {
  return Array.from(container.querySelectorAll<HTMLElement>('[role="group"][aria-label="Rate this reply"]'));
}

function vote(group: HTMLElement, rating: 1 | -1) {
  const button = group.querySelector<HTMLButtonElement>(`button[aria-label="${rating === 1 ? 'Helpful' : 'Not helpful'}"]`);
  if (!button) throw new Error('Feedback control missing');
  return button;
}

function feedbackCalls() {
  return fetchMock.mock.calls.filter(([url]) => new URL(String(url), 'http://localhost').pathname === '/api/v1/guest/feedback');
}

function assistant(messageId: unknown, status = 'completed', userRating = 0, content = 'Historical answer') {
  return { role: 'assistant', content, message_id: messageId, status, user_rating: userRating };
}

test('a live reply needs DONE and its real receipt; a vote changes only after the matching server acknowledgement', async () => {
  const pending = deferred<Response>();
  feedbackReply = () => pending.promise;
  const value = newStream();
  await mount();
  await send();
  await act(async () => value.send('data: Real partial\n\n'+receipt(501)));
  expect(groups()).toHaveLength(0);
  await act(async () => { value.send('data: [DONE]\n\n'); value.close(); });
  expect(groups()).toHaveLength(1);
  const group = groups()[0];
  await act(async () => vote(group, 1).click());
  const [url, init] = feedbackCalls()[0];
  expect(String(url)).toContain('/api/v1/guest/feedback');
  expect(JSON.parse(String(init.body))).toEqual({ message_id: 501, rating: 1, expected_session_key: '11111111-1111-4111-8111-111111111111' });
  expect(init.headers.Authorization).toBeUndefined(); expect(init.credentials).toBe('include');
  expect(vote(group, 1).getAttribute('aria-pressed')).toBe('false');
  expect(vote(group, 1).disabled).toBe(true);
  await act(async () => pending.resolve(json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', message_id: 501, rating: 1 })));
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('true');
  feedbackReply = init => json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', ...JSON.parse(String(init?.body)) });
  await act(async () => vote(groups()[0], -1).click());
  expect(feedbackCalls()).toHaveLength(2);
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('false');
  expect(vote(groups()[0], -1).getAttribute('aria-pressed')).toBe('true');
});

test('identical answer text from separate turns retains separate server identities', async () => {
  const first = newStream();
  await mount(); await send('First question'); await complete(first, 701);
  const second = newStream();
  await send('Second question'); await complete(second, 702);
  expect(groups()).toHaveLength(2);
  await act(async () => vote(groups()[1], -1).click());
  expect(JSON.parse(String(feedbackCalls()[0][1].body))).toEqual({ message_id: 702, rating: -1, expected_session_key: '11111111-1111-4111-8111-111111111111' });
  expect(vote(groups()[0], -1).getAttribute('aria-pressed')).toBe('false');
  expect(vote(groups()[1], -1).getAttribute('aria-pressed')).toBe('true');
});

test.each([
  ['no receipt', '', 'data: [DONE]\n\n'],
  ['wrong session', receipt(10, 2), 'data: [DONE]\n\n'],
  ['boolean identity', receipt(true), 'data: [DONE]\n\n'],
  ['conflicting receipts', receipt(10)+receipt(11), 'data: [DONE]\n\n'],
  ['EOF after a valid receipt', receipt(10), ''],
  ['server failure after a valid receipt', receipt(10), 'data: [STREAM_ERROR]\n\ndata: [DONE]\n\n'],
])('%s cannot authorize feedback', async (_name, metadata, ending) => {
  const value = newStream();
  await mount(); await send();
  await act(async () => { value.send('data: Preserved answer\n\n'+metadata+ending); value.close(); });
  expect(container.textContent).toContain('Preserved answer');
  expect(groups()).toHaveLength(0);
  expect(feedbackCalls()).toHaveLength(0);
});

test('stopping after receipt metadata keeps the partial response ineligible', async () => {
  const value = newStream();
  await mount(); await send();
  await act(async () => value.send('data: Partial before stopping\n\n'+receipt(42)));
  await act(async () => container.querySelector<HTMLButtonElement>('button[aria-label="Stop receiving"]')?.click());
  expect(container.textContent).toContain('Partial before stopping');
  expect(container.querySelector('[role="alert"]')?.textContent).toContain('Stopped receiving');
  expect(groups()).toHaveLength(0);
});

test('an empty completed body is not rateable even with a valid receipt', async () => {
  const value = newStream();
  await mount(); await send();
  await act(async () => { value.send(receipt(42)+'data: [DONE]\n\n'); value.close(); });
  expect(groups()).toHaveLength(0);
  expect(container.querySelector('[role="alert"]')).not.toBeNull();
});

test('history restores a recorded vote only for a completed assistant with a real identity', async () => {
  history = [assistant(101, 'completed', 1), assistant(102, 'failed', -1), assistant(103, 'pending'), assistant(104, 'processing'), assistant(undefined), assistant(105, 'completed', 0), { role: 'user', content: 'User question', message_id: 106, status: 'completed', user_rating: 1 }];
  await mount();
  expect(groups()).toHaveLength(2);
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('true');
  expect(vote(groups()[1], 1).getAttribute('aria-pressed')).toBe('false');
  await act(async () => vote(groups()[0], -1).click());
  expect(JSON.parse(String(feedbackCalls()[0][1].body))).toEqual({ message_id: 101, rating: -1, expected_session_key: '11111111-1111-4111-8111-111111111111' });
});

test('duplicate or malformed history IDs do not produce ambiguous feedback targets', async () => {
  history = [assistant(77), { role: 'user', content: 'Conflicting user', message_id: 77, status: 'completed' }, assistant(true), assistant('88'), assistant(-1)];
  await mount();
  expect(groups()).toHaveLength(0);
});

test.each(['en', 'zh'])('cookie visitors in %s can rate completed replies without an account token', async (locale) => {
  localStorage.removeItem('access_token');
  localStorage.setItem('portfolio-locale', locale);
  history = [assistant(91), assistant(undefined), assistant(92, 'failed')];
  await mount();
  expect(container.querySelectorAll('[role="group"]')).toHaveLength(1);
  expect(container.textContent).not.toContain(locale === 'en' ? 'Sign in to rate this reply' : '登录后可评价');
  expect(feedbackCalls()).toHaveLength(0);
  expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/auth/'))).toBe(false);
});

test('synchronous opposing clicks on one message issue a single pending vote', async () => {
  history = [assistant(111)];
  const pending = deferred<Response>();
  feedbackReply = () => pending.promise;
  await mount();
  const group = groups()[0];
  await act(async () => { vote(group, 1).click(); vote(group, -1).click(); });
  expect(feedbackCalls()).toHaveLength(1);
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('false');
  await act(async () => pending.resolve(json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', message_id: 111, rating: 1 })));
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('true');
});

test('different messages can submit independently and out-of-order acknowledgements cannot cross votes', async () => {
  history = [assistant(201), assistant(202)];
  const first = deferred<Response>(); const second = deferred<Response>();
  feedbackReply = init => JSON.parse(String(init?.body)).message_id === 201 ? first.promise : second.promise;
  await mount();
  await act(async () => vote(groups()[0], 1).click());
  expect(vote(groups()[1], -1).disabled).toBe(false);
  await act(async () => vote(groups()[1], -1).click());
  expect(feedbackCalls()).toHaveLength(2);
  await act(async () => second.resolve(json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', message_id: 202, rating: -1 })));
  expect(vote(groups()[1], -1).getAttribute('aria-pressed')).toBe('true');
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('false');
  await act(async () => first.resolve(json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', message_id: 201, rating: 1 })));
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('true');
});

test.each([403, 404, 429, 503, 'network', 'wrong-id', 'wrong-rating'] as const)('feedback failure %s preserves the recorded vote and allows an explicit retry', async (failure) => {
  history = [assistant(301, 'completed', -1)];
  feedbackReply = () => {
    if (failure === 'network') throw new Error('PRIVATE NETWORK DETAIL');
    if (failure === 'wrong-id') return json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', message_id: 999, rating: 1 });
    if (failure === 'wrong-rating') return json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', message_id: 301, rating: -1 });
    return json({ detail: 'PRIVATE RESPONSE DETAIL' }, failure);
  };
  await mount();
  await act(async () => vote(groups()[0], 1).click());
  expect(feedbackCalls()).toHaveLength(1);
  expect(container.querySelector('[role="alert"]')?.textContent).toBeTruthy();
  expect(container.textContent).not.toContain('PRIVATE');
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('false');
  expect(vote(groups()[0], -1).getAttribute('aria-pressed')).toBe('true');
  expect(vote(groups()[0], 1).disabled).toBe(false);
  feedbackReply = () => json({ success: true, session_id: 317, session_key: '11111111-1111-4111-8111-111111111111', message_id: 301, rating: 1 });
  await act(async () => vote(groups()[0], 1).click());
  expect(feedbackCalls()).toHaveLength(2);
  expect(vote(groups()[0], 1).getAttribute('aria-pressed')).toBe('true');
});

test('HTTP401 clears the visitor history and requires an explicit restart without leaking details', async () => {
  history = [assistant(401)];
  feedbackReply = () => json({ detail: 'PRIVATE RESPONSE DETAIL' }, 401);
  await mount();
  await act(async () => vote(groups()[0], 1).click());
  expect(feedbackCalls()).toHaveLength(1);
  expect(container.querySelector('[role="alert"]')?.textContent).toMatch(/visitor session expired/i);
  expect(container.textContent).not.toContain('PRIVATE');
  expect(groups()).toHaveLength(0);
});
