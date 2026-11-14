/** Real React and guest transport. Only fetch is controlled; no real cookies or browser. */
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ReadableStream as NativeReadableStream } from 'node:stream/web';
import { TextDecoder, TextEncoder } from 'node:util';
import { LanguageProvider } from '@/lib/i18n/context';
import { ChatInterfaceUpdated } from '@/components/ChatInterfaceUpdated';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true, TextDecoder, TextEncoder });
const scope = { session_id: 317, session_key: '11111111-1111-4111-8111-111111111111' };
const visitor = { visitor_id: '22222222-2222-4222-8222-222222222222', expires_at: '2099-01-01T00:00:00Z', session: { id: scope.session_id, session_key: scope.session_key } };
const json = (value: unknown, status = 200): Response => ({ ok: status >= 200 && status < 300, status, json: async () => value } as Response);
let root: Root | undefined;
let container: HTMLDivElement;
let transport: jest.Mock;
let bootstrapStatus: number;
let currentVisitor: typeof visitor;
let historyRows: unknown[];
let override: ((url: URL, init: RequestInit) => Response | Promise<Response> | undefined) | undefined;
const originalFetch = globalThis.fetch;

/** Every purchase control the old card UI exposed; none may ever render again. */
const REMOVED_CONTROLS = ['Add simulated item', 'Review simulated checkout', 'Confirm simulated order', 'Simulate payment · no charge', 'Check purchase status', 'Continue simulated purchase', 'Reload conversation', 'Clear this conversation'];

beforeEach(() => {
  localStorage.clear(); sessionStorage.clear(); localStorage.setItem('portfolio-locale', 'en');
  localStorage.setItem('access_token', 'UNRELATED_LEGACY_TOKEN');
  bootstrapStatus = 200;
  currentVisitor = { ...visitor, session: { ...visitor.session } }; override = undefined;
  historyRows = [{ role: 'assistant', content: 'This visitor history', message_id: 51, status: 'completed' }];
  container = document.createElement('div'); document.body.appendChild(container);
  transport = jest.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), 'http://localhost');
    const custom = override?.(url, init); if (custom !== undefined) return custom;
    if (url.pathname === '/health') return json({ status: 'healthy' });
    if (url.pathname === '/api/v1/guest/bootstrap') return json(currentVisitor, bootstrapStatus);
    if (url.pathname === '/api/v1/guest/history') return json({ session_id: currentVisitor.session.id, session_key: currentVisitor.session.session_key, messages: historyRows });
    if (url.pathname === '/api/v1/chat/history') return json({ session_id: 1, messages: [{ role: 'assistant', content: 'SHARED OLD HISTORY' }] });
    throw new Error(`Unexpected controlled request: ${url.pathname}`);
  });
  globalThis.fetch = transport;
});
afterEach(async () => { await act(async () => root?.unmount()); container.remove(); globalThis.fetch = originalFetch; });
async function mount() { await act(async () => { root = createRoot(container); root.render(<LanguageProvider><ChatInterfaceUpdated /></LanguageProvider>); }); }
function calls(path: string) { return transport.mock.calls.filter(([input]) => new URL(String(input), 'http://localhost').pathname === path); }
function button(prefix: string, enabled = true): HTMLButtonElement {
  const value = Array.from(container.querySelectorAll('button')).find(row => row.textContent?.startsWith(prefix) && (!enabled || !row.disabled));
  if (!value) throw new Error(`Missing button ${prefix}: ${container.textContent}`);
  return value;
}
async function click(prefix: string) { await act(async () => button(prefix).click()); }
async function write(selector: string, value: string) {
  const input = container.querySelector<HTMLInputElement>(selector)!;
  await act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value); input.dispatchEvent(new Event('input', { bubbles: true })); });
}
async function send(text: string) { await write('input[aria-label="Message"]', text); await act(async () => container.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')!.click()); }
function commerceRequests() { return transport.mock.calls.filter(([input]) => String(input).includes('/commerce')); }
function removedControls() {
  return Array.from(container.querySelectorAll('button'))
    .map(row => row.textContent ?? '')
    .filter(text => REMOVED_CONTROLS.some(label => text.startsWith(label)));
}
function sseData(content: string): string {
  // The transport parser joins consecutive `data:` lines with newlines, so
  // multi-line reply text must be encoded as one data line per source line.
  return content.split('\n').map(line => `data: ${line}`).join('\n');
}
function streamReply(messageId: number, content: string): Response {
  const receipt = { message_id: messageId, status: 'completed', session_id: currentVisitor.session.id, session_key: currentVisitor.session.session_key };
  const bytes = new TextEncoder().encode(`${sseData(content)}\n\nevent: completion\ndata: ${JSON.stringify(receipt)}\n\ndata: [DONE]\n\n`);
  return { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body: new NativeReadableStream({ start(controller) { controller.enqueue(bytes); controller.close(); } }) } as unknown as Response;
}
function streamText(content: string): Response {
  const bytes = new TextEncoder().encode(`${sseData(content)}\n\ndata: [DONE]\n\n`);
  return { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body: new NativeReadableStream({ start(controller) { controller.enqueue(bytes); controller.close(); } }) } as unknown as Response;
}

test('public chat bootstraps a cookie visitor and reads only its server-issued session', async () => {
  await mount();
  expect(calls('/api/v1/guest/bootstrap')).toHaveLength(1);
  const [, options] = calls('/api/v1/guest/bootstrap')[0];
  expect(options.credentials).toBe('include');
  expect(options.headers['X-Rag-Guest']).toBe('1');
  expect(options.headers.Authorization).toBeUndefined();
  const [url] = calls('/api/v1/guest/history')[0];
  expect(new URL(String(url), 'http://localhost').searchParams.get('session_id')).toBe('317');
  expect(new URL(String(url), 'http://localhost').searchParams.get('expected_session_key')).toBe(scope.session_key);
  expect(container.textContent).toContain('This visitor history');
  expect(container.textContent).not.toContain('SHARED OLD HISTORY');
  expect(calls('/api/v1/chat/history')).toHaveLength(0);
});

test('bootstrap failure never falls back to shared session1 or enables a send', async () => {
  bootstrapStatus = 503; await mount();
  expect(calls('/api/v1/chat/history')).toHaveLength(0);
  expect(container.textContent).not.toContain('SHARED OLD HISTORY');
  expect(container.querySelector<HTMLButtonElement>('button[aria-label="Send message"]')?.disabled).toBe(true);
});

test('concurrent StrictMode chat mounts share only one in-flight visitor allocation', async () => {
  let finish!: (value: Response) => void;
  override = url => url.pathname.endsWith('/bootstrap') ? new Promise(resolve => { finish = resolve; }) : undefined;
  await act(async () => { root = createRoot(container); root.render(<StrictMode><LanguageProvider><ChatInterfaceUpdated /><ChatInterfaceUpdated /></LanguageProvider></StrictMode>); });
  expect(calls('/api/v1/guest/bootstrap')).toHaveLength(1);
  await act(async () => finish(json(visitor)));
  expect(calls('/api/v1/guest/history')).toHaveLength(2);
  expect(container.textContent?.match(/This visitor history/g)).toHaveLength(2);
});

test('expired guest feedback clears private history and restarting never resends the old vote', async () => {
  override = url => url.pathname.endsWith('/feedback') ? json({ detail: 'SECRET COOKIE DETAIL' }, 401) : undefined;
  await mount();
  await act(async () => container.querySelector<HTMLButtonElement>('button[aria-label="Helpful"]')!.click());
  expect(container.textContent).not.toContain('This visitor history');
  expect(container.textContent).toContain('visitor session expired');
  expect(container.textContent).not.toContain('SECRET COOKIE DETAIL');
  expect(calls('/api/v1/guest/bootstrap')).toHaveLength(1);
  currentVisitor = { ...visitor, visitor_id: '55555555-5555-4555-8555-555555555555', session: { id: 318, session_key: '44444444-4444-4444-8444-444444444444' } };
  historyRows = [];
  await click('Start a new visitor session');
  expect(calls('/api/v1/guest/feedback')).toHaveLength(1);
  expect(calls('/api/v1/guest/bootstrap')).toHaveLength(2);
  expect(container.textContent).not.toContain('This visitor history');
});

// The whole purchase flow is agent-orchestrated chat text: catalog, cart,
// quote, order and simulated payment all arrive as ordinary assistant replies.
// The only structured UI is the server-owned approval card, and it renders
// only when a completion receipt carries validated approval data — this
// journey's receipts carry none, so it stays pure text.
const JOURNEY: Array<[string, string]> = [
  ['想买降噪耳机', '模拟商品目录（演示模式 · 仅模拟交易，不会真实扣款）：\n模拟降噪耳机 · 黑色 · ¥100.00 · 可选数量 10'],
  ['来两件', '模拟购物车：\n模拟降噪耳机 · 黑色 × 2 · 小计 ¥200.00'],
  ['结算', '模拟报价（有效期至 2099-01-01）：\n商品小计 ¥200.00 · 配送 ¥5.00\n请核对本次模拟总价：¥205.00'],
  ['确认下单', '模拟订单已创建：mock-order-one · 待付款 · ¥205.00'],
  ['付款', '模拟付款成功（无真实扣款）：mock-order-one · ¥205.00'],
];

test('a visitor completes a simulated purchase entirely through chat text with no purchase controls', async () => {
  historyRows = []; let step = 0; let nextId = 102;
  override = (url, init) => {
    if (!url.pathname.endsWith('/chat/stream')) return undefined;
    expect(JSON.parse(String(init.body))).toMatchObject({ session_id: 317, expected_session_key: currentVisitor.session.session_key });
    return streamReply(nextId++, JOURNEY[step++][1]);
  };
  await mount();
  for (const [question, reply] of JOURNEY) {
    await send(question);
    expect(container.textContent).toContain(reply);
  }
  expect(calls('/api/v1/guest/chat/stream')).toHaveLength(5);
  // Each confirmed turn is rateable; the text is all the UI offers.
  expect(container.querySelectorAll('[role="group"][aria-label="Rate this reply"]')).toHaveLength(5);
  const catalogBubble = Array.from(container.querySelectorAll('p.whitespace-pre-wrap')).find(row => row.textContent?.includes('模拟商品目录'));
  expect(catalogBubble?.textContent).toContain('\n模拟降噪耳机');
  expect(removedControls()).toEqual([]);
  expect(container.querySelector('input[aria-label="Simulated item quantity"]')).toBeNull();
  // No structured approval payloads arrived, so no confirmation card either:
  // the new card contract is opt-in per server-provided approval data.
  expect(container.querySelector('[data-approval-card]')).toBeNull();
  expect(container.querySelector('[data-approval-section]')).toBeNull();
  expect(commerceRequests()).toHaveLength(0);
  // Only the mount-time history read happens; no post-reply UI re-read.
  expect(calls('/api/v1/guest/history')).toHaveLength(1);
});

test('restored purchase history renders as plain chat text and never issues commerce requests', async () => {
  historyRows = [
    { role: 'assistant', content: '模拟付款成功（无真实扣款）：mock-order-one · ¥205.00', message_id: 51, status: 'completed', ui: { schema_version: 1, purchase_id: 'purchase-one', actions: [{ kind: 'select_sku' }] } },
  ];
  await mount();
  expect(container.textContent).toContain('模拟付款成功（无真实扣款）');
  expect(container.querySelector('input[aria-label="Simulated item quantity"]')).toBeNull();
  expect(removedControls()).toEqual([]);
  expect(commerceRequests()).toHaveLength(0);
  expect(calls('/api/v1/guest/history')).toHaveLength(1);
});

test('a reply without a server completion stays unconfirmed instead of claiming receipt', async () => {
  historyRows = [];
  override = url => url.pathname.endsWith('/chat/stream') ? streamText('模拟订单已创建：mock-order-two · 待付款 · ¥205.00') : undefined;
  await mount(); await send('再买一个');
  expect(container.textContent).toContain('mock-order-two');
  expect(container.textContent).not.toContain('Reply received');
  expect(container.textContent).toContain('did not confirm saving it');
  // No confirmed message id means no feedback offer and no fabricated controls.
  expect(container.querySelectorAll('[role="group"][aria-label="Rate this reply"]')).toHaveLength(0);
  expect(removedControls()).toEqual([]);
  expect(commerceRequests()).toHaveLength(0);
  // The conversation stays usable for a follow-up verification question.
  override = url => url.pathname.endsWith('/chat/stream') ? streamReply(120, '已核实：模拟订单 mock-order-two 待付款。') : undefined;
  await send('查一下状态');
  expect(container.textContent).toContain('已核实：模拟订单 mock-order-two 待付款。');
});
