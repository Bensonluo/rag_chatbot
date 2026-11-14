/** Price-watch ask cards: server-owned prices, ask-only, chat-channel buttons.

Real React and guest transport (only fetch is controlled). The card must
render exactly what the server's structured watch payload says — when the
model's reply text disagrees, the card wins — and its order button must
reuse the ordinary chat send channel instead of any new endpoint. Dismissal
is local-only: the server already recorded the ask, so no request leaves.
 */
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ReadableStream as NativeReadableStream } from 'node:stream/web';
import { TextDecoder, TextEncoder } from 'node:util';
import { LanguageProvider } from '@/lib/i18n/context';
import { ChatInterfaceUpdated } from '@/components/ChatInterfaceUpdated';
import { WatchCard } from '@/components/WatchCard';
import type { CommerceWatch } from '@/lib/api';

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true, TextDecoder, TextEncoder });
const scope = { session_id: 317, session_key: '11111111-1111-4111-8111-111111111111' };
const visitor = { visitor_id: '22222222-2222-4222-8222-222222222222', expires_at: '2099-01-01T00:00:00Z', session: { id: scope.session_id, session_key: scope.session_key } };
const json = (value: unknown, status = 200): Response => ({ ok: status >= 200 && status < 300, status, json: async () => value } as Response);
let root: Root | undefined;
let container: HTMLDivElement;
let transport: jest.Mock;
let historyRows: unknown[];
let override: ((url: URL, init: RequestInit) => Response | Promise<Response> | undefined) | undefined;
const originalFetch = globalThis.fetch;

const priceWatch = (over: Record<string, unknown> = {}): CommerceWatch => ({
  watch_id: 'watch-one',
  sku_id: 'ANC-02',
  sku_name: '轻音降噪耳机 · 白色',
  target_price_minor: 40000,
  current_price_minor: 30400,
  currency: 'CNY',
  condition: 'at_or_below',
  action: 'ask_order',
  action_label: '现在下单',
  triggered_at: '2026-11-11T08:00:00Z',
  expires_at: '2026-12-11T08:00:00Z',
  ...over,
} as CommerceWatch);

const commerceUi = (watch: CommerceWatch | null) => ({
  schema_version: 1,
  purchase_id: 'purchase-one',
  version: 4,
  mode: 'mock',
  stage: 'shopping',
  snapshot_at: '2026-11-11T00:00:00Z',
  blocks: [],
  watch,
});

beforeEach(() => {
  localStorage.clear(); sessionStorage.clear(); localStorage.setItem('portfolio-locale', 'zh');
  historyRows = [{ role: 'assistant', content: '历史记录', message_id: 51, status: 'completed' }];
  container = document.createElement('div'); document.body.appendChild(container);
  override = undefined;
  transport = jest.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), 'http://localhost');
    const custom = override?.(url, init); if (custom !== undefined) return custom;
    if (url.pathname === '/health') return json({ status: 'healthy' });
    if (url.pathname === '/api/v1/guest/bootstrap') return json(visitor);
    if (url.pathname === '/api/v1/guest/history') return json({ session_id: scope.session_id, session_key: scope.session_key, messages: historyRows });
    throw new Error(`Unexpected controlled request: ${url.pathname}`);
  });
  globalThis.fetch = transport;
});
afterEach(async () => { await act(async () => root?.unmount()); container.remove(); globalThis.fetch = originalFetch; });
async function mount() { await act(async () => { root = createRoot(container); root.render(<StrictMode><LanguageProvider><ChatInterfaceUpdated /></LanguageProvider></StrictMode>); }); }
const calls = (path: string) => transport.mock.calls.filter(([input]) => new URL(String(input), 'http://localhost').pathname === path);
async function send(text: string) {
  const input = container.querySelector<HTMLInputElement>('input[aria-label="消息"]')!;
  await act(async () => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, text); input.dispatchEvent(new Event('input', { bubbles: true })); });
  await act(async () => container.querySelector<HTMLButtonElement>('button[aria-label="发送消息"]')!.click());
}
function streamReply(messageId: number, content: string, watch: CommerceWatch | null): Response {
  const receipt = { message_id: messageId, status: 'completed', session_id: scope.session_id, session_key: scope.session_key, ui: commerceUi(watch) };
  const body = content.split('\n').map(line => `data: ${line}`).join('\n');
  const bytes = new TextEncoder().encode(`${body}\n\nevent: completion\ndata: ${JSON.stringify(receipt)}\n\ndata: [DONE]\n\n`);
  return { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body: new NativeReadableStream({ start(controller) { controller.enqueue(bytes); controller.close(); } }) } as unknown as Response;
}
const card = () => container.querySelector<HTMLElement>('[data-watch-card]');
const orderButton = () => container.querySelector<HTMLButtonElement>('[data-watch-order]')!;
const dismissButton = () => container.querySelector<HTMLButtonElement>('[data-watch-dismiss]')!;

// ── Component contract ─────────────────────────────────────────────────────

test('the card renders the server-owned target, live price and savings', () => {
  act(() => {
    root = createRoot(container);
    root.render(
      <WatchCard watch={priceWatch()} status="active" busy={false} isEn={false}
        onOrder={() => {}} onDismiss={() => {}} />
    );
  });
  expect(card()?.textContent).toContain('轻音降噪耳机 · 白色');
  expect(card()?.textContent).toContain('ANC-02');
  expect(container.querySelector('[data-watch-target]')?.textContent).toBe('¥400.00');
  expect(container.querySelector('[data-watch-current]')?.textContent).toBe('¥304.00');
  expect(container.querySelector('[data-watch-saved]')?.textContent).toContain('¥96.00');
  expect(orderButton().textContent).toBe('现在下单');
  expect(dismissButton().disabled).toBe(false);
  expect(card()?.textContent).toContain('以本卡为准');
});

test('a dismissed or ordered card disables both buttons', () => {
  act(() => {
    root = createRoot(container);
    root.render(
      <StrictMode>
        <WatchCard watch={priceWatch()} status="dismissed" busy={false} isEn={false} onOrder={() => {}} onDismiss={() => {}} />
        <WatchCard watch={priceWatch({ watch_id: 'watch-two' })} status="ordered" busy={false} isEn={false} onOrder={() => {}} onDismiss={() => {}} />
      </StrictMode>
    );
  });
  const dismissedCard = container.querySelectorAll('[data-watch-card]')[0] as HTMLElement;
  const orderedCard = container.querySelectorAll('[data-watch-card]')[1] as HTMLElement;
  expect(dismissedCard.getAttribute('data-watch-status')).toBe('dismissed');
  expect(dismissedCard.querySelector<HTMLButtonElement>('[data-watch-order]')!.disabled).toBe(true);
  expect(dismissedCard.textContent).toContain('已忽略');
  expect(orderedCard.getAttribute('data-watch-status')).toBe('ordered');
  expect(orderedCard.textContent).toContain('已请求下单');
});

// ── Chat integration ───────────────────────────────────────────────────────

test('a completion with a watch renders one card whose prices follow the card, not the reply text', async () => {
  historyRows = []; let nextId = 102;
  override = url => url.pathname.endsWith('/chat/stream')
    ? streamReply(nextId++, '模型口头说现在 ¥199.00，很划算。', priceWatch())
    : undefined;
  await mount();
  await send('看看降价');
  expect(card()).not.toBeNull();
  expect(container.querySelector('[data-watch-current]')?.textContent).toBe('¥304.00');
  expect(container.textContent).toContain('以本卡为准');
  // The same watch delivered again (defensive re-delivery) adds no card.
  await send('还有吗');
  expect(container.querySelectorAll('[data-watch-card]')).toHaveLength(1);
});

test('the order button sends a card-derived phrase through the chat stream channel', async () => {
  historyRows = []; let nextId = 102;
  override = url => url.pathname.endsWith('/chat/stream')
    ? streamReply(nextId++, '已收到。', priceWatch())
    : undefined;
  await mount();
  await send('看看降价');
  await act(async () => orderButton().click());
  const sent = calls('/api/v1/guest/chat/stream');
  expect(sent).toHaveLength(2);
  const message = JSON.parse(String(sent[1][1]!.body)).message;
  expect(message).toContain('ANC-02');
  expect(message).toContain('¥304.00');
  expect(message).toContain('报价确认');
  expect(sent[1][1]!.credentials).toBe('include');
  // No commerce endpoint exists for the card: everything rides the chat stream.
  expect(calls('/api/v1/guest/chat')).toHaveLength(0);
  expect(card()?.getAttribute('data-watch-status')).toBe('ordered');
});

test('dismiss is local-only: no request leaves and the card greys out', async () => {
  historyRows = []; let nextId = 102;
  override = url => url.pathname.endsWith('/chat/stream')
    ? streamReply(nextId++, '已收到。', priceWatch())
    : undefined;
  await mount();
  await send('看看降价');
  const before = transport.mock.calls.length;
  await act(async () => dismissButton().click());
  expect(transport.mock.calls.length).toBe(before);
  expect(card()?.getAttribute('data-watch-status')).toBe('dismissed');
  expect(card()?.textContent).toContain('已忽略');
});

test('a malformed watch payload never becomes a card', async () => {
  historyRows = []; let nextId = 102;
  const bad = { ...priceWatch(), current_price_minor: 'cheap' } as unknown;
  override = url => url.pathname.endsWith('/chat/stream')
    ? streamReply(nextId++, '请看降价提醒。', bad as CommerceWatch)
    : undefined;
  await mount();
  await send('看看降价');
  expect(card()).toBeNull();
  expect(container.querySelector('[data-watch-section]')).toBeNull();
});

test('completions without watch data render no card section', async () => {
  historyRows = []; let nextId = 102;
  override = url => url.pathname.endsWith('/chat/stream')
    ? streamReply(nextId++, '这里是普通回复。', null)
    : undefined;
  await mount();
  await send('你好');
  expect(container.querySelector('[data-watch-section]')).toBeNull();
});
