/** Structured approval cards: server-owned amounts, chat-channel buttons.

Real React and guest transport (only fetch is controlled). The card must
render exactly what the server's structured approval says — when the
model's reply text disagrees, the card wins — and its buttons must reuse
the ordinary chat send channel instead of any new endpoint.
 */
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ReadableStream as NativeReadableStream } from 'node:stream/web';
import { TextDecoder, TextEncoder } from 'node:util';
import { LanguageProvider } from '@/lib/i18n/context';
import { ChatInterfaceUpdated } from '@/components/ChatInterfaceUpdated';
import { ApprovalCard } from '@/components/ApprovalCard';
import type { CommerceApproval } from '@/lib/api';

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

const orderApproval = (over: Record<string, unknown> = {}): CommerceApproval => ({
  purchase_id: 'purchase-one',
  stage: 'quoted',
  action: 'confirm_order',
  action_label: '确认下单',
  lines: [
    { sku_id: 'ANC-02', name: '轻音降噪耳机 · 白色', quantity: 2, unit_price_minor: 30400, subtotal_minor: 60800 },
  ],
  total_minor: 60800,
  currency: 'CNY',
  quote_hash: 'a'.repeat(64),
  expires_at: '2099-01-01T00:00:00Z',
  purchase_version: 4,
  generation: 1,
  ...over,
} as CommerceApproval);

const payApproval = (over: Record<string, unknown> = {}): CommerceApproval => orderApproval({
  stage: 'awaiting_payment',
  action: 'confirm_payment',
  action_label: '确认支付',
  ...over,
});

const commerceUi = (approval: CommerceApproval | null) => ({
  schema_version: 1,
  purchase_id: approval?.purchase_id ?? 'purchase-one',
  version: 4,
  mode: 'mock',
  stage: approval?.stage ?? 'quoted',
  snapshot_at: '2026-11-11T00:00:00Z',
  blocks: [],
  approval,
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
function streamReply(messageId: number, content: string, approval: CommerceApproval | null): Response {
  const receipt = { message_id: messageId, status: 'completed', session_id: scope.session_id, session_key: scope.session_key, ui: commerceUi(approval) };
  const body = content.split('\n').map(line => `data: ${line}`).join('\n');
  const bytes = new TextEncoder().encode(`${body}\n\nevent: completion\ndata: ${JSON.stringify(receipt)}\n\ndata: [DONE]\n\n`);
  return { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body: new NativeReadableStream({ start(controller) { controller.enqueue(bytes); controller.close(); } }) } as unknown as Response;
}
const card = () => container.querySelector<HTMLElement>('[data-approval-card]');
const confirmButton = () => container.querySelector<HTMLButtonElement>('[data-approval-confirm]')!;
const cancelButton = () => container.querySelector<HTMLButtonElement>('[data-approval-cancel]')!;

// ── Component contract ─────────────────────────────────────────────────────

test('the card renders server-owned line items, total and countdown', () => {
  const now = Date.parse('2026-11-11T00:00:00Z');
  const approval = orderApproval({ expires_at: '2026-11-11T00:04:00Z' });
  act(() => {
    root = createRoot(container);
    root.render(
      <ApprovalCard approval={approval} status="active" busy={false} now={now} isEn={false}
        onConfirm={() => {}} onCancel={() => {}} />
    );
  });
  expect(card()?.textContent).toContain('轻音降噪耳机 · 白色');
  expect(container.querySelector('[data-approval-line="ANC-02"]')?.textContent).toContain('¥304.00');
  expect(container.querySelector('[data-approval-total]')?.textContent).toBe('¥608.00');
  expect(container.querySelector('[data-approval-meta]')?.textContent).toContain('04:00');
  expect(confirmButton().textContent).toBe('确认下单');
  expect(cancelButton().disabled).toBe(false);
});

test('a payment card shows no quote countdown and its label is 确认支付', () => {
  const approval = payApproval({ expires_at: '2020-01-01T00:00:00Z' });
  act(() => {
    root = createRoot(container);
    root.render(
      <ApprovalCard approval={approval} status="active" busy={false} now={Date.now()} isEn={false}
        onConfirm={() => {}} onCancel={() => {}} />
    );
  });
  // The order already exists: an expired quote must not disable payment.
  expect(container.querySelector('[data-approval-meta]')?.textContent).toContain('订单已创建');
  expect(confirmButton().textContent).toBe('确认支付');
  expect(confirmButton().disabled).toBe(false);
});

test('an expired or completed order card disables both buttons', () => {
  const now = Date.parse('2026-11-11T00:00:00Z');
  act(() => {
    root = createRoot(container);
    root.render(
      <StrictMode>
        <ApprovalCard approval={orderApproval({ expires_at: '2026-11-11T00:04:00Z' })} status="active" busy={false} now={now + 5 * 60 * 1000} isEn={false} onConfirm={() => {}} onCancel={() => {}} />
        <ApprovalCard approval={orderApproval()} status="done" busy={false} now={now} isEn={false} onConfirm={() => {}} onCancel={() => {}} />
      </StrictMode>
    );
  });
  const expiredCard = container.querySelectorAll('[data-approval-card]')[0] as HTMLElement;
  const doneCard = container.querySelectorAll('[data-approval-card]')[1] as HTMLElement;
  expect(expiredCard.getAttribute('data-approval-status')).toBe('expired');
  expect(expiredCard.querySelector<HTMLButtonElement>('[data-approval-confirm]')!.disabled).toBe(true);
  expect(expiredCard.textContent).toContain('报价已过期');
  expect(doneCard.getAttribute('data-approval-status')).toBe('done');
  expect(doneCard.querySelector<HTMLButtonElement>('[data-approval-confirm]')!.disabled).toBe(true);
  expect(doneCard.textContent).toContain('已完成');
});

// ── Chat integration ───────────────────────────────────────────────────────

test('a completion with approval renders one card whose amounts follow the card, not the reply text', async () => {
  historyRows = []; let nextId = 102;
  override = url => url.pathname.endsWith('/chat/stream')
    ? streamReply(nextId++, '模型口头说总价 ¥999.00，请确认。', orderApproval())
    : undefined;
  await mount();
  await send('结算');
  expect(card()).not.toBeNull();
  // The card total is the structured server amount (¥608.00), never ¥999.00.
  expect(container.querySelector('[data-approval-total]')?.textContent).toBe('¥608.00');
  expect(container.textContent).toContain('以本卡为准');
  // The same approval arriving again (e.g. a follow-up question) adds no card.
  await send('总价是多少');
  expect(container.querySelectorAll('[data-approval-card]')).toHaveLength(1);
});

test('confirm and cancel buttons send their pre-filled phrase through the chat stream channel', async () => {
  historyRows = []; let nextId = 102; let approval = orderApproval();
  override = (url, init) => url.pathname.endsWith('/chat/stream') ? streamReply(nextId++, '已收到。', approval) : undefined;
  await mount();
  await send('结算');
  await act(async () => confirmButton().click());
  const sent = calls('/api/v1/guest/chat/stream');
  expect(sent).toHaveLength(2);
  expect(JSON.parse(String(sent[1][1]!.body)).message).toContain('确认下单');
  expect(JSON.parse(String(sent[1][1]!.body)).message).toContain('¥608.00');
  expect(sent[1][1]!.credentials).toBe('include');
  // No commerce endpoint exists for the card: everything rides the chat stream.
  expect(calls('/api/v1/guest/chat')).toHaveLength(0);

  // Cancel path on a fresh payment card.
  approval = payApproval();
  await send('下单了');
  const payCard = container.querySelectorAll('[data-approval-card]')[1] as HTMLElement;
  const payCancel = payCard.querySelector<HTMLButtonElement>('[data-approval-cancel]')!;
  await act(async () => payCancel.click());
  const cancelSent = calls('/api/v1/guest/chat/stream');
  expect(JSON.parse(String(cancelSent[cancelSent.length - 1][1]!.body)).message).toContain('取消');
  expect(payCard.getAttribute('data-approval-status')).toBe('cancelled');
  expect(payCancel.disabled).toBe(true);
});

test('a stage advance greys the earlier order card and adds the payment card', async () => {
  historyRows = []; let nextId = 102; let approval: CommerceApproval | null = orderApproval();
  override = url => url.pathname.endsWith('/chat/stream') ? streamReply(nextId++, '已下单。', approval) : undefined;
  await mount();
  await send('结算');
  expect(container.querySelectorAll('[data-approval-card]')).toHaveLength(1);
  approval = payApproval();
  await send('确认下单');
  const cards = container.querySelectorAll('[data-approval-card]');
  expect(cards).toHaveLength(2);
  expect(cards[0].getAttribute('data-approval-status')).toBe('done');
  expect(cards[1].getAttribute('data-approval-status')).toBe('active');
  expect((cards[0] as HTMLElement).querySelector<HTMLButtonElement>('[data-approval-confirm]')!.disabled).toBe(true);
});

test('completions without approval data render no card at all', async () => {
  historyRows = []; let nextId = 102;
  override = url => url.pathname.endsWith('/chat/stream') ? streamReply(nextId++, '模拟订单已创建。', null) : undefined;
  await mount();
  await send('结算');
  expect(container.querySelector('[data-approval-section]')).toBeNull();
  expect(card()).toBeNull();
});

test('a malformed approval payload never becomes a card', async () => {
  historyRows = []; let nextId = 102;
  const bad = { ...orderApproval(), total_minor: 'lots' } as unknown;
  override = url => url.pathname.endsWith('/chat/stream') ? streamReply(nextId++, '请确认。', bad as CommerceApproval) : undefined;
  await mount();
  await send('结算');
  expect(card()).toBeNull();
});
