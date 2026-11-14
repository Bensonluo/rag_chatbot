import { ReadableStream } from 'node:stream/web';
import { TextDecoder, TextEncoder } from 'node:util';
import { ApiClient } from '@/lib/api';
import {
  applyTraceEvent,
  EMPTY_STATS,
  ExecutionChainPanel,
  type StageCard,
} from '@/components/ExecutionChainPanel';

const { renderToStaticMarkup } = require('react-dom/server.node') as typeof import('react-dom/server');

const originalFetch = Object.getOwnPropertyDescriptor(globalThis, 'fetch');
const originalDecoder = Object.getOwnPropertyDescriptor(globalThis, 'TextDecoder');

beforeEach(() => {
  localStorage.clear();
  Object.defineProperty(globalThis, 'TextDecoder', { configurable: true, value: TextDecoder });
});

afterEach(() => {
  for (const [name, descriptor] of [['fetch', originalFetch], ['TextDecoder', originalDecoder]] as const) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else Reflect.deleteProperty(globalThis, name);
  }
});

async function renderTraces(events: unknown[]): Promise<string> {
  const bytes = new TextEncoder().encode(events.map(event =>
    `event: trace\ndata: ${JSON.stringify(event)}\n\n`
  ).join('') + 'data: Actual answer\n\ndata: [DONE]\n\n');
  const body = new ReadableStream({
    start(controller) {
      // Metadata is decoded by the actual API client before its callback
      // reaches the actual stage reducer and React renderer.
      for (let offset = 0; offset < bytes.length; offset += 5) {
        controller.enqueue(bytes.slice(offset, offset + 5));
      }
      controller.close();
    },
  });
  const fetchStub = jest.fn(async () => ({
    ok: true,
    status: 200,
    headers: new Map([['Content-Type', 'text/event-stream']]),
    body,
  }));
  Object.defineProperty(globalThis, 'fetch', { configurable: true, writable: true, value: fetchStub });
  let stages: StageCard[] = [];
  const keys = { n: 1 };
  let answer = '';
  for await (const chunk of ApiClient.streamChat('Local question', 7, event => {
    stages = applyTraceEvent(stages, event, keys);
  })) answer += chunk;
  expect(answer).toBe('Actual answer');
  expect(fetchStub).toHaveBeenCalledTimes(1);
  return renderToStaticMarkup(<ExecutionChainPanel stages={stages} stats={EMPTY_STATS} turn={1} />);
}

test.each(['__proto__', 'constructor', 'toString'])('unknown stage %s uses its own fallback label', async stage => {
  const html = await renderTraces([{ stage, status: 'start' }]);
  expect(html).toContain(stage);
});

test.each([
  { name: 'pending slots are not an array', stage: 'cs.slots', detail: { pending_slots: 123 } },
  { name: 'pending slots are an array-like object', stage: 'cs.slots', detail: { pending_slots: { length: 1 } } },
  { name: 'retrieval sources are an array-like object', stage: 'cs.retrieval', detail: { sources: { length: 1 } } },
  { name: 'agent tools are an array-like object', stage: 'cs.agent', detail: { tools: { length: 1 } } },
  { name: 'filled slots are a string', stage: 'cs.slots', detail: { filled_slots: 'untrusted slots' } },
  { name: 'retrieval count is an object', stage: 'cs.retrieval', detail: { docs: { value: 2 } } },
  { name: 'handoff identity is an object', stage: 'cs.handoff', detail: { ticket_id: { value: 12 }, queue_position: -1 } },
  { name: 'confidence overflows a percentage', stage: 'cs.intent', detail: { intent: 'policy', confidence: 1e308 } },
  { name: 'claim fields are objects', stage: 'cs.claim_gate', detail: { action: 'rewrite', clause: { text: 'hidden' }, grounded: [] } },
])('$name cannot crash or masquerade as a valid display value', async ({ stage, detail }) => {
  const html = await renderTraces([{ stage, status: 'detail', detail }]);
  expect(html).not.toMatch(/\[object Object\]|Infinity|NaN|untrusted slots|0=u|位置 -1/);
  if (stage === 'cs.slots') expect(html).not.toContain('无槽位');
  if (stage === 'cs.agent') expect(html).not.toContain('未调用工具');
});

test.each([
  { stage: 'cs.slots', detail: { pending_slots: [{ bad: true }, 'order_id', null] }, valid: 'order_id' },
  { stage: 'cs.retrieval', detail: { sources: [{ bad: true }, 'policy-v1', null] }, valid: 'policy-v1' },
  { stage: 'cs.tool', detail: { tools: [{ bad: true }, 'refund_order', null] }, valid: 'refund_order' },
  { stage: 'cs.slots', detail: { filled_slots: { order_id: 'ORD1001', invalid: { bad: true } } }, valid: 'ORD1001' },
])('$stage preserves valid items while omitting malformed collection entries', async ({ stage, detail, valid }) => {
  const html = await renderTraces([{ stage, status: 'detail', detail }]);
  expect(html).toContain(valid);
  expect(html).not.toContain('[object Object]');
  expect(html).not.toContain('invalid=');
});

test('bad FAQ and guardrail flags cannot become success or blocked evidence', async () => {
  const html = await renderTraces([
    { stage: 'cs.faq', status: 'detail', detail: { hit: 'false', faq_id: 'policy' } },
    { stage: 'cs.guardrail_input', status: 'detail', detail: { blocked: 'false', redacted: {} } },
  ]);
  expect(html).not.toContain('hit:policy');
  expect(html).not.toContain('拦截');
  expect(html).not.toContain('PII 已脱敏');
  expect(html).not.toContain('通过');
});

test('invalid numeric facts do not become negative or object-valued counters', async () => {
  const html = await renderTraces([
    { stage: 'cs.memory', status: 'detail', detail: { facts: -4 } },
    { stage: 'cs.claim_gate', status: 'detail', detail: { facts: { value: 3 } } },
  ]);
  expect(html).not.toContain('召回 -4');
  expect(html).not.toContain('[object Object]');
  expect(html).not.toContain('校验通过');
});

test('valid trace details retain slots, sources, tools, counters and duration', async () => {
  const html = await renderTraces([
    { stage: 'cs.intent', status: 'detail', detail: { intent: 'policy', confidence: 0.91 } },
    { stage: 'cs.slots', status: 'detail', detail: { filled_slots: { order_id: 'ORD1001' }, pending_slots: ['reason'] } },
    { stage: 'cs.retrieval', status: 'detail', detail: { docs: 2, sources: ['policy-v1'], mode: 'hybrid' } },
    { stage: 'cs.tool', status: 'detail', detail: { tools: ['query_order'] } },
    { stage: 'cs.handoff', status: 'detail', detail: { ticket_id: 12, queue_position: 2 } },
    { stage: 'cs.memory', status: 'detail', detail: { facts: 3 } },
    { stage: 'cs.claim_gate', status: 'detail', detail: { facts: 4 } },
    { stage: 'cs.faq', status: 'detail', detail: { hit: false } },
    { stage: 'cs.guardrail_input', status: 'detail', detail: { blocked: false, redacted: false } },
    { stage: 'cs.retrieval', status: 'end', ms: 12 },
    { stage: 'cs.persistence', status: 'start' },
    { stage: 'cs.persistence', status: 'end', ms: 34 },
  ]);
  for (const text of ['91%', 'ORD1001', 'reason', 'policy-v1', 'hybrid', 'query_order', '12ms', '召回 3', '4 条策展事实校验通过', '通过', '会话持久化']) {
    expect(html).toContain(text);
  }
});
test('tool trace cards name every concrete tool, with raw fallback for unknown tools', async () => {
  const html = await renderTraces([
    { stage: 'cs.agent', status: 'start', detail: { max_steps: 6 } },
    { stage: 'cs.tool', status: 'start', detail: { tool: 'commerce_graph_context', step: 0 } },
    { stage: 'cs.tool', status: 'end', ms: 823, detail: { tool: 'commerce_graph_context', ok: true } },
    {
      stage: 'cs.tool',
      status: 'start',
      detail: { tool: 'commerce_propose_category', step: 1 },
    },
    {
      stage: 'cs.tool',
      status: 'end',
      ms: 32,
      detail: { tool: 'commerce_propose_category', ok: true },
    },
    {
      stage: 'cs.tool',
      status: 'start',
      detail: { tool: 'commerce_graph_upsert', step: 1, background: true },
    },
    {
      stage: 'cs.tool',
      status: 'end',
      ms: 640,
      detail: { tool: 'commerce_graph_upsert', ok: true, background: true },
    },
    { stage: 'cs.tool', status: 'start', detail: { tool: 'commerce_search_products', step: 1 } },
    { stage: 'cs.tool', status: 'end', ms: 137, detail: { tool: 'commerce_search_products', ok: true } },
    { stage: 'cs.tool', status: 'start', detail: { tool: 'search_knowledge_base', step: 2 } },
    { stage: 'cs.tool', status: 'end', ms: 91, detail: { tool: 'search_knowledge_base', ok: true } },
    { stage: 'cs.tool', status: 'start', detail: { tool: 'future_vendor_tool', step: 3 } },
    { stage: 'cs.tool', status: 'end', ms: 12, detail: { tool: 'future_vendor_tool', ok: true } },
    { stage: 'cs.agent', status: 'end', ms: 1100, detail: { tools: 6, outcome: 'answered' } },
  ]);
  for (const text of [
    '商品图谱语义查询',
    '动态铸类',
    '图谱更新',
    '商品搜索',
    '知识库检索',
    'future_vendor_tool',
    '823ms',
  ]) {
    expect(html).toContain(text);
  }
  expect(html).not.toContain('开始');
  expect(html).not.toContain('结束');
});
