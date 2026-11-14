/** Bilingual copy for the Agent Store landing page.
 *
 * The page follows the visitor's locale (see `lib/i18n/context.tsx`) with a
 * manual EN / 中文 toggle in the nav. `zh` is the server-rendered default
 * (static export renders one HTML); the client hydrates to the stored or
 * system locale. Keep both trees structurally identical — the page test
 * asserts key parity so a missing translation fails the suite.
 *
 * Rich paragraphs with inline emphasis are stored as segment arrays
 * (`{ t, b }` = text + bold); everything else is plain strings. */

import type { Locale } from './context';

export interface CopySegment {
  t: string;
  b?: boolean;
}

export interface FlowStep {
  label: string;
  gate: boolean;
}

export interface StoreCopy {
  nav: { demo: string; trust: string; service: string; code: string };
  toggle: { toEn: string; toZh: string };
  hero: {
    kicker: string;
    sub: string;
    lede: CopySegment[];
    ctaPrimary: string;
    ctaGhost: string;
    note: string;
  };
  flow: {
    buyTag: string;
    serviceTag: string;
    buySteps: FlowStep[];
    serviceSteps: FlowStep[];
  };
  demo: {
    h: string;
    p: string;
    chatHead: string;
    caption: string;
  };
  trust: {
    h: string;
    p: string;
    cards: { title: string; body: CopySegment[] }[];
  };
  service: {
    h: string;
    p: string;
    cards: { title: string; body: string }[];
    cta: string;
  };
  bounds: {
    h: string;
    items: { v: string; k: string }[];
  };
  footer: {
    demo: string;
    code: string;
    builtBy: string;
    legal: string;
  };
}

const zh: StoreCopy = {
  nav: {
    demo: '在线体验',
    trust: '为什么可信',
    service: '智能客服',
    code: '工程实现',
  },
  toggle: { toEn: 'EN', toZh: '中文' },
  hero: {
    kicker: 'AGENT STORE · OPEN SOURCE',
    sub: '购物 × 客服，一个可托付的 Agent',
    lede: [
      { t: '帮你买', b: true },
      { t: '——搜索 → 推荐 → 报价 → 你确认 → 下单 → 付款再确认；' },
      { t: '管到底', b: true },
      { t: '——查订单、问政策（回答带引用）、办退款（再确认）、转人工。' },
      { t: '商品与交易由独立的权威系统记账，模型只做语义与编排——' },
      { t: '它说的话不算数，卡片才算数', b: true },
      { t: '。' },
    ],
    ctaPrimary: '开始购物 ↓',
    ctaGhost: '看工程是怎么实现的 →',
    note: 'simulated commerce · mock payments only — 不产生真实扣款',
  },
  flow: {
    buyTag: '买',
    serviceTag: '客服',
    buySteps: [
      { label: '搜索', gate: false },
      { label: '推荐', gate: false },
      { label: '报价', gate: false },
      { label: '◉ 你确认', gate: true },
      { label: '下单', gate: false },
      { label: '◉ 付款再确认', gate: true },
    ],
    serviceSteps: [
      { label: '查订单', gate: false },
      { label: '政策问答·带引用', gate: false },
      { label: '◉ 退款再确认', gate: true },
      { label: '人工兜底', gate: false },
    ],
  },
  demo: {
    h: '直接开聊，不用注册',
    p: '游客身份即开即用。说一句「送女友一副降噪耳机，预算 400 内」，看它怎么把事情办完——每一步都停下来等你点头。',
    chatHead: 'Agent Store · AI 店员',
    caption: '真实界面 · 无脚本 — 触发后每个确认点都由你决定',
  },
  trust: {
    h: '为什么敢让它碰钱',
    p: '信任不靠人设，靠机制。三条硬约束写进交易链路，模型绕不过去。',
    cards: [
      {
        title: '报价确认，绝不先斩后奏',
        body: [
          { t: '每一笔都' },
          { t: '先报价、等你点头', b: true },
          { t: '；下单和付款各自单独确认。服务器下发的权威卡片永远压过模型的话术——' },
          { t: '卡片才算数', b: true },
          { t: '。' },
        ],
      },
      {
        title: 'exactly-once 降价提醒',
        body: [
          { t: '「ANC-02 降到 300 以内来问我」——' },
          { t: '预授权一次，触发只问你一次', b: true },
          { t: '，绝不代你下单。规则过期自动失效，不打扰。' },
        ],
      },
      {
        title: '幂等交易，断线不重复扣款',
        body: [
          { t: '每笔写入带幂等键：断线重连、连点两次「确认」，' },
          { t: '都只会有一笔订单', b: true },
          { t: '。购物车带版本号，并发修改会被拒收。' },
        ],
      },
    ],
  },
  service: {
    h: '回答有出处，退款过确认门，人工兜底',
    p: '客服不是另一个机器人，也不止于售后：同一条对话里查单、问政策、办退款——政策回答附引用，退款必须再确认，复杂诉求自动建工单转人工。',
    cards: [
      {
        title: '订单时间线',
        body: '「我的订单怎么样了」凭订单号从权威行重建全链事件，只读且仅本人可见。',
      },
      {
        title: '政策问答 · 带引用',
        body: '退款/退货/物流政策回答附 document/version/chunk 来源，答案可溯源、不编不猜。',
      },
      {
        title: '幂等退款',
        body: '只在你最新一次明确确认后执行；重复说「确认」只退一次，绝不退两次。',
      },
      {
        title: '人工衔接',
        body: '情绪/复杂度触发自动建工单，客服带着完整上下文接手，双向可续。',
      },
    ],
    cta: '试试「退货政策是什么」「我要退款」↓',
  },
  bounds: {
    h: '边界写在代码里，不写在文案里',
    items: [
      { v: '4,148', k: 'tests · 93.71% coverage' },
      { v: '≤ 5', k: 'typed SKU / 每轮候选' },
      { v: '20 / 30', k: 'steps / LLM calls 上限' },
      { v: '300s', k: '报价有效期 TTL' },
      { v: '5 / 30d', k: 'watch 规则数 / 有效期' },
    ],
  },
  footer: {
    demo: '在线体验',
    code: '工程实现（GitHub）',
    builtBy: 'Built by Peng Luo · benluo.art',
    legal: '© 2026 Agent Store — 演示项目：模拟商品目录与模拟支付，不产生真实交易',
  },
};

const en: StoreCopy = {
  nav: {
    demo: 'Live demo',
    trust: 'Why trust it',
    service: 'Support',
    code: 'The code',
  },
  toggle: { toEn: 'EN', toZh: '中文' },
  hero: {
    kicker: 'AGENT STORE · OPEN SOURCE',
    sub: 'Shopping × support, one agent you can trust',
    lede: [
      { t: 'It shops for you', b: true },
      {
        t: ' — search → recommend → quote → your confirmation → order → payment confirmed again; ',
      },
      { t: 'it owns the follow-through', b: true },
      {
        t: ' — order lookup, policy questions (answers cite sources), refunds (behind a fresh confirmation), human handoff. ',
      },
      {
        t: 'Goods and transactions are recorded by an independent authoritative system; the model only handles semantics and orchestration — ',
      },
      { t: "what it says doesn't count, the cards do", b: true },
      { t: '.' },
    ],
    ctaPrimary: 'Start shopping ↓',
    ctaGhost: "See how it's built →",
    note: 'simulated commerce · mock payments only — no real charges',
  },
  flow: {
    buyTag: 'Buy',
    serviceTag: 'Support',
    buySteps: [
      { label: 'Search', gate: false },
      { label: 'Recommend', gate: false },
      { label: 'Quote', gate: false },
      { label: '◉ You confirm', gate: true },
      { label: 'Order', gate: false },
      { label: '◉ Pay — confirmed again', gate: true },
    ],
    serviceSteps: [
      { label: 'Order lookup', gate: false },
      { label: 'Policy answers · cited', gate: false },
      { label: '◉ Refund — confirmed again', gate: true },
      { label: 'Human handoff', gate: false },
    ],
  },
  demo: {
    h: 'Start chatting — no sign-up',
    p: 'Guest access, ready on the first message. Say “noise-cancelling earbuds as a gift, under ¥400” and watch it get things done — pausing for your nod at every step.',
    chatHead: 'Agent Store · AI clerk',
    caption: 'Real interface · unscripted — every confirmation point is yours to decide',
  },
  trust: {
    h: 'Why it is safe to let it touch money',
    p: 'Trust comes from mechanisms, not personas. Three hard constraints are wired into the transaction path — the model cannot route around them.',
    cards: [
      {
        title: 'Quoted first, never act-then-report',
        body: [
          { t: 'Every purchase is ' },
          { t: 'quoted first and waits for your nod', b: true },
          {
            t: '; ordering and paying are each confirmed separately. The server-issued authoritative card always overrides the model’s pitch — ',
          },
          { t: 'the card is what counts', b: true },
          { t: '.' },
        ],
      },
      {
        title: 'exactly-once price alerts',
        body: [
          { t: '“Ask me when ANC-02 drops under ¥300” — ' },
          { t: 'authorized once, it asks you exactly once', b: true },
          { t: ', and never orders for you. Expired rules lapse silently.' },
        ],
      },
      {
        title: 'Idempotent transactions, no double charges',
        body: [
          { t: 'Every write carries an idempotency key: reconnect, double-click “confirm” — ' },
          { t: 'there is still exactly one order', b: true },
          { t: '. Carts carry version numbers; conflicting edits are rejected.' },
        ],
      },
    ],
  },
  service: {
    h: 'Answers cite sources, refunds pass a gate, humans catch the rest',
    p: 'Support is not another chatbot, and it does not stop at after-sales: in the same conversation — order lookup, policy questions, refunds — answers carry citations, refunds require a fresh confirmation, and complex cases open a ticket to a human.',
    cards: [
      {
        title: 'Order timeline',
        body: '“Where is my order” rebuilds the full event chain from authoritative rows — read-only, visible to you alone.',
      },
      {
        title: 'Policy answers · cited',
        body: 'Refund / return / shipping answers cite document/version/chunk sources — traceable, never invented.',
      },
      {
        title: 'Idempotent refunds',
        body: 'Executed only after your latest explicit confirmation; saying “confirm” twice refunds once, never twice.',
      },
      {
        title: 'Human handoff',
        body: 'Sentiment and complexity triggers auto-create a ticket; staff take over with full context, two-way.',
      },
    ],
    cta: 'Try “what is the return policy” or “I want a refund” ↓',
  },
  bounds: {
    h: 'The bounds live in code, not in copy',
    items: [
      { v: '4,148', k: 'tests · 93.71% coverage' },
      { v: '≤ 5', k: 'typed SKUs / candidates per turn' },
      { v: '20 / 30', k: 'steps / LLM-call caps' },
      { v: '300s', k: 'quote validity TTL' },
      { v: '5 / 30d', k: 'watch rules / validity' },
    ],
  },
  footer: {
    demo: 'Live demo',
    code: 'The code (GitHub)',
    builtBy: 'Built by Peng Luo · benluo.art',
    legal: '© 2026 Agent Store — a demo: simulated catalog and mock payments, no real transactions',
  },
};

export const storeCopy: Record<Locale, StoreCopy> = { zh, en };
