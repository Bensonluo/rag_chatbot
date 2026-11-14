/** Store page contract: the adapted branding page keeps its section structure
 * and points every "engineering" link at the repository, never back at the
 * author's portfolio detail page (the decoupling contract of 2026-11-14).
 * The page renders the zh tree on the server (static export) and follows the
 * visitor's locale on the client — the copy contract covers both trees. */
import { renderToStaticMarkup } from 'react-dom/server.node';
import StorePage from '@/app/page';
import { storeCopy } from '@/lib/i18n/store-copy';

const html = renderToStaticMarkup(<StorePage />);

test('renders the product hero with the open-source kicker', () => {
  expect(html).toContain('AGENT STORE · OPEN SOURCE');
  expect(html).toMatch(/Agent\s*<!-- -->?\s*<span>Store<\/span>|Agent <span>Store<\/span>/);
  expect(html).toContain('购物 × 客服，一个可托付的 Agent');
  expect(html).toContain('simulated commerce · mock payments only');
});

test('keeps the trust, service and hard-bounds sections', () => {
  expect(html).toContain('为什么敢让它碰钱');
  expect(html).toContain('exactly-once');
  expect(html).toContain('幂等交易，断线不重复扣款');
  expect(html).toContain('回答有出处，退款过确认门，人工兜底');
  expect(html).toContain('边界写在代码里，不写在文案里');
});

test('hard bounds carry the current measured baseline, not stale counts', () => {
  expect(html).toContain('4,148');
  expect(html).toContain('93.71%');
  expect(html).not.toContain('4,146');
  expect(html).not.toContain('4,099');
});

test('every engineering link points at the repository, never the portfolio', () => {
  const repoLinks = html.match(/https:\/\/github\.com\/Bensonluo\/rag_chatbot/g) ?? [];
  expect(repoLinks.length).toBeGreaterThanOrEqual(3); // nav, hero CTA, footer
  expect(html).not.toContain('/projects/rag-chatbot');
  expect(html).not.toContain('BENLUO.ART · AGENT STORE');
});

test('footer states the demo boundary and the repo as home', () => {
  expect(html).toContain('演示项目：模拟商品目录与模拟支付，不产生真实交易');
  expect(html).toContain('github.com/Bensonluo/rag_chatbot');
});

test('the live chat panel is embedded with its dark scope', () => {
  expect(html).toContain('Agent Store · AI 店员');
  expect(html).toContain('● online');
});

test('a language toggle is rendered in the nav', () => {
  expect(html).toContain('<button');
  expect(html).toContain('EN');
  expect(html).toMatch(/aria-label="(Switch to English|切换到中文)"/);
});

/** Structural parity: every locale must carry the same copy tree shape, so a
 * key added to one language without its mirror fails here, not in production. */
function shape(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map((v) => shape(v)).join(',')}]`;
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((k) => `${k}:${shape(record[k])}`).join(',')}}`;
  }
  return typeof value;
}

test('zh and en copy trees are structurally identical', () => {
  expect(shape(storeCopy.en)).toBe(shape(storeCopy.zh));
  expect(storeCopy.en.hero.sub).toBe('Shopping × support, one agent you can trust');
  expect(storeCopy.en.bounds.items[0].v).toBe('4,148');
});
