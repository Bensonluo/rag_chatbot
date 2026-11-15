/** Store page contract: the adapted branding page keeps its section structure
 * and points every "engineering" link at the repository, never back at the
 * author's portfolio detail page (the decoupling contract of 2026-11-14).
 * The page ships BOTH locale trees in the static HTML; a pre-paint script
 * pins html[data-locale] and CSS gates the trees, so the visitor's locale
 * paints first with zero hydration swap — the copy contract covers both. */
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
  expect(html).toContain('4,170');
  expect(html).toContain('93.71%');
  expect(html).not.toContain('4,148');
  expect(html).not.toContain('4,146');
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
});

test('the status line follows the connection probe, never a hardcoded online', () => {
  // Static render happens before any probe resolves, so the shipped HTML
  // shows the honest "checking" state in both locales — a hardcoded
  // "● online" badge that lies when the API is down must not return.
  expect(html).toContain('● 连接中…');
  expect(html).toContain('● checking…');
  expect(html).not.toContain('● online');
  expect(html).not.toContain('● 在线');
  // W3C ARIA22: the badge is a polite live region so screen readers hear
  // the probe verdict change without needing focus on it.
  expect(html).toMatch(/<div[^>]*role="status">●/);
});

test('page content sits inside a main landmark for assistive tech', () => {
  expect(html).toContain('<main');
});

test('both locale trees ship in the static HTML for a zero-shift paint', () => {
  // The single-tree design swapped zh→en during hydration and shifted the
  // hero (CLS 0.207 for non-zh first visits); the fix is structural and
  // must not regress to swapping copy after paint.
  expect(html).toContain('data-locale="zh"');
  expect(html).toContain('data-locale="en"');
  expect(html).toContain(storeCopy.en.hero.sub);
  expect(html).toContain(storeCopy.zh.hero.sub);
  // The parse-blocking bootstrap that pins <html data-locale> pre-paint.
  expect(html).toContain('portfolio-locale');
  expect(html).toContain('dataset.locale');
});

test('a language toggle is rendered in the nav', () => {
  expect(html).toContain('<button');
  expect(html).toContain('EN');
  expect(html).toMatch(/aria-label="(Switch to English|切换到中文)"/);
});

test('structured data describes the application for search and AI agents', () => {
  expect(html).toContain('application/ld+json');
  expect(html).toContain('"@type":"SoftwareApplication"');
  expect(html).toMatch(/"softwareVersion":"0\.\d+\.\d+"/);
  expect(html).toContain('https://github.com/Bensonluo/rag_chatbot');
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
  expect(storeCopy.en.bounds.items[0].v).toBe('4,170');
});
