import type { Metadata } from "next";
import { Fragment } from "react";
import { ChatInterfaceUpdated } from "@/components/ChatInterfaceUpdated";
import styles from "./page.module.css";

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"
  ),
  title: "Agent Store — 可托付的购物 Agent · 智能客服",
  description:
    "既是把事办完的购物 Agent，也是可溯源的智能客服。购物：搜索 → 推荐 → 报价 → 你确认 → 下单 → 付款再确认；客服：订单时间线、带引用政策问答、幂等退款、人工衔接工单。报价确认、幂等下单、断线不重复扣款、exactly-once 降价提醒。模拟商城，不产生真实交易。",
  openGraph: {
    title: "Agent Store · 购物 Agent × 智能客服",
    description:
      "shop & support in one agent · quotes before orders · idempotent payments · cited policy answers · human handoff — 模拟商城演示，不产生真实交易",
    images: ["/images/projects/rag-cover.png"],
    type: "website",
  },
};

// SKU dot-grid motif: indigo cells, cyan every 3rd, outlined every 7th,
// with a few "hot" amber glows — mirrors the OG cover's visual signature.
const MOTIF_CELLS = Array.from({ length: 88 }, (_, index) => index + 1);
const MOTIF_HOT = new Set([13, 25, 37, 55, 79]);

const FLOW_STEPS = [
  { label: "搜索", gate: false },
  { label: "推荐", gate: false },
  { label: "报价", gate: false },
  { label: "◉ 你确认", gate: true },
  { label: "下单", gate: false },
  { label: "◉ 付款再确认", gate: true },
];

// Customer service is a co-equal main line: same chat, same gates.
const SERVICE_STEPS = [
  { label: "查订单", gate: false },
  { label: "政策问答·带引用", gate: false },
  { label: "◉ 退款再确认", gate: true },
  { label: "人工兜底", gate: false },
];

export default function StorePage() {
  return (
    <div className={styles.root} lang="zh-CN">
      <nav className={styles.bar}>
        <div className={`${styles.wrap} ${styles.barIn}`}>
          <div className={styles.brand}>
            <span className={styles.brandDot} aria-hidden />
            Agent&nbsp;<span className={styles.brandStore}>Store</span>
          </div>
          <div className={`${styles.barLinks} ${styles.mono}`}>
            <a href="#demo" className={styles.ctaLink}>
              在线体验
            </a>
            <a href="#trust" className={styles.hideM}>
              为什么可信
            </a>
            <a href="#service" className={styles.hideM}>
              智能客服
            </a>
            <a
              href="https://github.com/Bensonluo/rag_chatbot"
              className={styles.hideM}
            >
              工程实现
            </a>
          </div>
        </div>
      </nav>

      <header className={styles.hero}>
        <div className={styles.gridMotif} aria-hidden>
          {MOTIF_CELLS.map((cell) => (
            <i
              key={cell}
              className={MOTIF_HOT.has(cell) ? styles.hot : undefined}
            />
          ))}
        </div>
        <div className={styles.wrap}>
          <div className={`${styles.kicker} ${styles.mono}`}>
            <span className={styles.kickerDot} aria-hidden />
            AGENT STORE · OPEN SOURCE
          </div>
          <h1>
            Agent <span>Store</span>
          </h1>
          <div className={styles.sub}>购物 × 客服，一个可托付的 Agent</div>
          <p className={styles.lede}>
            <b>帮你买</b>——搜索 → 推荐 → 报价 → 你确认 → 下单 → 付款再确认；
            <b>管到底</b>——查订单、问政策（回答带引用）、办退款（再确认）、转人工。
            商品与交易由独立的权威系统记账，模型只做语义与编排——
            <b>它说的话不算数，卡片才算数</b>。
          </p>
          <div className={styles.ctaRow}>
            <a className={`${styles.btn} ${styles.btnPrimary}`} href="#demo">
              开始购物 ↓
            </a>
            <a
              className={`${styles.btn} ${styles.btnGhost}`}
              href="https://github.com/Bensonluo/rag_chatbot"
            >
              看工程是怎么实现的 →
            </a>
          </div>
          <div className={`${styles.heroNote} ${styles.mono}`}>
            simulated commerce · mock payments only — 不产生真实扣款
          </div>
        </div>
      </header>

      <div className={styles.flow}>
        <div className={`${styles.wrap} ${styles.flowStack}`}>
          <div className={`${styles.flowIn} ${styles.mono}`}>
            <span className={styles.flowTag}>买</span>
            {FLOW_STEPS.map((step, index) => (
              <Fragment key={step.label}>
                <span
                  className={`${styles.step} ${step.gate ? styles.gate : ""}`}
                >
                  <b>{step.label}</b>
                </span>
                {index < FLOW_STEPS.length - 1 && (
                  <span className={styles.arr} aria-hidden>
                    →
                  </span>
                )}
              </Fragment>
            ))}
          </div>
          <div className={`${styles.flowIn} ${styles.mono}`}>
            <span className={styles.flowTag}>客服</span>
            {SERVICE_STEPS.map((step, index) => (
              <Fragment key={step.label}>
                <span
                  className={`${styles.step} ${step.gate ? styles.gate : ""}`}
                >
                  <b>{step.label}</b>
                </span>
                {index < SERVICE_STEPS.length - 1 && (
                  <span className={styles.arr} aria-hidden>
                    →
                  </span>
                )}
              </Fragment>
            ))}
          </div>
        </div>
      </div>

      <section className={`${styles.sec} ${styles.demoSec}`} id="demo">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>LIVE DEMO</div>
          <h2 className={styles.secH}>直接开聊，不用注册</h2>
          <p className={styles.secP}>
            游客身份即开即用。说一句「送女友一副降噪耳机，预算 400
            内」，看它怎么把事情办完——每一步都停下来等你点头。
          </p>

          <div className={styles.chatPanel}>
            <div className={styles.chatHead}>
              <div className={styles.who}>
                <span className={styles.whoDot} aria-hidden />
                Agent Store · AI 店员
              </div>
              <div className={`${styles.status} ${styles.mono}`}>● online</div>
            </div>
            <div className={`${styles.livePad} dark`}>
              <ChatInterfaceUpdated />
            </div>
          </div>
          <div className={`${styles.demoCap} ${styles.mono}`}>
            真实界面 · 无脚本 — 触发后每个确认点都由你决定
          </div>
        </div>
      </section>

      <section className={`${styles.sec} ${styles.trustSec}`} id="trust">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            WHY TRUST IT
          </div>
          <h2 className={styles.secH}>为什么敢让它碰钱</h2>
          <p className={styles.secP}>
            信任不靠人设，靠机制。三条硬约束写进交易链路，模型绕不过去。
          </p>
          <div className={styles.cards}>
            <div className={styles.card}>
              <div className={`${styles.num} ${styles.mono}`}>01</div>
              <div className={styles.glyph} aria-hidden>
                ¥
              </div>
              <h3>报价确认，绝不先斩后奏</h3>
              <p>
                每一笔都<b>先报价、等你点头</b>；下单和付款各自单独确认。服务器下发的权威卡片永远压过模型的话术——
                <b>卡片才算数</b>。
              </p>
            </div>
            <div className={styles.card}>
              <div className={`${styles.num} ${styles.mono}`}>02</div>
              <div className={styles.glyph} aria-hidden>
                ◔
              </div>
              <h3>
                <em>exactly-once</em> 降价提醒
              </h3>
              <p>
                「ANC-02 降到 300 以内来问我」——
                <b>预授权一次，触发只问你一次</b>，绝不代你下单。规则过期自动失效，不打扰。
              </p>
            </div>
            <div className={styles.card}>
              <div className={`${styles.num} ${styles.mono}`}>03</div>
              <div className={styles.glyph} aria-hidden>
                ⧉
              </div>
              <h3>幂等交易，断线不重复扣款</h3>
              <p>
                每笔写入带幂等键：断线重连、连点两次「确认」，
                <b>都只会有一笔订单</b>。购物车带版本号，并发修改会被拒收。
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className={`${styles.sec} ${styles.afterSec}`} id="service">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            CUSTOMER SERVICE
          </div>
          <h2 className={styles.secH}>回答有出处，退款过确认门，人工兜底</h2>
          <p className={styles.secP}>
            客服不是另一个机器人，也不止于售后：同一条对话里查单、问政策、办退款——
            政策回答附引用，退款必须再确认，复杂诉求自动建工单转人工。
          </p>
          <div className={styles.miniCards}>
            <div className={styles.miniCard}>
              <h3>订单时间线</h3>
              <p>
                「我的订单怎么样了」凭订单号从权威行重建全链事件，
                只读且仅本人可见。
              </p>
            </div>
            <div className={styles.miniCard}>
              <h3>政策问答 · 带引用</h3>
              <p>
                退款/退货/物流政策回答附 document/version/chunk 来源，
                答案可溯源、不编不猜。
              </p>
            </div>
            <div className={styles.miniCard}>
              <h3>幂等退款</h3>
              <p>
                只在你最新一次明确确认后执行；
                重复说「确认」只退一次，绝不退两次。
              </p>
            </div>
            <div className={styles.miniCard}>
              <h3>人工衔接</h3>
              <p>
                情绪/复杂度触发自动建工单，客服带着完整上下文接手，
                双向可续。
              </p>
            </div>
          </div>
          <div className={styles.loopCtas}>
            <a className={`${styles.btn} ${styles.btnPrimary}`} href="#demo">
              试试「退货政策是什么」「我要退款」↓
            </a>
          </div>
        </div>
      </section>

      <section className={styles.boundsSec}>
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            HARD BOUNDS
          </div>
          <h2 className={styles.secH} style={{ fontSize: 26 }}>
            边界写在代码里，不写在文案里
          </h2>
          <div
            className={`${styles.bounds} ${styles.mono}`}
            style={{ marginTop: 28 }}
          >
            <div className={styles.bound}>
              <span className={styles.v}>4,146</span>
              <span className={styles.k}>tests · 93.66% coverage</span>
            </div>
            <div className={styles.bound}>
              <span className={styles.v}>≤ 5</span>
              <span className={styles.k}>typed SKU / 每轮候选</span>
            </div>
            <div className={styles.bound}>
              <span className={styles.v}>20 / 30</span>
              <span className={styles.k}>steps / LLM calls 上限</span>
            </div>
            <div className={styles.bound}>
              <span className={styles.v}>300s</span>
              <span className={styles.k}>报价有效期 TTL</span>
            </div>
            <div className={styles.bound}>
              <span className={styles.v}>5 / 30d</span>
              <span className={styles.k}>watch 规则数 / 有效期</span>
            </div>
          </div>
        </div>
      </section>

      <footer className={styles.footEl}>
        <div className={styles.wrap}>
          <div className={styles.foot}>
            <div>
              <div className={styles.footT}>
                Agent <span>Store</span>
              </div>
              <div className={`${styles.footEn} ${styles.mono}`}>
                quotes before orders · idempotent payments · cited policy
                answers · cards win over narration
              </div>
            </div>
            <div className={styles.footR}>
              <a href="#demo">在线体验</a>
              <a href="https://github.com/Bensonluo/rag_chatbot">
                工程实现（GitHub）
              </a>
              <div className={`${styles.built} ${styles.mono}`}>
                Built by Peng Luo · benluo.art
              </div>
            </div>
          </div>
          <div className={`${styles.legal} ${styles.mono}`}>
            <span>
              © 2026 Agent Store —
              演示项目：模拟商品目录与模拟支付，不产生真实交易
            </span>
            <span>github.com/Bensonluo/rag_chatbot</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
