'use client';

/** The Agent Store landing page body. A client component so the nav toggle
 * and the embedded chat stay interactive. Landing copy is rendered
 * BILINGUALLY: both locale trees ship in the static HTML and CSS gates them
 * on <html data-locale>, which the parse-blocking LOCALE_BOOTSTRAP script
 * pins before first paint. The page therefore paints in the visitor's
 * locale with zero post-hydration reflow — the previous single-tree design
 * swapped zh→en during hydration and measurably shifted the hero (CLS
 * 0.207 on non-zh first visits). The server-exported `page.tsx` keeps the
 * metadata and shells out to this. */

import { Fragment, useState, type ReactNode } from "react";
import { ChatErrorBoundary } from "@/components/ChatErrorBoundary";
import { ChatInterfaceUpdated } from "@/components/ChatInterfaceUpdated";
import { type ChatApiStatus } from "@/lib/chat-display";
import { LOCALE_BOOTSTRAP, useLanguage, type Locale } from "@/lib/i18n/context";
import {
  storeCopy,
  type CopySegment,
  type StoreCopy,
} from "@/lib/i18n/store-copy";
import styles from "./page.module.css";

// SKU dot-grid motif: indigo cells, cyan every 3rd, outlined every 7th,
// with a few "hot" amber glows — mirrors the OG cover's visual signature.
const MOTIF_CELLS = Array.from({ length: 88 }, (_, index) => index + 1);
const MOTIF_HOT = new Set([13, 25, 37, 55, 79]);

function Segments({ parts }: { parts: CopySegment[] }) {
  return (
    <>
      {parts.map((part, index) =>
        part.b ? <b key={index}>{part.t}</b> : <Fragment key={index}>{part.t}</Fragment>
      )}
    </>
  );
}

/** Renders children once per locale tree on the original element type, so
 * every container keeps its exact box and its nth-child styling. CSS in
 * page.module.css hides the tree that does not match html[data-locale];
 * display:none also drops it from the a11y tree and tab order. */
function Bi({
  as: Tag = "span",
  className,
  role,
  children,
}: {
  as?: "span" | "div" | "h2" | "p";
  className?: string;
  /** Passed to BOTH locale trees — the CSS-hidden one is excluded from the
   * accessibility tree, so `role="status"` live regions fire once. */
  role?: "status" | "alert";
  children: (copy: StoreCopy, locale: Locale) => ReactNode;
}) {
  return (
    <>
      <Tag data-locale="zh" lang="zh-CN" className={className} role={role}>
        {children(storeCopy.zh, "zh")}
      </Tag>
      <Tag data-locale="en" lang="en" className={className} role={role}>
        {children(storeCopy.en, "en")}
      </Tag>
    </>
  );
}

export function StoreLanding() {
  const { setLocale } = useLanguage();
  // The chat island lifts its health-probe verdict here; the header badge
  // renders the real state instead of a hardcoded "online".
  const [chatStatus, setChatStatus] = useState<ChatApiStatus>("checking");

  return (
    <div className={styles.root}>
      {/* Runs during parse, before anything below paints: resolve the
       * visitor's locale (stored preference, else system) onto
       * <html data-locale> so the bilingual trees below render in the
       * right language from the first frame. */}
      <script dangerouslySetInnerHTML={{ __html: LOCALE_BOOTSTRAP }} />
      <nav className={styles.bar}>
        <div className={`${styles.wrap} ${styles.barIn}`}>
          <div className={styles.brand}>
            <span className={styles.brandDot} aria-hidden />
            Agent&nbsp;<span className={styles.brandStore}>Store</span>
          </div>
          <Bi as="div" className={`${styles.barLinks} ${styles.mono}`}>
            {(copy, locale) => (
              <>
                <button
                  type="button"
                  className={styles.langToggle}
                  onClick={() => setLocale(locale === "zh" ? "en" : "zh")}
                  aria-label={locale === "zh" ? "Switch to English" : "切换到中文"}
                >
                  {locale === "zh" ? copy.toggle.toEn : copy.toggle.toZh}
                </button>
                <a href="#demo" className={styles.ctaLink}>
                  {copy.nav.demo}
                </a>
                <a href="#trust" className={styles.hideM}>
                  {copy.nav.trust}
                </a>
                <a href="#service" className={styles.hideM}>
                  {copy.nav.service}
                </a>
                <a
                  href="https://github.com/Bensonluo/rag_chatbot"
                  className={styles.hideM}
                >
                  {copy.nav.code}
                </a>
              </>
            )}
          </Bi>
        </div>
      </nav>

      <main>
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
          <Bi as="div" className={`${styles.kicker} ${styles.mono}`}>
            {(copy) => (
              <>
                <span className={styles.kickerDot} aria-hidden />
                {copy.hero.kicker}
              </>
            )}
          </Bi>
          <h1>
            Agent <span>Store</span>
          </h1>
          <Bi as="div" className={styles.sub}>
            {(copy) => copy.hero.sub}
          </Bi>
          <Bi as="p" className={styles.lede}>
            {(copy) => <Segments parts={copy.hero.lede} />}
          </Bi>
          <Bi as="div" className={styles.ctaRow}>
            {(copy) => (
              <>
                <a className={`${styles.btn} ${styles.btnPrimary}`} href="#demo">
                  {copy.hero.ctaPrimary}
                </a>
                <a
                  className={`${styles.btn} ${styles.btnGhost}`}
                  href="https://github.com/Bensonluo/rag_chatbot"
                >
                  {copy.hero.ctaGhost}
                </a>
              </>
            )}
          </Bi>
          <Bi as="div" className={`${styles.heroNote} ${styles.mono}`}>
            {(copy) => copy.hero.note}
          </Bi>
        </div>
      </header>

      <div className={styles.flow}>
        <Bi as="div" className={`${styles.wrap} ${styles.flowStack}`}>
          {(copy) => (
            <>
              <div className={`${styles.flowIn} ${styles.mono}`}>
                <span className={styles.flowTag}>{copy.flow.buyTag}</span>
                {copy.flow.buySteps.map((step, index) => (
                  <Fragment key={step.label}>
                    <span
                      className={`${styles.step} ${step.gate ? styles.gate : ""}`}
                    >
                      <b>{step.label}</b>
                    </span>
                    {index < copy.flow.buySteps.length - 1 && (
                      <span className={styles.arr} aria-hidden>
                        →
                      </span>
                    )}
                  </Fragment>
                ))}
              </div>
              <div className={`${styles.flowIn} ${styles.mono}`}>
                <span className={styles.flowTag}>{copy.flow.serviceTag}</span>
                {copy.flow.serviceSteps.map((step, index) => (
                  <Fragment key={step.label}>
                    <span
                      className={`${styles.step} ${step.gate ? styles.gate : ""}`}
                    >
                      <b>{step.label}</b>
                    </span>
                    {index < copy.flow.serviceSteps.length - 1 && (
                      <span className={styles.arr} aria-hidden>
                        →
                      </span>
                    )}
                  </Fragment>
                ))}
              </div>
            </>
          )}
        </Bi>
      </div>

      <section className={`${styles.sec} ${styles.demoSec}`} id="demo">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>LIVE DEMO</div>
          <Bi as="h2" className={styles.secH}>
            {(copy) => copy.demo.h}
          </Bi>
          <Bi as="p" className={styles.secP}>
            {(copy) => copy.demo.p}
          </Bi>

          <div className={styles.chatPanel}>
            <div className={styles.chatHead}>
              <Bi as="div" className={styles.who}>
                {(copy) => (
                  <>
                    <span className={styles.whoDot} aria-hidden />
                    {copy.demo.chatHead}
                  </>
                )}
              </Bi>
              {/* role=status (implicit aria-live=polite, W3C ARIA22) so
               * screen readers hear the probe verdict change. */}
              <Bi
                as="div"
                role="status"
                className={`${styles.status} ${styles.mono}`}
              >
                {(copy) => `● ${copy.demo.status[chatStatus]}`}
              </Bi>
            </div>
            <div className={`${styles.livePad} dark`}>
              <ChatErrorBoundary>
                <ChatInterfaceUpdated onApiStatusChange={setChatStatus} />
              </ChatErrorBoundary>
            </div>
          </div>
          <Bi as="div" className={`${styles.demoCap} ${styles.mono}`}>
            {(copy) => copy.demo.caption}
          </Bi>
        </div>
      </section>

      <section className={`${styles.sec} ${styles.trustSec}`} id="trust">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            WHY TRUST IT
          </div>
          <Bi as="h2" className={styles.secH}>
            {(copy) => copy.trust.h}
          </Bi>
          <Bi as="p" className={styles.secP}>
            {(copy) => copy.trust.p}
          </Bi>
          <Bi as="div" className={styles.cards}>
            {(copy) =>
              copy.trust.cards.map((card, index) => (
                <div className={styles.card} key={card.title}>
                  <div className={`${styles.num} ${styles.mono}`}>
                    {String(index + 1).padStart(2, "0")}
                  </div>
                  <div className={styles.glyph} aria-hidden>
                    {index === 0 ? "¥" : index === 1 ? "◔" : "⧉"}
                  </div>
                  <h3>{card.title}</h3>
                  <p>
                    <Segments parts={card.body} />
                  </p>
                </div>
              ))
            }
          </Bi>
        </div>
      </section>

      <section className={`${styles.sec} ${styles.afterSec}`} id="service">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            CUSTOMER SERVICE
          </div>
          <Bi as="h2" className={styles.secH}>
            {(copy) => copy.service.h}
          </Bi>
          <Bi as="p" className={styles.secP}>
            {(copy) => copy.service.p}
          </Bi>
          <Bi as="div" className={styles.miniCards}>
            {(copy) =>
              copy.service.cards.map((card) => (
                <div className={styles.miniCard} key={card.title}>
                  <h3>{card.title}</h3>
                  <p>{card.body}</p>
                </div>
              ))
            }
          </Bi>
          <Bi as="div" className={styles.loopCtas}>
            {(copy) => (
              <a className={`${styles.btn} ${styles.btnPrimary}`} href="#demo">
                {copy.service.cta}
              </a>
            )}
          </Bi>
        </div>
      </section>

      <section className={styles.boundsSec}>
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            HARD BOUNDS
          </div>
          <Bi>
            {(copy) => (
              <h2 className={styles.secH} style={{ fontSize: 26 }}>
                {copy.bounds.h}
              </h2>
            )}
          </Bi>
          <Bi>
            {(copy) => (
              <div
                className={`${styles.bounds} ${styles.mono}`}
                style={{ marginTop: 28 }}
              >
                {copy.bounds.items.map((item) => (
                  <div className={styles.bound} key={item.k}>
                    <span className={styles.v}>{item.v}</span>
                    <span className={styles.k}>{item.k}</span>
                  </div>
                ))}
              </div>
            )}
          </Bi>
        </div>
      </section>
      </main>

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
              <Bi>{(copy) => <a href="#demo">{copy.footer.demo}</a>}</Bi>
              <Bi>
                {(copy) => (
                  <a href="https://github.com/Bensonluo/rag_chatbot">
                    {copy.footer.code}
                  </a>
                )}
              </Bi>
              <Bi as="div" className={`${styles.built} ${styles.mono}`}>
                {(copy) => copy.footer.builtBy}
              </Bi>
            </div>
          </div>
          <div className={`${styles.legal} ${styles.mono}`}>
            <Bi>{(copy) => <span>{copy.footer.legal}</span>}</Bi>
            <span>github.com/Bensonluo/rag_chatbot</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
