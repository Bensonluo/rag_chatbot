'use client';

/** The Agent Store landing page body. A client component so every string can
 * follow the visitor's locale (`lib/i18n/context.tsx`) — the static export
 * renders the zh tree, then hydrates to the stored / system locale. The
 * server-exported `page.tsx` keeps the metadata and shells out to this. */

import { Fragment } from "react";
import { ChatInterfaceUpdated } from "@/components/ChatInterfaceUpdated";
import { useLanguage } from "@/lib/i18n/context";
import { storeCopy, type CopySegment } from "@/lib/i18n/store-copy";
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

export function StoreLanding() {
  const { locale, setLocale } = useLanguage();
  const copy = storeCopy[locale];

  return (
    <div className={styles.root} lang={locale === "zh" ? "zh-CN" : "en"}>
      <nav className={styles.bar}>
        <div className={`${styles.wrap} ${styles.barIn}`}>
          <div className={styles.brand}>
            <span className={styles.brandDot} aria-hidden />
            Agent&nbsp;<span className={styles.brandStore}>Store</span>
          </div>
          <div className={`${styles.barLinks} ${styles.mono}`}>
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
            {copy.hero.kicker}
          </div>
          <h1>
            Agent <span>Store</span>
          </h1>
          <div className={styles.sub}>{copy.hero.sub}</div>
          <p className={styles.lede}>
            <Segments parts={copy.hero.lede} />
          </p>
          <div className={styles.ctaRow}>
            <a className={`${styles.btn} ${styles.btnPrimary}`} href="#demo">
              {copy.hero.ctaPrimary}
            </a>
            <a
              className={`${styles.btn} ${styles.btnGhost}`}
              href="https://github.com/Bensonluo/rag_chatbot"
            >
              {copy.hero.ctaGhost}
            </a>
          </div>
          <div className={`${styles.heroNote} ${styles.mono}`}>
            {copy.hero.note}
          </div>
        </div>
      </header>

      <div className={styles.flow}>
        <div className={`${styles.wrap} ${styles.flowStack}`}>
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
        </div>
      </div>

      <section className={`${styles.sec} ${styles.demoSec}`} id="demo">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>LIVE DEMO</div>
          <h2 className={styles.secH}>{copy.demo.h}</h2>
          <p className={styles.secP}>{copy.demo.p}</p>

          <div className={styles.chatPanel}>
            <div className={styles.chatHead}>
              <div className={styles.who}>
                <span className={styles.whoDot} aria-hidden />
                {copy.demo.chatHead}
              </div>
              <div className={`${styles.status} ${styles.mono}`}>● online</div>
            </div>
            <div className={`${styles.livePad} dark`}>
              <ChatInterfaceUpdated />
            </div>
          </div>
          <div className={`${styles.demoCap} ${styles.mono}`}>
            {copy.demo.caption}
          </div>
        </div>
      </section>

      <section className={`${styles.sec} ${styles.trustSec}`} id="trust">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            WHY TRUST IT
          </div>
          <h2 className={styles.secH}>{copy.trust.h}</h2>
          <p className={styles.secP}>{copy.trust.p}</p>
          <div className={styles.cards}>
            {copy.trust.cards.map((card, index) => (
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
            ))}
          </div>
        </div>
      </section>

      <section className={`${styles.sec} ${styles.afterSec}`} id="service">
        <div className={styles.wrap}>
          <div className={`${styles.secKicker} ${styles.mono}`}>
            CUSTOMER SERVICE
          </div>
          <h2 className={styles.secH}>{copy.service.h}</h2>
          <p className={styles.secP}>{copy.service.p}</p>
          <div className={styles.miniCards}>
            {copy.service.cards.map((card) => (
              <div className={styles.miniCard} key={card.title}>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </div>
            ))}
          </div>
          <div className={styles.loopCtas}>
            <a className={`${styles.btn} ${styles.btnPrimary}`} href="#demo">
              {copy.service.cta}
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
            {copy.bounds.h}
          </h2>
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
              <a href="#demo">{copy.footer.demo}</a>
              <a href="https://github.com/Bensonluo/rag_chatbot">
                {copy.footer.code}
              </a>
              <div className={`${styles.built} ${styles.mono}`}>
                {copy.footer.builtBy}
              </div>
            </div>
          </div>
          <div className={`${styles.legal} ${styles.mono}`}>
            <span>{copy.footer.legal}</span>
            <span>github.com/Bensonluo/rag_chatbot</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
