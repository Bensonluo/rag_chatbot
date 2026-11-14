import Link from "next/link";
import styles from "./not-found.module.css";

// Bilingual by design: a not-found page has no language toggle of its own,
// and a mistyped URL can land anywhere — both languages are always shown
// so no visitor gets a dead end in the wrong language.
export default function NotFound() {
  return (
    <main className={styles.main}>
      <p className={styles.code} aria-hidden="true">
        404
      </p>
      <h1 className={styles.title}>这一页不存在 · This page could not be found</h1>
      <p className={styles.hint}>
        链接可能已失效或输错了。· The link may be outdated or mistyped.
      </p>
      <Link href="/" className={styles.home}>
        回到 Agent Store · Back to the Agent Store
      </Link>
    </main>
  );
}
