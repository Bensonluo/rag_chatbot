import type { MetadataRoute } from "next";

// Static export requires metadata route handlers to be explicitly static.
export const dynamic = "force-static";

/* The product welcomes agents — it is one. An explicit allow-all robots.txt
 * (plus the sitemap pointer) removes ambiguity for search and AI crawlers
 * alike; nothing on this demo store is secret. */
export default function robots(): MetadataRoute.Robots {
  const site = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  return {
    rules: [{ userAgent: "*", allow: "/" }],
    sitemap: `${site}${base}/sitemap.xml`,
  };
}
