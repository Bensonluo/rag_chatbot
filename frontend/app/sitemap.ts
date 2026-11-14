import type { MetadataRoute } from "next";

// Static export requires metadata route handlers to be explicitly static.
export const dynamic = "force-static";

/* Single-page static export: the landing page (with the live guest chat)
 * is the only indexable URL. NEXT_PUBLIC_SITE_URL / NEXT_PUBLIC_BASE_PATH
 * are the same build-time knobs next.config.mjs uses for the mount point. */
export default function sitemap(): MetadataRoute.Sitemap {
  const site = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
  return [
    {
      url: `${site}${base}/`,
      lastModified: new Date(),
      changeFrequency: "monthly",
      priority: 1,
    },
  ];
}
