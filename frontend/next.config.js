/** @type {import('next').NextConfig} */
// NEXT_PUBLIC_BASE_PATH mounts the static export under a path prefix, e.g.
// "/store" when the edge nginx routes benluo.art/store/ to this app while the
// portfolio still owns the root. Empty (default) serves at the site root —
// the bundled full-stack compose and any self-hoster use that.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig = {
  output: "export",
  basePath,
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

module.exports = nextConfig;
