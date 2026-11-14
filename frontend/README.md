# Agent Store Frontend

The standalone Agent Store branding page — the product's public frontend. A
single-page Next.js static export (chat included) that talks to the Agent Store
backend API. This is the canonical home of the page; the author's portfolio
site keeps its own separate project-detail variant.

## Run the full stack (one command)

From the repository root:

```bash
cp .env.example .env        # set GLM_API_KEY
docker compose -f docker-compose.fullstack.yml up -d
# Page:  http://localhost:3000
# API:   http://localhost:8000/docs
```

The bundled `frontend/nginx.conf` serves the static export and proxies
`/api/...` to the backend same-origin, so the page works out of the box with
`NEXT_PUBLIC_API_URL` empty.

## Development

```bash
cd frontend
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                  # http://localhost:3000
```

`npm run build` produces the static export in `out/`; `npm run type-check`
runs TypeScript. The API base URL is `NEXT_PUBLIC_API_URL` (empty = same
origin, resolved at build time — rebuild or set it in `.env.local` for dev).

## Deployment notes

Any static file server + reverse proxy works: serve `out/` and proxy
`/api/` to the backend (SSE enabled). For a different public origin, set
`NEXT_PUBLIC_SITE_URL` at build time for correct OpenGraph URLs.

### Not-found behavior

The bundled nginx configs return a **hard 404** — `try_files ... =404` plus
`error_page 404 /404/index.html` serving the generated bilingual not-found
page. If you serve `out/` with your own server, replicate that: never fall
back to serving `index.html` with a 200 for unknown paths (a soft-404 breaks
link checkers and search indexing). Under a path prefix (e.g. `/store`),
also copy `out/404/` to the prefix root so `error_page` can find it.

## Layout

- `app/page.tsx` (metadata) + `app/store-landing.tsx` +
  `app/page.module.css` — the branding page (hero, flow, live demo, trust
  cards, service, hard bounds, footer), bilingual via
  `lib/i18n/store-copy.ts` (zh/en dictionary) and the shared locale context
- `components/ChatInterfaceUpdated.tsx` — the guest chat (server-authoritative
  approval/price-watch cards, execution chain, citations)
- `lib/api.ts`, `lib/guest-api.ts` — typed API clients (`NEXT_PUBLIC_API_URL`)
- `lib/i18n/` — zh/en language context
