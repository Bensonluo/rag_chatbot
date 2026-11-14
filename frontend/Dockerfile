# Agent Store branding page — static export served by nginx with a same-origin
# /api proxy to the backend, so the page works with NEXT_PUBLIC_API_URL empty.
#
# BASE_PATH (default "") serves at the site root (bundled full-stack compose,
# self-hosters). BASE_PATH=/store mounts every asset and route under /store for
# deployments where an edge nginx routes a path prefix to this container.
FROM node:22-alpine AS build
ARG BASE_PATH=""
ENV NEXT_PUBLIC_BASE_PATH=${BASE_PATH}
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS serve
ARG BASE_PATH=""
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/out /tmp/out
RUN if [ -n "$BASE_PATH" ]; then \
      mkdir -p "/usr/share/nginx/html$BASE_PATH" && \
      cp -a /tmp/out/. "/usr/share/nginx/html$BASE_PATH/" && \
      # error_page 404 points at the root absolute path; keep a copy
      # there so the not-found page works under a path-prefix mount too.
      cp -a /tmp/out/404 /usr/share/nginx/html/404; \
    else \
      cp -a /tmp/out/. /usr/share/nginx/html/; \
    fi
EXPOSE 80
