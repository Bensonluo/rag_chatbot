"""Outbound HTTP pool contract for the default LLM provider.

Streaming completions hold one connection for the full SSE duration
(seconds). httpx's default pool caps at 100 connections, which the
per-node template load (~30-50 QPS x 3-8s in flight) exceeds — the
pool then starves into ``PoolTimeout`` and turns healthy provider
capacity into client-side 5xx. The client must therefore carry an
explicitly sized pool.
"""

from unittest.mock import patch

from app.services.llm.glm_client import GLMClient


class TestGLMClientPoolLimits:
    def test_streaming_pool_exceeds_httpx_default_ceiling(self):
        with patch("app.services.llm.glm_client.httpx.AsyncClient") as ctor:
            GLMClient(api_key="id.secret")

        limits = ctor.call_args.kwargs["limits"]
        # > httpx default 100: concurrency must surface as provider
        # backpressure, not client-side pool starvation.
        assert limits.max_connections > 100
        # Keepalive well below the ceiling so idle streams don't pin
        # the whole budget.
        assert 0 < limits.max_keepalive_connections < limits.max_connections
