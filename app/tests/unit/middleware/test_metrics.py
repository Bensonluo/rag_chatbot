"""Tests for Prometheus metrics middleware"""

from collections.abc import MutableMapping
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY


async def _receive() -> MutableMapping[str, Any]:
    """No-op ASGI receive channel — these tests never read a request body."""
    return {}


async def _ok_call_next(request):
    """Minimal call_next returning a 200, for dispatch() tests."""
    return JSONResponse(content={"status": "ok"})


class TestPrometheusMetrics:
    """Test Prometheus metrics collection"""

    def test_metrics_initialization(self):
        """Test metrics middleware initialization"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()

        # Act
        middleware = PrometheusMiddleware(app)

        # Assert
        assert middleware.app == app
        assert middleware.app_name == "rag_chatbot"

    def test_metrics_with_custom_app_name(self):
        """Test metrics with custom app name"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()

        # Act
        middleware = PrometheusMiddleware(app, app_name="custom_app")

        # Assert
        assert middleware.app_name == "custom_app"

    @pytest.mark.asyncio
    async def test_tracks_request_count(self):
        """Test tracking request count"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=_receive,
        )

        # Act - Make multiple requests
        await middleware.dispatch(request, _ok_call_next)
        await middleware.dispatch(request, _ok_call_next)

        # Assert - Should track count (public REGISTRY API, not internals)
        assert (
            REGISTRY.get_sample_value(
                "http_requests_total",
                {"method": "GET", "endpoint": "/test", "status": "200"},
            )
            == 2
        )

    @pytest.mark.asyncio
    async def test_tracks_request_latency(self):
        """Test tracking request latency"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": "/api/v1/chat",
                "headers": [],
                "query_string": b"",
            },
            receive=_receive,
        )

        # Act
        await middleware.dispatch(request, _ok_call_next)

        # Assert - Should track latency histogram
        samples = list(middleware.request_latency.collect())[0].samples
        assert len(samples) > 0

    @pytest.mark.asyncio
    async def test_tracks_active_requests(self):
        """Test tracking active requests (gauge)"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=_receive,
        )

        # Act
        import asyncio

        task = asyncio.create_task(middleware.dispatch(request, _ok_call_next))
        await asyncio.sleep(0.01)  # Let it start
        await task

        # Assert - Should track gauge (0 when complete)
        assert REGISTRY.get_sample_value("http_requests_active") == 0

    @pytest.mark.asyncio
    async def test_tracks_errors_by_status(self):
        """Test tracking errors by status code"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        async def failing_call_next(request):
            return JSONResponse(content={"error": "Not found"}, status_code=404)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/notfound",
                "headers": [],
                "query_string": b"",
            },
            receive=_receive,
        )

        # Act
        await middleware.dispatch(request, failing_call_next)

        # Assert - Should track 404 errors
        assert (
            REGISTRY.get_sample_value(
                "http_requests_total",
                {"method": "GET", "endpoint": "/notfound", "status": "404"},
            )
            == 1
        )

    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware, metrics_endpoint

        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)
        app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])

        # Act - Get metrics
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.get("/metrics")

        # Assert
        assert response.status_code == 200
        # Should have Prometheus format
        assert "http_requests_total" in response.text or "request_count" in response.text


class TestMetricsFormats:
    """Test metrics output formats"""

    def test_prometheus_format(self):
        """Test Prometheus text format"""
        # Arrange
        from prometheus_client import Counter

        counter = Counter("test_requests_total", "Test requests", ["method"])

        # Act
        counter.labels(method="GET").inc()

        # Assert
        from prometheus_client import exposition

        output = exposition.generate_latest(counter).decode()
        assert "test_requests_total" in output

    def test_histogram_buckets(self):
        """Test histogram has proper buckets"""
        # Arrange
        from prometheus_client import Histogram

        histogram = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

        # Act - Observe some latencies
        histogram.labels(endpoint="/test").observe(0.1)
        histogram.labels(endpoint="/test").observe(0.5)
        histogram.labels(endpoint="/test").observe(1.5)

        # Assert
        samples = list(histogram.collect())[0].samples
        # Should have bucket samples
        bucket_samples = [s for s in samples if s.name.endswith("_bucket")]
        assert len(bucket_samples) > 0


class TestMetricsLabels:
    """Test metrics labeling"""

    def test_default_labels(self):
        """Test default metric labels"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        # Assert - Should have standard labels
        assert hasattr(middleware, "request_count")
        assert hasattr(middleware, "request_latency")
        assert hasattr(middleware, "active_requests")

    def test_custom_labels_allowed(self):
        """Test custom labels can be added"""
        # Arrange & Act
        from prometheus_client import Counter

        counter = Counter("custom_metric", "Custom metric", ["label1", "label2"])

        # Assert
        counter.labels(label1="value1", label2="value2").inc()
        # Counter samples are exposed with the _total suffix
        assert (
            REGISTRY.get_sample_value(
                "custom_metric_total", {"label1": "value1", "label2": "value2"}
            )
            == 1
        )

    async def _mock_call_next(self, request):
        """Mock call_next function"""
        return JSONResponse(content={"status": "ok"})


class TestMetricsIntegration:
    """Test metrics integration"""

    @pytest.mark.asyncio
    async def test_full_request_tracked(self):
        """Test complete request is tracked"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
                "app": app,
            },
            receive=_receive,
        )

        # Act
        await middleware.dispatch(request, _ok_call_next)

        # Assert - Metrics should be recorded
        total = REGISTRY.get_sample_value(
            "http_requests_total",
            {"method": "GET", "endpoint": "/test", "status": "200"},
        )
        assert total is not None
        assert total > 0
