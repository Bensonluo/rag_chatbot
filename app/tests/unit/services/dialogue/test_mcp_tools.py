"""MCP tool facade (Phase C slice 1 — read-only remote tools).

Industry pattern (Shopify get_order_status / Dynamics 365 Commerce MCP):
the bot's tool layer grows beyond mocks by importing read-only lookup
tools from an MCP-shaped manifest (``name`` / ``description`` /
``inputSchema``, per the official tools spec) and proxying calls to the
remote server over stateless JSON-RPC ``tools/call``. Write tools are
rejected until a later slice wires remote confirmation flows — the
research consensus is read-only facade first, money moves only via
human-confirmed surfaces.

Two disciplines this suite pins:

- **Never-fail-chat**: a dead/slow remote tool degrades to a structured
  ``ToolResult`` failure the agent loop can answer around — it must
  never raise into the dialogue graph.
- **Identity containment**: the authenticated ``user_id`` injected
  server-side is for local authorization; it is stripped before the
  payload leaves the process. No accidental identity leakage across
  the trust boundary.
"""

import json
from pathlib import Path
from typing import Any

from app.services.dialogue.mcp_tools import (
    ToolInvocationError,
    load_mcp_tools,
)
from app.services.dialogue.tools import ToolRegistry


def _read_only_entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "server": "logistics",
        "url": "https://mcp.internal.example/logistics",
        "name": "get_delivery_eta",
        "description": "查询订单预计送达时间（外部物流系统）。",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        "annotations": {"readOnlyHint": True},
    }
    entry.update(overrides)
    return entry


class FakeClient:
    """Transport seam double — records payloads, returns canned wire responses."""

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None):
        self.response = response or {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "structuredContent": {"eta": "2 天内送达"},
                "isError": False,
            },
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def call(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        if self.error:
            raise self.error
        return self.response


def _write_manifest(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    manifest = tmp_path / "mcp_tools.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    return manifest


class TestManifestLoading:
    async def test_valid_read_only_tool_is_registered(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        client = FakeClient()

        count = load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        assert count == 1
        tool = registry.get_tool_by_name("get_delivery_eta")
        assert tool is not None
        assert tool.intent == "mcp:get_delivery_eta"

    async def test_exported_to_function_schemas_from_inputschem(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), FakeClient())

        schemas = registry.to_function_schemas()
        exported = [s["function"]["name"] for s in schemas]
        assert "get_delivery_eta" in exported
        remote = next(s for s in schemas if s["function"]["name"] == "get_delivery_eta")
        assert remote["function"]["parameters"]["required"] == ["order_id"]

    async def test_required_slots_derive_from_schema(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), FakeClient())

        tool = registry.get_tool_by_name("get_delivery_eta")
        assert tool is not None
        assert tool.required_slots == ["order_id"]

    async def test_remote_tool_never_requires_confirmation(self, tmp_path: Path) -> None:
        """Read-only gate means the confirmation flow can never trigger."""
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), FakeClient())

        tool = registry.get_tool_by_name("get_delivery_eta")
        assert tool is not None
        assert tool.requires_confirmation is False


class TestReadOnlyGate:
    async def test_write_tool_rejected(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        entries = [
            _read_only_entry(
                name="process_remote_refund",
                annotations={"readOnlyHint": False},
            )
        ]

        count = load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient())

        assert count == 0
        assert registry.get_tool_by_name("process_remote_refund") is None

    async def test_missing_annotations_rejected(self, tmp_path: Path) -> None:
        """Annotations are untrusted per the MCP spec; the operator-side
        manifest must positively declare readOnlyHint or the tool stays out."""
        registry = ToolRegistry()
        entries = [_read_only_entry(annotations={})]

        count = load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient())

        assert count == 0

    async def test_destructive_hint_rejects_even_if_read_only_claimed(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        entries = [_read_only_entry(annotations={"readOnlyHint": True, "destructiveHint": True})]

        count = load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient())

        assert count == 0

    async def test_builtin_name_collision_rejected(self, tmp_path: Path) -> None:
        """Built-ins win: a remote tool must never shadow a local tool."""
        from app.services.dialogue.tools import create_default_tool_registry

        registry = create_default_tool_registry()
        entries = [_read_only_entry(name="query_order_status")]

        count = load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient())

        assert count == 0
        # The local mock tool is still the one registered.
        assert registry.get_tool_by_name("query_order_status").intent == "query_order"  # type: ignore[union-attr]


class TestManifestHygiene:
    async def test_malformed_entries_skipped_not_fatal(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        entries = [
            {"description": "no name"},
            _read_only_entry(name="ok_tool"),
            _read_only_entry(name="bad_schema", inputSchema="not-a-dict"),
        ]

        count = load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient())

        assert count == 1
        assert registry.get_tool_by_name("bad_schema") is None
        assert registry.get_tool_by_name("ok_tool") is not None

    async def test_unreadable_manifest_is_loud_but_not_fatal(self, tmp_path: Path) -> None:
        registry = ToolRegistry()

        count = load_mcp_tools(registry, tmp_path / "missing.json", FakeClient())

        assert count == 0
        assert registry.to_function_schemas() == []

    async def test_empty_manifest_registers_nothing(self, tmp_path: Path) -> None:
        registry = ToolRegistry()

        assert load_mcp_tools(registry, _write_manifest(tmp_path, []), FakeClient()) == 0


class TestRemoteExecution:
    async def test_structured_content_returned_on_success(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), FakeClient())

        result = await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD1001"})

        assert result.success
        assert result.data == {"eta": "2 天内送达"}

    async def test_text_content_fallback_when_no_structured(self, tmp_path: Path) -> None:
        client = FakeClient(
            response={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "预计明天送达"}]},
            }
        )
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        result = await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD1001"})

        assert result.success
        assert result.data == {"text": "预计明天送达"}

    async def test_wire_payload_is_a_tools_call_jsonrpc(self, tmp_path: Path) -> None:
        client = FakeClient()
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD1001"})

        (call,) = client.calls
        assert call["url"].endswith("/logistics")
        payload = call["payload"]
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "get_delivery_eta"
        assert payload["params"]["arguments"] == {"order_id": "ORD1001"}

    async def test_is_error_response_degrades_to_failed_tool_result(self, tmp_path: Path) -> None:
        client = FakeClient(
            response={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "订单不存在"}],
                },
            }
        )
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        result = await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD9999"})

        assert not result.success
        assert "订单不存在" in result.message or "暂不可用" in result.message

    async def test_transport_failure_degrades_never_raises(self, tmp_path: Path) -> None:
        client = FakeClient(error=TimeoutError("connect timeout"))
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        result = await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD1001"})

        assert not result.success
        assert "暂不可用" in result.message

    async def test_timeout_is_passed_to_transport(self, tmp_path: Path) -> None:
        client = FakeClient()
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD1001"})

        assert client.calls[0]["timeout"] > 0


class TestIdentityContainment:
    async def test_user_id_never_crosses_the_boundary(self, tmp_path: Path) -> None:
        """The registry injects the authenticated user_id into args for local
        authorization; the remote payload must strip it — internal identity
        never leaves the process uninvited."""
        client = FakeClient()
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD1001"}, user_id=1)

        (call,) = client.calls
        assert "user_id" not in call["payload"]["params"]["arguments"]


class TestSettingsContract:
    def test_mcp_tools_off_by_default(self) -> None:
        """Remote tools are opt-in: a fresh deployment ships the mock
        registry unchanged until an operator points at a manifest."""
        from app.config.settings import Settings

        settings = Settings()
        assert settings.MCP_TOOLS_ENABLED is False
        assert settings.MCP_TOOLS_MANIFEST == ""
        assert settings.MCP_TOOL_TIMEOUT_SECONDS > 0


class TestExtractionContract:
    def test_is_error_raises_tool_invocation_error(self) -> None:
        from app.services.dialogue.mcp_tools import _extract_result

        try:
            _extract_result(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"isError": True, "content": [{"type": "text", "text": "boom"}]},
                }
            )
        except ToolInvocationError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected ToolInvocationError")

    def test_jsonrpc_error_raises_tool_invocation_error(self) -> None:
        from app.services.dialogue.mcp_tools import _extract_result

        try:
            _extract_result({"jsonrpc": "2.0", "id": 1, "error": {"message": "bad params"}})
        except ToolInvocationError as exc:
            assert "bad params" in str(exc)
        else:
            raise AssertionError("expected ToolInvocationError")
