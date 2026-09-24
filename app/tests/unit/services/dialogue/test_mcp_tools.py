"""MCP tool facade (Phase C — read + confirmed-write remote tools).

Industry pattern (Shopify get_order_status / Dynamics 365 Commerce MCP):
the bot's tool layer grows beyond mocks by importing lookup tools from
an MCP-shaped manifest (``name`` / ``description`` / ``inputSchema``,
per the official tools spec) and proxying calls to the remote server
over stateless JSON-RPC ``tools/call``. Read tools register directly;
write tools register with the human confirmation gate forced on and
carry a deterministic idempotency key (SEP-3182 shape) so a
timeout-then-reconfirm flow cannot double-execute server-side.

Disciplines this suite pins:

- **Never-fail-chat**: a dead/slow remote tool degrades to a structured
  ``ToolResult`` failure the agent loop can answer around — it must
  never raise into the dialogue graph.
- **Identity containment**: the authenticated ``user_id`` injected
  server-side is for local authorization; it (and any model-forged
  ``idempotency_key``) is stripped before the payload leaves the
  process. No accidental identity leakage across the trust boundary.
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
    # (Write tools are no longer rejected outright — slice 2 admits them
    # with forced confirmation; see TestWriteToolAdmission.)

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


class TestWriteToolAdmission:
    """Remote write tools (Phase C slice 2): admitted, but the
    confirmation gate is forced on — the manifest can never opt a write
    tool out of human confirmation (Shopify Checkout / UCP pattern:
    money moves only via confirmed surfaces)."""

    def _write_entry(self, **overrides: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "server": "order-system",
            "url": "https://mcp.internal.example/orders",
            "name": "cancel_remote_order",
            "description": "取消尚未发货的订单。执行前必须向用户确认。",
            "inputSchema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            "annotations": {"readOnlyHint": False},
        }
        entry.update(overrides)
        return entry

    async def test_explicit_write_tool_admitted_with_forced_confirmation(
        self, tmp_path: Path
    ) -> None:
        registry = ToolRegistry()
        entries = [self._write_entry()]

        count = load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient())

        assert count == 1
        tool = registry.get_tool_by_name("cancel_remote_order")
        assert tool is not None
        assert tool.requires_confirmation is True

    async def test_manifest_cannot_opt_a_write_tool_out_of_confirmation(
        self, tmp_path: Path
    ) -> None:
        """Even a manifest entry that (wrongly) claims confirmable=False
        keeps the forced confirmation — server-side policy wins."""
        registry = ToolRegistry()
        entries = [self._write_entry(requiresConfirmation=False)]

        load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient())

        tool = registry.get_tool_by_name("cancel_remote_order")
        assert tool is not None
        assert tool.requires_confirmation is True

    async def test_missing_annotations_still_rejected(self, tmp_path: Path) -> None:
        """Ambiguity is not a write admission: annotations missing → out."""
        registry = ToolRegistry()
        entries = [self._write_entry(annotations={})]

        assert load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient()) == 0

    async def test_destructive_write_rejected(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        entries = [self._write_entry(annotations={"readOnlyHint": False, "destructiveHint": True})]

        assert load_mcp_tools(registry, _write_manifest(tmp_path, entries), FakeClient()) == 0

    async def test_write_tool_reaches_agent_function_schemas(self, tmp_path: Path) -> None:
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [self._write_entry()]), FakeClient())

        exported = [s["function"]["name"] for s in registry.to_function_schemas()]
        assert "cancel_remote_order" in exported


class TestIdempotencyKey:
    """Write calls carry a deterministic idempotency key (SEP-3182 shape:
    ``params.idempotencyKey``, sibling of ``arguments``).

    Determinism is the whole point: the same logical operation — same
    user, same tool, same model-visible args — derives the same key, so
    a timeout-then-reconfirm flow cannot double-execute server-side.
    Different args or a different user derive different keys."""

    def _write_entry(self) -> dict[str, Any]:
        return {
            "server": "order-system",
            "url": "https://mcp.internal.example/orders",
            "name": "cancel_remote_order",
            "description": "取消尚未发货的订单。执行前必须向用户确认。",
            "inputSchema": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            "annotations": {"readOnlyHint": False},
        }

    def _load_write_registry(self, tmp_path: Path, client: FakeClient) -> ToolRegistry:
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [self._write_entry()]), client)
        return registry

    async def test_write_call_carries_idempotency_key(self, tmp_path: Path) -> None:
        client = FakeClient()
        registry = self._load_write_registry(tmp_path, client)

        await registry.execute("mcp:cancel_remote_order", {"order_id": "ORD1001"}, user_id=1)

        (call,) = client.calls
        assert "idempotencyKey" in call["payload"]["params"]

    async def test_key_stable_across_reconfirmations(self, tmp_path: Path) -> None:
        """Timeout → user re-confirms the same logical op → same key →
        the remote server can dedup instead of double-cancelling."""
        client = FakeClient()
        registry = self._load_write_registry(tmp_path, client)

        await registry.execute("mcp:cancel_remote_order", {"order_id": "ORD1001"}, user_id=1)
        await registry.execute("mcp:cancel_remote_order", {"order_id": "ORD1001"}, user_id=1)

        key1 = client.calls[0]["payload"]["params"]["idempotencyKey"]
        key2 = client.calls[1]["payload"]["params"]["idempotencyKey"]
        assert key1 == key2

    async def test_different_args_derive_different_keys(self, tmp_path: Path) -> None:
        client = FakeClient()
        registry = self._load_write_registry(tmp_path, client)

        await registry.execute("mcp:cancel_remote_order", {"order_id": "ORD1001"}, user_id=1)
        await registry.execute("mcp:cancel_remote_order", {"order_id": "ORD2001"}, user_id=1)

        key1 = client.calls[0]["payload"]["params"]["idempotencyKey"]
        key2 = client.calls[1]["payload"]["params"]["idempotencyKey"]
        assert key1 != key2

    async def test_different_users_derive_different_keys(self, tmp_path: Path) -> None:
        client = FakeClient()
        registry = self._load_write_registry(tmp_path, client)

        await registry.execute("mcp:cancel_remote_order", {"order_id": "ORD1001"}, user_id=1)
        await registry.execute("mcp:cancel_remote_order", {"order_id": "ORD1001"}, user_id=2)

        key1 = client.calls[0]["payload"]["params"]["idempotencyKey"]
        key2 = client.calls[1]["payload"]["params"]["idempotencyKey"]
        assert key1 != key2

    async def test_read_call_carries_no_idempotency_key(self, tmp_path: Path) -> None:
        client = FakeClient()
        registry = ToolRegistry()
        load_mcp_tools(registry, _write_manifest(tmp_path, [_read_only_entry()]), client)

        await registry.execute("mcp:get_delivery_eta", {"order_id": "ORD1001"})

        (call,) = client.calls
        assert "idempotencyKey" not in call["payload"]["params"]

    async def test_forged_key_args_never_reach_the_wire(self, tmp_path: Path) -> None:
        """A model (or injected prompt) supplying ``idempotency_key`` in
        tool-call args must not influence the derived key — server-side
        derivation over model-visible args only."""
        client = FakeClient()
        registry = self._load_write_registry(tmp_path, client)

        await registry.execute(
            "mcp:cancel_remote_order",
            {"order_id": "ORD1001", "idempotency_key": "forged"},
            user_id=1,
        )

        (call,) = client.calls
        assert call["payload"]["params"]["idempotencyKey"] != "forged"
        assert "idempotency_key" not in call["payload"]["params"]["arguments"]


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
