"""MCP tool facade (Phase C slice 1) — read-only remote tools for the agent loop.

Industry pattern (Shopify ``get_order_status`` / Dynamics 365 Commerce MCP,
2026 spec): a commerce bot's tool layer grows beyond built-in mocks by
importing tools described in the MCP shape — ``name`` / ``description`` /
``inputSchema`` (JSON Schema, camelCase per the official tools spec) — and
proxying invocations to the remote server over stateless JSON-RPC
``tools/call`` (2026-07-28 spec: no session pinning, full context per
request). Imported tools flow through the existing ``ToolRegistry``: the
agent loop sees them as ordinary function-calling tools, and all local
disciplines (step budget, confirmation gate, claim gate) keep applying.

Safety posture, matching the research consensus (read-only facade first,
money moves only via human-confirmed surfaces):

- **Read-only gate.** Only manifest entries whose ``annotations``
  positively declare ``readOnlyHint: true`` (and never
  ``destructiveHint: true``) are admitted. The spec treats server
  annotations as untrusted; here the *manifest is the operator's own
  config*, so it is the trust boundary — write tools stay out until a
  later slice wires remote confirmation flows.
- **Identity containment.** ``ToolRegistry.execute`` injects the
  authenticated ``user_id`` for local authorization; this layer strips
  it before the payload leaves the process. Internal identity never
  crosses the boundary uninvited.
- **Never-fail-chat.** A dead or slow remote server degrades to a
  structured ``ToolResult`` failure the agent loop can answer around;
  nothing raises into the dialogue graph.

Manifest format (a JSON array of entries; ``MCP_TOOLS_MANIFEST`` setting
points at the file, empty/unset = no remote tools):

.. code-block:: json

    [
      {
        "server": "logistics",
        "url": "https://mcp.internal.example/logistics",
        "name": "get_delivery_eta",
        "description": "查询订单预计送达时间（外部物流系统）。",
        "inputSchema": {
          "type": "object",
          "properties": {"order_id": {"type": "string"}},
          "required": ["order_id"]
        },
        "annotations": {"readOnlyHint": true}
      }
    ]
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from app.services.dialogue.tools import ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)

# Monotonic JSON-RPC request ids — stateless protocol, ids are per-request.
_REQUEST_IDS = itertools.count(1)

# Keys never forwarded to a remote server. ``user_id`` is injected
# server-side for local authorization and is internal identity.
_STRIP_ARG_KEYS = ("user_id",)


class ToolInvocationError(Exception):
    """The remote server answered with an error payload (tools/call
    ``isError`` or JSON-RPC ``error``) — degrades to a failed ToolResult."""


class RemoteToolClient(Protocol):
    """Transport seam: POST one JSON-RPC tools/call, return the decoded body."""

    async def call(
        self, url: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]: ...  # pragma: no cover


class HttpxRemoteToolClient:
    """Production transport over httpx.

    A client per call keeps factory instantiation free of connection
    lifecycle concerns (no sockets bound to a build-time event loop).
    Remote tools are optional, low-QPS lookups; a pooled client is a
    later optimization if volume justifies it.
    """

    async def call(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return body


def load_mcp_tools(
    registry: ToolRegistry,
    manifest_path: str | Path,
    client: RemoteToolClient,
    *,
    timeout: float = 5.0,
) -> int:
    """Import read-only tools from an MCP-shaped manifest into ``registry``.

    Malformed entries and manifest-level failures are loud (logged) but
    never fatal — a bad manifest must not take the chat pipeline down.
    Returns the number of tools registered.
    """
    path = Path(manifest_path)
    try:
        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("MCP tool manifest unreadable (%s); no remote tools loaded", exc)
        return 0

    if not isinstance(raw, list):
        logger.warning("MCP tool manifest is not a JSON array; no remote tools loaded")
        return 0

    registered = 0
    for entry in raw:
        if _register_entry(registry, entry, client, timeout=timeout):
            registered += 1
    if registered:
        logger.info("Loaded %d MCP remote tool(s) from %s", registered, path)
    return registered


def _register_entry(
    registry: ToolRegistry,
    entry: dict[str, Any],
    client: RemoteToolClient,
    *,
    timeout: float,
) -> bool:
    name = str(entry.get("name") or "").strip()
    url = str(entry.get("url") or "").strip()
    description = str(entry.get("description") or "").strip()
    input_schema = entry.get("inputSchema")

    if not name or not url or not description or not isinstance(input_schema, dict):
        logger.warning("Skipping malformed MCP tool entry (name=%s): missing required fields", name)
        return False

    annotations = entry.get("annotations")
    if not isinstance(annotations, dict) or annotations.get("readOnlyHint") is not True:
        logger.warning(
            "Skipping MCP tool %s: not positively read-only (readOnlyHint missing/false) — "
            "write tools are not admitted by the facade yet",
            name,
        )
        return False
    if annotations.get("destructiveHint") is True:
        logger.warning("Skipping MCP tool %s: destructiveHint is set", name)
        return False

    if registry.get_tool_by_name(name) is not None:
        logger.warning(
            "Skipping MCP tool %s: name collides with an existing tool (built-ins win)", name
        )
        return False

    registry.register(
        ToolDefinition(
            name=name,
            # Registry key only — never a dialogue intent, so the slot
            # pipeline cannot route here; agent-loop reach is explicit.
            intent=f"mcp:{name}",
            description=description,
            required_slots=list(input_schema.get("required", [])),
            handler=_make_remote_handler(url, name, client, timeout=timeout),
            # Read-only gate already ran; confirmation flow never applies.
            requires_confirmation=False,
            parameters_schema=input_schema,
        )
    )
    return True


def _make_remote_handler(
    url: str,
    name: str,
    client: RemoteToolClient,
    *,
    timeout: float,
) -> Any:
    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        payload = _build_call_payload(name, args)
        try:
            response = await client.call(url, payload, timeout)
            return _extract_result(response)
        except ToolInvocationError:
            raise  # remote said no — surface its reason verbatim
        except Exception as exc:
            # Transport failure (timeout / connect / 5xx): degrade to a
            # user-facing message, never a raw stack detail.
            logger.warning("MCP tool %s invocation failed: %s", name, exc)
            raise ToolInvocationError("工具服务暂不可用，请稍后再试或转人工客服") from exc

    return handler


def _build_call_payload(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """One stateless JSON-RPC 2.0 ``tools/call`` request.

    Internal identity keys are stripped before the payload leaves the
    process (identity containment).
    """
    arguments = {key: value for key, value in args.items() if key not in _STRIP_ARG_KEYS}
    return {
        "jsonrpc": "2.0",
        "id": next(_REQUEST_IDS),
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _extract_result(response: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a tools/call response into a plain result dict.

    Structured content wins (spec-recommended for machine consumers);
    otherwise text content items are joined into ``{"text": ...}``.
    ``isError`` and JSON-RPC ``error`` bodies raise ``ToolInvocationError``
    so ``ToolRegistry.execute`` degrades them into failed results.
    """
    error = response.get("error")
    if isinstance(error, dict):
        raise ToolInvocationError(str(error.get("message") or "remote tool call failed"))

    result = response.get("result")
    if not isinstance(result, dict):
        raise ToolInvocationError("malformed tools/call response: no result object")

    if result.get("isError") is True:
        raise ToolInvocationError(_content_text(result) or "remote tool reported an error")

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    text = _content_text(result)
    if text:
        return {"text": text}
    raise ToolInvocationError("malformed tools/call response: no content")


def _content_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    texts = [
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(text for text in texts if text)
