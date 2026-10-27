"""Synthetic CLI, identity and disclosure attacks against the live probe.

All traffic uses MockTransport. No environment value is read except the
synthetic token variables explicitly installed by these tests.
"""

import importlib.util
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[4]
_SECRET = "SYNTHETIC_PRIVATE_RESPONSE_DO_NOT_PRINT"
_TOKEN_ENV = "RAG_PROBE_SECURITY_TEST_TOKEN"
_TOKEN = "synthetic.access.token"


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location(
        "probe_live_security_under_test", _ROOT / "scripts" / "probe_live.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://synthetic-probe.invalid", transport=httpx.MockTransport(handler)
    )


def _case(name: str = "synthetic", check: Any = None) -> Any:
    return probe.ProbeCase(name, [probe.Turn(name, check or (lambda reply: None), "safe label")])


def _reply(session_id: int, content: str = "Hello.") -> dict[str, Any]:
    return {
        "content": content,
        "session_id": session_id,
        "intent": "greeting",
        "metadata": {"pending_slots": [], "filled_slots": {}},
        "dialogue_state": {"phase": "greeting", "pending_slots": [], "filled_slots": {}},
        "sources": None,
        "citations": [],
    }


def _identity(user_id: Any = 7, *, admin: bool = False) -> dict[str, Any]:
    return {
        "id": user_id,
        "email": "synthetic@probe.invalid",
        "full_name": None,
        "is_active": True,
        "is_admin": admin,
        "created_at": "2026-10-28T00:00:00+00:00",
    }


def _session(session_id: Any = 101, user_id: Any = 7) -> dict[str, Any]:
    return {
        "id": session_id,
        "user_id": user_id,
        "title": "Synthetic probe",
        "memory_type": "sliding_window",
        "context_window": 10,
        "message_count": 0,
        "created_at": "2026-10-28T00:00:00+00:00",
        "updated_at": "2026-10-28T00:00:00+00:00",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["--timeout", "nan"],
        ["--timeout", "inf"],
        ["--timeout", "0"],
        ["--timeout", "-1"],
        ["--pace", "nan"],
        ["--pace", "inf"],
        ["--pace", "-1"],
    ],
)
async def test_invalid_numeric_configuration_makes_no_requests(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(probe, "build_cases", lambda: [_case()])
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    async with _client(handler) as client:
        try:
            code = await probe.main(argv, client=client)
        except SystemExit as exc:
            code = exc.code
    assert code != 0
    assert requests == []


@pytest.mark.parametrize("value", [None, "", " \t "])
async def test_missing_or_empty_token_variable_stops_before_requests(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(_TOKEN_ENV, value)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    async with _client(handler) as client:
        try:
            code = await probe.main(["--auth-token-env", _TOKEN_ENV], client=client)
        except SystemExit as exc:
            code = exc.code
    assert code != 0
    assert requests == []


async def test_required_ticket_evidence_cannot_silently_run_without_observer() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    async with _client(handler) as client:
        with pytest.raises(SystemExit) as rejected:
            await probe.main(["--require-ticket-evidence"], client=client)
    assert rejected.value.code != 0
    assert requests == []


@pytest.mark.parametrize("kind", ["invalid_200_content", "http_error_body", "transport_error"])
async def test_untrusted_response_and_error_text_are_never_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    monkeypatch.setattr(probe, "build_cases", lambda: [_case(check=probe.is_ticket())])
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if kind == "transport_error":
            raise httpx.ConnectError(_SECRET, request=request)
        if kind == "http_error_body":
            return httpx.Response(500, text=_SECRET)
        body = json.loads(request.content)
        return httpx.Response(200, json=_reply(body["session_id"], _SECRET))

    async with _client(handler) as client:
        code = await probe.main([], client=client)
    assert code == 1
    assert requests
    output = capsys.readouterr()
    assert _SECRET not in output.out
    assert _SECRET not in output.err
    assert "FAIL" in output.out


@pytest.mark.parametrize("malformed", [b"not-json", b"[]", b"null", b'"string"', b"123"])
async def test_malformed_200_fails_case_and_later_case_still_runs(
    monkeypatch: pytest.MonkeyPatch, malformed: bytes
) -> None:
    monkeypatch.setattr(probe, "build_cases", lambda: [_case("malformed"), _case("healthy")])
    sent = []
    history: dict[int, tuple[str, dict[str, Any]]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v1/chat/history":
            session_id = int(request.url.params["session_id"])
            message, reply = history[session_id]
            return httpx.Response(
                200,
                json={
                    "session_id": session_id,
                    "messages": [
                        {"role": "user", "content": message, "sources": None, "citations": []},
                        {
                            "role": "assistant",
                            "content": reply["content"],
                            "sources": None,
                            "citations": [],
                        },
                    ],
                },
            )
        body = json.loads(request.content)
        sent.append(body["message"])
        if body["message"] == "malformed":
            return httpx.Response(
                200, content=malformed, headers={"Content-Type": "application/json"}
            )
        reply = _reply(body["session_id"])
        history[body["session_id"]] = (body["message"], reply)
        return httpx.Response(200, json=reply)

    async with _client(handler) as client:
        cases = await probe.run_probes(client)
    assert sent == ["malformed", "healthy"]
    assert cases[0].failures
    assert not cases[1].failures


@pytest.mark.parametrize("invalid", ["boolean_identity", "boolean_session", "foreign_owner"])
async def test_authenticated_identity_or_created_session_mismatch_never_sends_chat(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    monkeypatch.setattr(probe, "build_cases", lambda: [_case()])
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append((request.method, request.url.path))
        assert request.headers["Authorization"] == f"Bearer {_TOKEN}"
        if request.url.path == "/api/v1/auth/me":
            return httpx.Response(200, json=_identity(True if invalid == "boolean_identity" else 7))
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            return httpx.Response(
                200,
                json=_session(
                    True if invalid == "boolean_session" else 101,
                    8 if invalid == "foreign_owner" else 7,
                ),
            )
        return httpx.Response(500)

    async with _client(handler) as client:
        code = await probe.main(["--auth-token-env", _TOKEN_ENV], client=client)
    assert code == 1
    assert ("GET", "/api/v1/auth/me") in paths
    assert ("POST", "/api/v1/chat") not in paths
    if invalid == "boolean_identity":
        assert ("POST", "/api/v1/sessions") not in paths
    else:
        assert ("POST", "/api/v1/sessions") in paths


async def test_observer_is_read_only_and_rejects_ticket_from_another_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "build_cases", lambda: [_case(check=probe.is_ticket())])
    session_ids = []
    observer_requests = []

    def chat_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        session_ids.append(body["session_id"])
        reply = _reply(body["session_id"], "Your request is recorded (ticket #12).")
        reply.update(
            intent="handoff",
            metadata={"handoff_outcome": "queued", "handoff_ticket_id": 12},
            dialogue_state={"phase": "handoff", "pending_slots": [], "filled_slots": {}},
        )
        return httpx.Response(200, json=reply)

    def observer_handler(request: httpx.Request) -> httpx.Response:
        observer_requests.append((request.method, request.url.path))
        assert request.method == "GET", "ticket observation must never claim or resolve tickets"
        if request.url.path == "/api/v1/auth/me":
            return httpx.Response(200, json=_identity(11, admin=True))
        return httpx.Response(
            200,
            json={
                "tickets": [
                    {
                        "id": 12,
                        "session_id": session_ids[-1] + 1,
                        "user_id": 0,
                        "status": "open",
                        "assigned_to": None,
                        "claimed_at": None,
                    }
                ],
                "count": 1,
            },
        )

    async with _client(chat_handler) as client, _client(observer_handler) as observer:
        code = await probe.main(
            ["--require-ticket-evidence"], client=client, ticket_client=observer
        )
    assert code == 1
    assert session_ids
    assert any(path == "/api/v1/handoff/tickets" for _, path in observer_requests)


async def test_user_and_staff_tokens_cannot_share_the_same_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staff_env = "RAG_PROBE_SECURITY_TEST_STAFF_TOKEN"
    monkeypatch.setenv(_TOKEN_ENV, _TOKEN)
    monkeypatch.setenv(staff_env, "synthetic.staff.token")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    async with _client(handler) as client:
        client.headers["Authorization"] = "Bearer original.synthetic.token"
        with pytest.raises(SystemExit) as rejected:
            await probe.main(
                ["--auth-token-env", _TOKEN_ENV, "--ticket-token-env", staff_env],
                client=client,
                ticket_client=client,
            )
        assert client.headers["Authorization"] == "Bearer original.synthetic.token"
    assert rejected.value.code != 0
    assert requests == []


@pytest.mark.parametrize("source_type", [[], {}], ids=["array_type", "object_type"])
async def test_nested_malformed_citation_fails_without_aborting_other_cases(
    monkeypatch: pytest.MonkeyPatch, source_type: object
) -> None:
    monkeypatch.setattr(
        probe,
        "build_cases",
        lambda: [
            _case("bad-citation", probe.is_knowledge_answer(probe._EN_POLICY_EVIDENCE)),
            _case("healthy"),
        ],
    )
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body["message"])
        reply = _reply(body["session_id"])
        if body["message"] == "bad-citation":
            reply.update(
                content="Returns within 7 days.",
                intent="policy",
                sources=["synthetic-policy"],
                citations=[
                    {
                        "source_type": source_type,
                        "document_id": "synthetic-policy",
                        "version_id": "synthetic-version",
                        "excerpt": "Returns within 7 days.",
                    }
                ],
            )
        return httpx.Response(200, json=reply)

    async with _client(handler) as client:
        cases = await probe.run_probes(client)
    assert sent == ["bad-citation", "healthy"]
    assert cases[0].failures
    assert not cases[1].failures


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("0", 0.0), ("2", 2.0), ("99999", 60.0), ("NaN", 15.0), ("inf", 15.0), ("-5", 15.0)],
)
def test_retry_after_numeric_values_are_finite_and_bounded(
    retry_after: str, expected: float
) -> None:
    response = httpx.Response(429, headers={"Retry-After": retry_after})
    assert probe._retry_delay(response) == expected


@pytest.mark.parametrize(("offset", "expected"), [(20, 20.0), (7200, 60.0), (-20, 15.0)])
def test_retry_after_http_date_uses_utc_and_caps_wait(
    monkeypatch: pytest.MonkeyPatch, offset: int, expected: float
) -> None:
    from unittest.mock import Mock

    now = datetime(2026, 10, 28, 12, tzinfo=UTC)
    monkeypatch.setattr(probe, "datetime", Mock(now=Mock(return_value=now)))
    response = httpx.Response(
        429, headers={"Retry-After": format_datetime(now + timedelta(seconds=offset), usegmt=True)}
    )
    assert probe._retry_delay(response) == expected
