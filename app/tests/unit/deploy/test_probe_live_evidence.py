"""Probe evidence must survive independent history and staff observations."""

import copy
import json
from typing import Any

import httpx
import pytest

from app.models.schemas.citation import SourceCitation
from scripts import probe_live as probe

_OWNER = 1
_SESSION = 101
_TICKET = 12
_QUESTION = "what is the return policy"
_ANSWER = "Returns are accepted within 7 days."


def _knowledge_reply(source_type: str = "document") -> dict[str, Any]:
    citation: dict[str, Any] = {
        "source_type": source_type,
        "version_id": "returns-v1",
        "excerpt": _ANSWER,
    }
    if source_type == "faq":
        citation["faq_id"] = "returns-policy"
        sources = ["faq:returns-policy"]
    else:
        citation.update(document_id="returns-doc", page_start=1)
        sources = ["returns-doc"]
    return {
        "session_id": _SESSION,
        "content": _ANSWER,
        "intent": "policy",
        "metadata": {"pending_slots": []},
        "sources": sources,
        "citations": [citation],
    }


def _ticket_reply() -> dict[str, Any]:
    return {
        "session_id": _SESSION,
        "content": f"Your request is recorded (ticket #{_TICKET}).",
        "intent": "handoff",
        "metadata": {"handoff_outcome": "queued", "handoff_ticket_id": _TICKET},
        "sources": None,
        "citations": [],
    }


def _owned_chat_handler(
    case: Any, reply: dict[str, Any], *, missing_history_field: str | None = None
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/me":
            return httpx.Response(200, json={"id": _OWNER, "is_active": True, "is_admin": False})
        if request.url.path == "/api/v1/sessions":
            return httpx.Response(200, json={"id": _SESSION, "user_id": _OWNER})
        if request.url.path == "/api/v1/chat/history":
            assistant = {
                "role": "assistant",
                "content": reply["content"],
                "sources": copy.deepcopy(reply["sources"]),
                "citations": copy.deepcopy(reply["citations"]),
            }
            if missing_history_field is not None:
                assistant[missing_history_field] = []
            return httpx.Response(
                200,
                json={
                    "session_id": _SESSION,
                    "messages": [
                        {
                            "role": "user",
                            "content": case.turns[0].send,
                            "sources": None,
                            "citations": [],
                        },
                        assistant,
                    ],
                },
            )
        assert request.method == "POST" and request.url.path == "/api/v1/chat"
        assert json.loads(request.content)["session_id"] == _SESSION
        return httpx.Response(200, json=copy.deepcopy(reply))

    return handler


def _client(handler: Any, *, authenticated: bool = False) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="http://offline-evidence.invalid",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer local-probe-fixture"} if authenticated else {},
    )


@pytest.mark.parametrize("missing_field", ["sources", "citations"])
async def test_owned_history_cannot_drop_the_answer_evidence(monkeypatch, missing_field):
    case = probe.ProbeCase(
        "history-evidence",
        [probe.Turn(_QUESTION, probe.is_knowledge_answer(probe._EN_POLICY_EVIDENCE))],
    )
    monkeypatch.setattr(probe, "build_cases", lambda: [case])
    async with _client(
        _owned_chat_handler(case, _knowledge_reply(), missing_history_field=missing_field),
        authenticated=True,
    ) as client:
        code = await probe.main([], client=client)

    assert code == 1
    assert case.failures
    assert not case.history_verified


def _ticket_row(*, ticket_id: int = _TICKET, session_id: int = _SESSION, user_id: Any = _OWNER):
    return {
        "id": ticket_id,
        "session_id": session_id,
        "user_id": user_id,
        "status": "open",
        "assigned_to": None,
        "claimed_at": None,
    }


@pytest.mark.parametrize("observed_owner", [999, True], ids=["foreign-owner", "boolean-owner"])
async def test_staff_observation_cannot_misattribute_an_owned_ticket(monkeypatch, observed_owner):
    case = probe.ProbeCase("ticket-owner", [probe.Turn("human agent", probe.is_ticket(True))])
    monkeypatch.setattr(probe, "build_cases", lambda: [case])
    reads = []

    def staff(request):
        reads.append((request.method, request.url.path))
        assert request.method == "GET"
        if request.url.path == "/api/v1/auth/me":
            return httpx.Response(200, json={"id": 2, "is_admin": True, "is_active": True})
        rows = (
            [_ticket_row(user_id=observed_owner)] if request.url.params["status"] == "open" else []
        )
        return httpx.Response(200, json={"tickets": rows, "count": len(rows)})

    async with (
        _client(_owned_chat_handler(case, _ticket_reply()), authenticated=True) as client,
        _client(staff) as observer,
    ):
        code = await probe.main(
            ["--require-ticket-evidence"], client=client, ticket_client=observer
        )

    assert code == 1
    assert case.failures
    assert not case.verified_ticket_ids
    assert any(path == "/api/v1/handoff/tickets" for _, path in reads)


@pytest.mark.parametrize("fault", ["missing", "wrong-session", "pagination-truncated"])
async def test_unobserved_ticket_never_passes_the_bounded_staff_scan(monkeypatch, fault):
    case = probe.ProbeCase("ticket-scan", [probe.Turn("human agent", probe.is_ticket(True))])
    monkeypatch.setattr(probe, "build_cases", lambda: [case])
    monkeypatch.setattr(probe, "TICKET_PAGE_SIZE", 2)
    monkeypatch.setattr(probe, "TICKET_MAX_PAGES", 2)
    reads = []
    # In the truncation case the target really is in the scripted queue,
    # but only on a third page that the probe must not request.
    queue = [_ticket_row(ticket_id=100 + i, session_id=201 + i) for i in range(4)]
    queue.append(_ticket_row())

    def staff(request):
        assert request.method == "GET"
        if request.url.path == "/api/v1/auth/me":
            return httpx.Response(200, json={"id": 2, "is_admin": True, "is_active": True})
        status = request.url.params["status"]
        skip = int(request.url.params["skip"])
        reads.append((status, skip))
        if fault == "missing" or status != "open":
            rows = []
        elif fault == "wrong-session":
            rows = [_ticket_row(session_id=_SESSION + 1)]
        else:
            rows = queue[skip : skip + 2]
        return httpx.Response(200, json={"tickets": rows, "count": len(rows)})

    async with (
        _client(_owned_chat_handler(case, _ticket_reply()), authenticated=True) as client,
        _client(staff) as observer,
    ):
        code = await probe.main(
            ["--require-ticket-evidence"], client=client, ticket_client=observer
        )

    assert code == 1
    assert case.failures
    assert not case.verified_ticket_ids
    if fault == "pagination-truncated":
        assert ("open", 0) in reads and ("open", 2) in reads
        assert all(skip < 4 for _, skip in reads)


def test_schema_invalid_citation_cannot_pass_on_its_policy_words():
    reply = _knowledge_reply()
    reply["citations"][0]["page_start"] = -1

    assert probe.is_knowledge_answer(probe._EN_POLICY_EVIDENCE)(reply) is not None


@pytest.mark.parametrize("source_type", ["document", "faq"])
def test_well_formed_versioned_citation_remains_usable(source_type):
    reply = _knowledge_reply(source_type)
    SourceCitation.model_validate(reply["citations"][0])

    assert probe.is_knowledge_answer(probe._EN_POLICY_EVIDENCE)(reply) is None


def test_probe_keeps_its_stricter_seeded_policy_revision_requirement():
    reply = _knowledge_reply()
    del reply["citations"][0]["version_id"]
    # The public DTO permits an unknown revision, while these seeded-policy
    # probes deliberately require one. This is a stricter probe contract.
    SourceCitation.model_validate(reply["citations"][0])

    assert probe.is_knowledge_answer(probe._EN_POLICY_EVIDENCE)(reply) is not None
