"""Live-probe harness logic against a scripted transport, never a real server.

The probe replays the invariant families every live bug came from
(mid-collection handoff, phrasing battery, polite-split refund, EN
language purity). Here a MockTransport plays the server: a well-behaved
one must yield all-PASS and exit 0; regressions (chitchat swallow, CJK
leak, HTTP errors) must surface as FAIL and exit 1.
"""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[4]


def _load_probe() -> Any:
    path = ROOT / "scripts" / "probe_live.py"
    spec = importlib.util.spec_from_file_location("probe_live", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

Handler = Callable[[httpx.Request], httpx.Response]

ZH_TICKET = "您的人工客服请求已记录（工单号 #12）。当前排队位置：第 2 位。"
EN_TICKET = (
    "Your request for a human agent is recorded (ticket #12), "
    "your position in the queue: 2, the ticket is linked to this conversation."
)
ZH_SLOT = "好的，请提供您的订单号，我来帮您处理。"
EN_SLOT = "Sure — could you share your order number so I can look into it?"
ZH_KB = "退货政策：签收后七天内可申请退货，商品须保持完好并保留购买凭证。"
EN_KB = "Our return policy allows returns within 7 days of delivery in resaleable condition."


_TICKET_MESSAGES = frozenset(
    probe.HANDOFF_PHRASES_ZH
    + probe.HANDOFF_PHRASES_EN
    + probe.HANDOFF_PERTURBED
    + ["那我要人工", "then get me a human agent"]
)
_KB_MESSAGES = frozenset(
    ["退货政策是什么", "what is the return policy", "and how long do refunds take"]
)


def _classify(message: str) -> str:
    """The behavior a correct server would exhibit, by request text."""
    if message in _TICKET_MESSAGES:
        return "ticket"
    if message in _KB_MESSAGES:
        return "kb"
    return "slot"  # complaint / refund task turns


def _well_behaved_handler(seen: list[tuple[int, str]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        session_id = int(body["session_id"])
        message = str(body["message"])
        seen.append((session_id, message))
        kind = _classify(message)
        content = {
            "ticket": ZH_TICKET if probe.has_cjk(message) else EN_TICKET,
            "slot": ZH_SLOT if probe.has_cjk(message) else EN_SLOT,
            "kb": ZH_KB if probe.has_cjk(message) else EN_KB,
        }[kind]
        return httpx.Response(
            200, json={"content": content, "session_id": session_id, "intent": kind}
        )

    return handler


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://probe.test", transport=httpx.MockTransport(handler))


class TestProbeGreenPath:
    async def test_all_invariants_pass_on_well_behaved_server(self) -> None:
        seen: list[tuple[int, str]] = []
        async with _client(_well_behaved_handler(seen)) as client:
            cases = await probe.run_probes(client)

        assert not any(c.failures for c in cases), [c.failures for c in cases]
        report = probe.format_report(cases)
        assert f"{len(cases)}/{len(cases)} cases passed" in report

    async def test_main_returns_zero_on_green_run(self) -> None:
        seen: list[tuple[int, str]] = []
        async with _client(_well_behaved_handler(seen)) as client:
            exit_code = await probe.main([], client=client)

        assert exit_code == 0

    async def test_each_case_gets_its_own_session(self) -> None:
        """Case isolation: no session id may serve two cases."""
        seen: list[tuple[int, str]] = []
        async with _client(_well_behaved_handler(seen)) as client:
            await probe.run_probes(client)

        first_turns: dict[int, str] = {}
        for session_id, message in seen:
            first_turns.setdefault(session_id, message)
        # Multi-turn cases send their second turn on the same session;
        # every *first* message must sit on a distinct session.
        sessions = list(first_turns)
        assert len(sessions) == len(probe.build_cases())
        assert all(sid > 0 for sid in sessions)


class TestProbeFailureDetection:
    async def test_chitchat_swallow_fails_every_case_and_exits_nonzero(self) -> None:
        """The canonical live failure: handoff ask answered with chitchat."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "content": "您好呀，今天想聊点什么呢？",
                    "session_id": body["session_id"],
                    "intent": "chitchat",
                },
            )

        async with _client(handler) as client:
            cases = await probe.run_probes(client)

        assert all(c.failures for c in cases)
        report = probe.format_report(cases)
        assert "FAIL" in report
        assert "0/" in report

    async def test_cjk_leak_in_en_reply_is_a_failure(self) -> None:
        """EN handoff answered with the zh ticket copy must be caught."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={"content": ZH_TICKET, "session_id": body["session_id"], "intent": "handoff"},
            )

        async with _client(handler) as client:
            cases = await probe.run_probes(client)

        en_cases = [c for c in cases if "-en:" in c.name or c.name.endswith("-en")]
        assert en_cases, "expected EN cases in the battery"
        for case in en_cases:
            assert case.failures, case.name
        # The single-turn EN battery cases reach their ticket check
        # directly, so the leak itself must be the recorded failure.
        for case in en_cases:
            if len(case.turns) == 1:
                assert any("CJK" in f for f in case.failures), case.name

    async def test_http_error_stops_the_case_with_a_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        async with _client(handler) as client:
            cases = await probe.run_probes(client)

        assert all(c.failures for c in cases)
        assert all("HTTP 500" in f for c in cases for f in c.failures)

    async def test_ticket_on_slot_turn_is_a_failure(self) -> None:
        """我要退款 must collect slots first — an instant ticket is wrong."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={"content": ZH_TICKET, "session_id": body["session_id"], "intent": "handoff"},
            )

        async with _client(handler) as client:
            cases = await probe.run_probes(client)

        refund = next(c for c in cases if "refund" in c.name)
        assert any("slot-collection" in f for f in refund.failures)


class TestPredicates:
    def test_has_cjk_matches_server_detection_window(self) -> None:
        assert probe.has_cjk("已为您转接")
        assert not probe.has_cjk("plain ascii 123 #!")

    def test_has_ticket_matches_both_language_acks(self) -> None:
        assert probe.has_ticket(ZH_TICKET)
        assert probe.has_ticket(EN_TICKET)

    def test_order_ask_matches_both_language_prompts(self) -> None:
        assert probe.asks_for_order(ZH_SLOT)
        assert probe.asks_for_order(EN_SLOT)
        assert not probe.asks_for_order(ZH_KB)


class TestMainExitCode:
    async def test_main_exit_code_follows_failures(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() must be import-callable with argv, CI/cron friendly."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={"content": ZH_TICKET, "session_id": body["session_id"], "intent": "handoff"},
            )

        async with _client(handler) as client:
            exit_code = await probe.main([], client=client)

        assert exit_code == 1
        report = capsys.readouterr().out
        assert "FAIL" in report
