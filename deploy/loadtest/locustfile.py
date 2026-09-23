"""E-commerce customer-service load profile for the RAG chatbot API.

Scale bar (from the product brief): 亿级用户, 日请求 80-100 万
(≈ 12 QPS average, 50-120 QPS promo peak). Capacity math with the
think times below (avg ~17s per turn → 1/17 req/s per virtual user):

    常态基线   ~200 users  ≈ 12 QPS
    大促峰值   ~1000 users ≈ 60 QPS    (chains: embeddings + LLM)
    大促极限   ~2000 users ≈ 120 QPS   (find the real ceiling)

Runbook:

    # local smoke (deterministic intents only, no LLM key needed)
    LOADTEST_SMOKE=1 locust --headless -u 5 -r 1 -t 30s --host http://localhost:8010

    # baseline soak against a deployed env
    locust --headless -u 200 -r 10 -t 30m --host https://cs.example.com

    # promo peak
    locust --headless -u 2000 -r 20 -t 30m --csv results/promo --host ...

Traffic profile mirrors the golden-set intent distribution: FAQ/policy
questions and order/logistics queries dominate, refunds run the full
confirmation-gate flow, a slice asks for a human. Each virtual user is
one customer holding one session across turns (the LangGraph
checkpointer keys state by session_id, so session reuse exercises
memory growth too).

LOADTEST_SMOKE=1 restricts the mix to deterministic paths (greeting /
order / shipping / refund-gate / complaint) so the suite runs green
without an LLM key — LLM-dependent intents (chitchat, open FAQ) are
dropped rather than degraded.
"""

import itertools
import os
import random

from locust import HttpUser, between, task

# Smoke mode is read at import time: locust registers tasks in the
# metaclass at class-definition time, so filtering must happen via the
# decorator, not by mutating the class afterwards.
_SMOKE = os.environ.get("LOADTEST_SMOKE") == "1"


def _task_if(weight: int, enabled: bool):
    """@task(weight) when enabled; plain (unregistered) method when not.

    Locust only picks up methods wrapped by @task — an undecorated
    method is invisible to the task list, which is exactly the smoke
    behavior we want for LLM-dependent intents.
    """
    if enabled:
        return task(weight)
    return lambda fn: fn


# Session ids: high random range so a load run never collides with real
# persisted session rows (the persister tolerates FK misses, but keeps
# test traffic out of real conversation history).
_SESSION_IDS = itertools.count(random.randint(10_000_000, 99_000_000))

FAQ_QUESTIONS = [
    "退货流程是什么？",
    "退款多久能到账？",
    "你们的退货政策是怎样的",
    "订单包邮的门槛是多少",
    "可以开发票吗",
    "优惠券怎么用",
    "怎么修改收货地址",
    "支持七天无理由退货吗",
]

ORDER_QUERIES = [
    "我的订单到哪了",
    "帮我查一下订单状态",
    "ORD1001 发货了吗",
    "我昨天下的单什么时候发货",
    "看一下我的订单",
]

SHIPPING_QUERIES = [
    "物流到哪了",
    "快递什么时候送达",
    "帮我查物流 ORD1001",
    "我的包裹怎么还没动",
]

REFUND_OPENERS = [
    "ORD1001 有质量问题，我要退款",
    "买错了，帮我退款",
    "ORD1002 不想要了，申请退款",
]

COMPLAINTS = [
    "我要投诉物流太慢",
    "客服态度很差，我要投诉",
    "商品质量和描述不符，投诉",
]

GREETINGS = ["你好", "在吗", "hi，有人吗", "你好呀"]

CHITCHAT = [
    "今天天气真好啊",
    "你们机器人会取代人类客服吗",
    "无聊，聊聊天",
    "你是AI吗",
]

HANDOFF = ["给我转人工", "我要找真人客服", "转人工，快点"]

CONFIRM_WORDS = ["确认", "是的，确认", "对，办理吧"]


class CustomerServiceUser(HttpUser):
    """One virtual customer: a session, a realistic turn mix."""

    wait_time = between(8, 30)  # CS think time; avg ~19s/turn

    def on_start(self) -> None:
        self.session_id = next(_SESSION_IDS)
        self.user_id = random.randint(1, 1000)
        self.turns = 0
        self.awaiting_confirm = False
        # Fail fast with a clear error if the target is not our API.
        with self.client.get("/health", name="health:ready", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"target does not look like the CS API (health={r.status_code})")

    def _chat(self, message: str, name: str) -> None:
        self.turns += 1
        self.client.post(
            "/api/v1/chat",
            json={
                "message": message,
                "session_id": self.session_id,
                "user_id": self.user_id,
            },
            name=name,
        )

    @_task_if(25, enabled=not _SMOKE)  # needs embeddings (BGE-M3) offline
    def faq_policy(self) -> None:
        self._chat(random.choice(FAQ_QUESTIONS), "chat:faq_policy")

    @task(20)
    def order_query(self) -> None:
        self._chat(random.choice(ORDER_QUERIES), "chat:order_query")

    @task(15)
    def shipping(self) -> None:
        self._chat(random.choice(SHIPPING_QUERIES), "chat:shipping")

    @task(10)
    def refund_flow(self) -> None:
        # Two-turn irreversible flow: stage → confirm (exercises the
        # confirmation gate, ownership authz and tool execution).
        if self.awaiting_confirm:
            self._chat(random.choice(CONFIRM_WORDS), "chat:refund_confirm")
            self.awaiting_confirm = False
        else:
            self._chat(random.choice(REFUND_OPENERS), "chat:refund_open")
            self.awaiting_confirm = True

    @task(5)
    def complaint(self) -> None:
        self._chat(random.choice(COMPLAINTS), "chat:complaint")

    @task(10)
    def greeting(self) -> None:
        self._chat(random.choice(GREETINGS), "chat:greeting")

    @_task_if(10, enabled=not _SMOKE)  # needs an LLM key
    def chitchat(self) -> None:
        self._chat(random.choice(CHITCHAT), "chat:chitchat")

    @task(5)
    def handoff(self) -> None:
        self._chat(random.choice(HANDOFF), "chat:handoff")


class StreamingUser(HttpUser):
    """SSE slice (~20% of traffic): exercises the queue-bridge stream."""

    weight = 25 if not _SMOKE else 0  # dropped in smoke (LLM/embeddings)
    wait_time = between(10, 40)

    def on_start(self) -> None:
        self.session_id = next(_SESSION_IDS)
        self.user_id = random.randint(1, 1000)

    @task
    def stream_turn(self) -> None:
        message = random.choice(FAQ_QUESTIONS + CHITCHAT)
        with self.client.post(
            "/api/v1/chat/stream",
            json={
                "message": message,
                "session_id": self.session_id,
                "user_id": self.user_id,
            },
            stream=True,
            name="chat:stream",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"stream http {response.status_code}")
                return
            chunks = 0
            for line in response.iter_lines():
                if line.startswith(b"data:"):
                    chunks += 1
            if chunks == 0:
                response.failure("stream closed without any data frame")
