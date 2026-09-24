"""Agent-loop telemetry (the funnel's deepest serving layer).

The funnel counter (``chat_funnel_layers_total``) says how many turns
reached the agent layer; these metrics say how the layer performs. The
north star weights deep-funnel tool-loop users as the real
customer-service need, so layer health must be computable, not
anecdotal: outcome mix (answered / staged / exhausted / fallback to
the slot pipeline), per-tool success and error rates, and tool-call
depth per completed run.
"""

from prometheus_client import REGISTRY, Counter, Histogram

AGENT_LOOP_OUTCOMES = Counter(
    "agent_loop_outcomes_total",
    "Agent loop runs by terminal outcome",
    labelnames=["outcome"],
    registry=REGISTRY,
)

AGENT_TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Tool calls executed inside the agent loop by result",
    labelnames=["tool", "outcome"],
    registry=REGISTRY,
)

AGENT_LOOP_TOOL_CALLS = Histogram(
    "agent_loop_tool_calls",
    "Tool calls per completed agent-loop run",
    buckets=(0, 1, 2, 3, 4, 6, 10),
    registry=REGISTRY,
)

# Terminal outcome labels — renaming breaks dashboards and queries.
OUTCOME_ANSWERED = "answered"
OUTCOME_STAGED = "staged"
OUTCOME_BUDGET_EXHAUSTED = "budget_exhausted"
OUTCOME_STEPS_EXHAUSTED = "steps_exhausted"
OUTCOME_FALLBACK = "fallback"


def record_agent_outcome(outcome: str, tool_calls: int) -> None:
    """Record one completed agent run: its outcome and tool-call depth."""
    AGENT_LOOP_OUTCOMES.labels(outcome=outcome).inc()
    AGENT_LOOP_TOOL_CALLS.observe(tool_calls)
