# Documentation Index

Design contracts, operational runbooks, and dated evaluation evidence for
Agent Store. Design docs define the invariants the code must uphold; the
reviews directory records what was actually measured and when. Most design
notes were written during development in Chinese; the README and docstrings
carry the English summary.

## Design contracts (`design/`)

| Document | What it fixes |
|---|---|
| [product-north-star.md](design/product-north-star.md) | Product north star and the rules that decide what gets built next |
| [public-commerce.md](design/public-commerce.md) | Public visitors and the agent-driven text-only mock purchase flow |
| [conversational-commerce-assessment.md](design/conversational-commerce-assessment.md) | Assessment of chat-driven purchasing integration options |
| [commerce-mcp-server.md](design/commerce-mcp-server.md) | MCP (Model Context Protocol) surface for the commerce provider |
| [durable-actions.md](design/durable-actions.md) | Contract for confirmed business actions and crash recovery |
| [session-incarnation.md](design/session-incarnation.md) | Session incarnation ids: isolating stale requests after reset |
| [human-conversation.md](design/human-conversation.md) | Private counseling sessions and human handoff |
| [message-feedback.md](design/message-feedback.md) | Message identity, processing status, and user feedback |
| [quality-mutations.md](design/quality-mutations.md) | Binding feedback and QA writes to sessions and evidence |
| [knowledge-access.md](design/knowledge-access.md) | Access control for the shared knowledge base |
| [knowledge-publication.md](design/knowledge-publication.md) | Read eligibility for published knowledge revisions |
| [answer-citations.md](design/answer-citations.md) | Answer-time citation snapshots of the exact published revision |
| [per-user-architecture.md](design/per-user-architecture.md) | Per-user knowledge isolation architecture |
| [visitor-preference-memory.md](design/visitor-preference-memory.md) | Cross-session visitor preference memory |
| [cache-layering-plan.md](cache-layering-plan.md) | Historical multi-layer cache design discussion (2026-09-24) |

## Operations (`operations/`)

| Document | What it covers |
|---|---|
| [backup-and-recovery.md](operations/backup-and-recovery.md) | PostgreSQL backup and verified-restore runbook (sanitized; no host specifics) |
| [live-probe.md](operations/live-probe.md) | Online contract probe: 23 fixed zh/en scenarios against a running stack |
| [answer-quality-evaluation.md](operations/answer-quality-evaluation.md) | Answer-quality evaluation methodology |
| [session-quality-metrics.md](operations/session-quality-metrics.md) | Session quality metrics: failure, completion, and resolution evidence |
| [containment-alerts.md](operations/containment-alerts.md) | Containment alerting and statistics availability |
| [shared-answer-cache.md](operations/shared-answer-cache.md) | Context boundaries of the shared answer cache |
| [handoff-context.md](operations/handoff-context.md) | Staff workspace and ticket session context |

## Reviews and evaluation evidence (`reviews/`)

- [agentic-rag-review-2026-09-26.md](reviews/agentic-rag-review-2026-09-26.md) — the
  full architecture review that drove the 2026-10 quality campaign: 12 finding
  categories, each with its fix ledger (commit, acceptance test, what remains).
- [product-quality-2026-10-16.md](reviews/product-quality-2026-10-16.md) — product
  quality iteration notes.
- [public-purchase-milestone-2026-10-28.md](reviews/public-purchase-milestone-2026-10-28.md) —
  public-visitor mock-purchase reliability milestone.
- `*-results-*.json` — machine-readable evidence for the dated evaluation runs
  (answer quality, containment alerts, context cache, durable actions, handoff
  context, human conversation, knowledge access/publication, live probe,
  message feedback, public commerce, refunds, session incarnation, support
  continuity). Each file records scope, commands, and observed outcomes.
