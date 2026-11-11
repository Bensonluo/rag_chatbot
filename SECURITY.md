# Security Policy

## Scope

This repository runs a public demo ([benluo.art/projects/rag-chatbot](https://benluo.art/projects/rag-chatbot/)) of an agentic-commerce and RAG customer-service engine. The commerce catalog is a deterministic synthetic dataset and payments are simulated — no real money moves.

Security issues in the following areas are still in scope and taken seriously:

- Authentication, session, and guest-cookie handling
- Authorization boundaries (session ownership, admin endpoints, knowledge-base management)
- Prompt injection and guardrail bypasses that could trigger unintended business actions
- The commerce state machine (idempotency, exactly-once semantics, confirmation gates)
- Injection vectors in retrieval, ingestion, or template rendering
- Anything in a dependency that is exploitable in the demo deployment

## Reporting a Vulnerability

**Please do not open a public issue for security reports.**

Use GitHub's private vulnerability reporting: **Security tab → Report a vulnerability** on this repository. If private reporting is unavailable, open a regular issue asking the maintainer to enable it, without vulnerability details.

Include when possible:

1. Affected component or endpoint (e.g. `app/api/v1/...`, `/commerce/*`)
2. Steps to reproduce, with the exact request/payload
3. Impact assessment (what an attacker gains)
4. Suggested fix, if any

## Response Expectations

This is a portfolio/open-source project maintained in spare time. Aim for:

- Acknowledgment within a few days
- Assessment and (if accepted) a fix in a reasonable timeframe
- Credit in the release notes if you wish

## Demo Boundaries

The live deployment intentionally keeps secrets out of the repository, uses token-gated internal services, and runs simulated commerce only. Reports that require physical access, compromised credentials, or control of the hosting server are out of scope.
