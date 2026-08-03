"""Job radar v2: source-first, incremental pipeline with deterministic gating and
budget-controlled LLM matching.

Subpackages:
- src.radar.core: Models, gating, scoring, governor, salary, signals, queue
- src.radar.sources: Board registry, ATS interceptor, GitHub poller, dorking,
  crawler, discovery, instant poller, agents
- src.radar.engine: Orchestrator, outreach
"""

from __future__ import annotations
