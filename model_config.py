"""
Central model configuration for all Prompt2ML agents.

Tiering rationale:
  CODING_MODEL    — agents that WRITE and DEBUG code (preprocessing executor,
                    ML trainer). Code quality is capability-bound, so these get
                    the strongest generally-available coding model.
  REASONING_MODEL — agents whose output quality depends on multi-step reasoning
                    (strategist, ML planner, validators, requirement gatherer).
  LIGHT_MODEL     — high-volume / low-stakes agents (research sweeps, report
                    writers, dataset extractor) where the cost-efficient tier
                    is good enough.

Override any tier via environment variables without touching code:
  PROMPT2ML_CODING_MODEL / PROMPT2ML_REASONING_MODEL / PROMPT2ML_LIGHT_MODEL
"""

import os

CODING_MODEL = os.getenv("PROMPT2ML_CODING_MODEL", "gemini-3.5-flash")
REASONING_MODEL = os.getenv("PROMPT2ML_REASONING_MODEL", "gemini-3.5-flash")
LIGHT_MODEL = os.getenv("PROMPT2ML_LIGHT_MODEL", "gemini-3.1-flash-lite")
