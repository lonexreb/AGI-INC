"""AGI SDK v0.3.5 integration for HALO Agent.

This module provides the integration layer between HALO Agent and AGI SDK,
including agent implementations, harness wrappers, and task execution utilities.

CRITICAL: Always use task_version="v2" when running tasks.
The SDK defaults to v1 if version is omitted - this MUST be avoided.

Supported modes:
- baseline_worker: Worker only, no manager, no caches
- hierarchy_mgr_gate: Worker + Manager (gated), no caches
- hierarchy_vac: Worker + Manager + VAC
- hierarchy_vac_macros: Worker + Manager + VAC + Macro cache (full HALO)
"""

from .agent import HaloAgent, create_halo_agent, validate_action, repair_action
from .harness import (
    HaloAgentArgs,
    create_harness,
    run_single_task,
    run_task_subset,
    ensure_v2_task,
    SUPPORTED_MODES,
)

__all__ = [
    "HaloAgent",
    "create_halo_agent",
    "validate_action",
    "repair_action",
    "HaloAgentArgs",
    "create_harness",
    "run_single_task",
    "run_task_subset",
    "ensure_v2_task",
    "SUPPORTED_MODES",
]
