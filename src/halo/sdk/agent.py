"""Base agent class matching AGI SDK interface.

This module provides the HaloAgent class that integrates with AGI SDK v0.3.5
for REAL Bench tasks. Simplified for Online RL - just policy -> action -> reward.

Supported modes:
- baseline_worker: GPT-4o-mini worker
- qwen_worker_zero: Qwen zero-shot worker
- qwen_worker_bc: Qwen + BC LoRA
- qwen_worker_dpo: Qwen + DPO LoRA
- qwen_worker_grpo: Qwen + GRPO LoRA
"""

import logging
import os
import re
from typing import Tuple, Dict, Optional, Any

from agisdk.REAL.browsergym.experiments.agent import Agent, AgentInfo

from ..agent import Orchestrator, OrchestratorConfig
from ..logging import TrajectoryLogger

logger = logging.getLogger(__name__)

# Valid action patterns for validation
VALID_ACTION_PATTERNS = [
    r'^click\("[^"]+"\)$',
    r'^fill\("[^"]+",\s*"[^"]*"\)$',
    r'^select_option\("[^"]+",\s*"[^"]*"\)$',
    r'^scroll\(-?\d+,\s*-?\d+\)$',
    r'^go_back\(\)$',
    r'^go_forward\(\)$',
    r'^goto\("[^"]+"\)$',
    r'^send_msg_to_user\("[^"]*"\)$',
    r'^noop\(\)$',
    r'^hover\("[^"]+"\)$',
    r'^press\("[^"]+"\)$',
    r'^focus\("[^"]+"\)$',
]

# Safe fallback actions in order of preference
SAFE_FALLBACK_ACTIONS = [
    'noop()',
    'scroll(0, 100)',
    'go_back()',
]


def validate_action(action: str) -> bool:
    """Validate action string against known patterns."""
    if not action or not isinstance(action, str):
        return False
    action = action.strip()
    for pattern in VALID_ACTION_PATTERNS:
        if re.match(pattern, action):
            return True
    # Also accept actions that start with valid prefixes (more lenient)
    valid_prefixes = ['click(', 'fill(', 'select_option(', 'scroll(', 'go_back()', 
                      'go_forward()', 'goto(', 'send_msg_to_user(', 'noop()', 
                      'hover(', 'press(', 'focus(']
    for prefix in valid_prefixes:
        if action.startswith(prefix):
            return True
    return False


def repair_action(action: str) -> str:
    """Attempt to repair a malformed action string."""
    if not action:
        return SAFE_FALLBACK_ACTIONS[0]
    
    action = action.strip()
    
    # Try to fix common issues
    # Missing closing paren
    if action.count('(') > action.count(')'):
        action = action + ')'
    
    # Missing quotes around bid
    if action.startswith('click(') and not '"' in action:
        bid = action[6:-1] if action.endswith(')') else action[6:]
        action = f'click("{bid}")'
    
    if validate_action(action):
        return action
    
    return SAFE_FALLBACK_ACTIONS[0]


class HaloAgent(Agent):
    """HALO Agent implementation for AGI SDK.

    Simplified for Online RL - just policy -> action -> reward.
    Uses Qwen3-VL-8B Vision-Language Model for GUI control.

    Supported modes:
    - gpt4o_baseline: GPT-4o for comparison / MCTS critic
    - qwen3vl_base: Qwen3-VL-8B base (before training)
    - qwen3vl_grpo: Qwen3-VL + Online GRPO LoRA
    - qwen3vl_mcts: Qwen3-VL + MCTS-trained LoRA (Agent Q style)
    """

    # Valid modes
    VALID_MODES = [
        'gpt4o_baseline',    # GPT-4o for comparison / MCTS critic
        'qwen3vl_base',      # Qwen3-VL-8B base (before training)
        'qwen3vl_grpo',      # Qwen3-VL + Online GRPO LoRA
        'qwen3vl_mcts',      # Qwen3-VL + MCTS-trained LoRA (Agent Q style)
    ]

    def __init__(
        self,
        mode: str = "qwen3vl_base",
        worker_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        worker_temperature: float = 0.0,
        max_steps: int = 70,
        enable_recovery_policies: Optional[bool] = None,
        qwen_backend: Optional[str] = None,
        qwen_base_url: Optional[str] = None,
        traj_logger: Optional[TrajectoryLogger] = None,
        # Legacy parameters (ignored)
        use_cache: Optional[bool] = None,
        use_macros: Optional[bool] = None,
        use_manager: Optional[bool] = None,
        manager_model: str = "gpt-4o",
        manager_warm_start: Optional[bool] = None,
        always_call_manager: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # Warn about deprecated parameters
        if use_cache is not None or use_macros is not None or use_manager is not None:
            logger.warning("use_cache, use_macros, use_manager are deprecated and ignored")

        # Set model based on mode
        if mode == "gpt4o_baseline":
            worker_model = "gpt-4o"
        elif mode.startswith("qwen3vl"):
            # All Qwen3-VL modes use the VLM
            if worker_model == "Qwen/Qwen3-VL-8B-Instruct" or worker_model == "gpt-4o-mini":
                worker_model = os.environ.get("HALO_WORKER_MODEL") or "Qwen/Qwen3-VL-8B-Instruct"
            if qwen_backend is None:
                qwen_backend = os.environ.get("HALO_WORKER_BACKEND") or "vllm"
            if qwen_base_url is None:
                qwen_base_url = os.environ.get("HALO_VLLM_URL") or "http://localhost:8000/v1"

        resolved_enable_recovery_policies = enable_recovery_policies if enable_recovery_policies is not None else True

        config = OrchestratorConfig(
            worker_model=worker_model,
            max_steps=max_steps,
            worker_temperature=float(worker_temperature),
            enable_recovery_policies=resolved_enable_recovery_policies,
            qwen_backend=qwen_backend or "vllm",
            qwen_base_url=qwen_base_url or "http://localhost:8000/v1",
        )

        self.orchestrator = Orchestrator(config=config, mode=mode)
        self.orchestrator.traj_logger = traj_logger
        self.mode = mode
        self.max_steps = max_steps
        self.invalid_action_count = 0
        self.total_action_count = 0

        if traj_logger is not None:
            existing_metadata = dict(getattr(traj_logger, "run_metadata", {}) or {})
            existing_metadata.update({
                "mode": mode,
                "worker_model": worker_model,
                "worker_temperature": float(worker_temperature),
                "max_steps": int(max_steps),
                "enable_recovery_policies": bool(resolved_enable_recovery_policies),
                "qwen_backend": config.qwen_backend,
                "qwen_base_url": config.qwen_base_url,
            })
            traj_logger.run_metadata = existing_metadata

    def reset(self, goal: str = ""):
        """Reset agent for new episode."""
        self.orchestrator.reset(goal=goal)

    def get_action(self, obs: Dict) -> Tuple[str, AgentInfo]:
        """Get action from orchestrator.

        This method is required by the browsergym interface.
        CRITICAL: Always returns a valid action string, never None.

        Args:
            obs: Preprocessed observation from obs_preprocessor()

        Returns:
            Tuple of (action_string, agent_info)
        """
        self.total_action_count += 1
        
        try:
            # Check step limit
            if self.orchestrator.step_count >= self.max_steps:
                action = 'send_msg_to_user("Maximum steps reached")'
                info = AgentInfo()
                info.think = "Step limit reached"
                info.stats = {
                    "step": self.orchestrator.step_count, 
                    "mode": self.mode,
                    "invalid_action_rate": self.invalid_action_count / max(1, self.total_action_count)
                }
                return action, info

            # Get action from orchestrator
            action, orch_info = self.orchestrator.get_action(obs)

            # Validate and repair action if needed
            if not validate_action(action):
                logger.warning(f"Invalid action from orchestrator: {action}")
                self.invalid_action_count += 1
                action = repair_action(action)
                if not validate_action(action):
                    action = SAFE_FALLBACK_ACTIONS[0]
                    logger.warning(f"Using fallback action: {action}")

            # Build AgentInfo
            info = AgentInfo()
            info.think = f"Step {orch_info['step']}: {orch_info['action_source']} -> {action[:50]}..."
            info.stats = {
                "step": orch_info['step'],
                "action_source": orch_info['action_source'],
                "cache_hit": orch_info.get('cache_hit', False),
                "manager_called": orch_info.get('manager_called', False),
                "mode": self.mode,
                "invalid_action_rate": self.invalid_action_count / max(1, self.total_action_count)
            }

            return action, info

        except Exception as e:
            logger.error(f"Error in get_action: {e}", exc_info=True)
            self.invalid_action_count += 1
            # Safe fallback - cycle through fallback actions
            fallback_idx = self.invalid_action_count % len(SAFE_FALLBACK_ACTIONS)
            action = SAFE_FALLBACK_ACTIONS[fallback_idx]
            info = AgentInfo()
            info.think = f"Error recovery: {e}"
            info.stats = {
                "error": str(e), 
                "mode": self.mode,
                "invalid_action_rate": self.invalid_action_count / max(1, self.total_action_count)
            }
            return action, info

    def obs_preprocessor(self, obs: Dict) -> Any:
        """Preprocess observations before feeding to get_action().

        Args:
            obs: Raw observation from environment

        Returns:
            Preprocessed observation (passed through with axtree)
        """
        # Keep the observation as-is since orchestrator handles processing
        return obs

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = self.orchestrator.get_stats()
        stats['invalid_action_count'] = self.invalid_action_count
        stats['total_action_count'] = self.total_action_count
        stats['invalid_action_rate'] = self.invalid_action_count / max(1, self.total_action_count)
        return stats


def create_halo_agent(
    mode: str = "qwen3vl_base",
    traj_logger: Optional[TrajectoryLogger] = None,
    max_steps: int = 70,
    **kwargs
) -> HaloAgent:
    """Factory function to create HaloAgent.

    Args:
        mode: Agent mode:
            - gpt4o_baseline: GPT-4o for comparison / MCTS critic
            - qwen3vl_base: Qwen3-VL-8B base (before training)
            - qwen3vl_grpo: Qwen3-VL + Online GRPO LoRA
            - qwen3vl_mcts: Qwen3-VL + MCTS-trained LoRA (Agent Q style)
        traj_logger: Optional trajectory logger
        max_steps: Maximum steps (default 70 for score-mode, use 25 for speed-mode)
        **kwargs: Additional arguments for HaloAgent

    Returns:
        Configured HaloAgent instance
    """
    return HaloAgent(mode=mode, traj_logger=traj_logger, max_steps=max_steps, **kwargs)
