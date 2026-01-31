"""Orchestrator for HALO Agent.

Simplified for Online RL - just policy -> action -> reward.
Uses Qwen3-VL-30B-A3B Vision-Language Model (MoE) for GUI control.
"""

import os
import re
import time
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from ..constants import DEFAULT_MODEL
from ..obs import (
    summarize_observation, extract_actionable_nodes, get_obs_hash,
    build_state_key, state_key_hash, classify_page_type
)
from ..obs.fingerprint import extract_site_id
from ..obs.probe import extract_signals, compute_state_hash
from ..policy import WorkerPolicy, QwenWorkerPolicy, create_qwen_worker_policy
from ..verify import ActionVerifier, LoopDetector
from ..logging import TrajectoryLogger
from ..rl.progress import score_progress

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator."""
    worker_model: str = DEFAULT_MODEL
    max_steps: int = 70
    worker_temperature: float = 0.0
    # Recovery policies
    no_progress_threshold: int = 3  # go_back after N no-progress steps
    enable_recovery_policies: bool = True
    # Qwen worker backend configuration
    qwen_backend: str = "vllm"
    qwen_base_url: str = "http://localhost:8000/v1"


class Orchestrator:
    """Main orchestrator for Online RL training."""

    def __init__(self, config: OrchestratorConfig = None, mode: str = "qwen3vl_base"):
        self.config = config or OrchestratorConfig()
        self.mode = mode

        # Initialize worker policy based on mode
        if mode == "gpt5_baseline":
            # GPT-5.2 for comparison / MCTS critic (best vision model)
            self.worker = WorkerPolicy(model="gpt-5.2", temperature=self.config.worker_temperature)
            logger.info("Using GPT-5.2 baseline policy")

        elif mode.startswith('qwen3vl'):
            # Qwen3-VL-30B-A3B Vision-Language Model modes (MoE)
            qwen_backend = self.config.qwen_backend or "vllm"
            qwen_base_url = self.config.qwen_base_url or "http://localhost:8000/v1"
            default_model_name = os.environ.get("HALO_WORKER_MODEL") or DEFAULT_MODEL
            requested = self.config.worker_model

            # Adapter paths for trained modes
            adapter_defaults = {
                "qwen3vl_grpo": "checkpoints/qwen3vl_grpo_lora",
                "qwen3vl_mcts": "checkpoints/qwen3vl_mcts_lora",
            }

            adapter_path = None
            model_name = requested or default_model_name

            if mode in adapter_defaults:
                # Check if custom adapter path was provided
                if requested and (
                    os.path.exists(requested)
                    or requested.startswith("checkpoints/")
                    or requested.startswith("models/")
                    or requested.endswith("_lora")
                ):
                    adapter_path = requested
                    model_name = default_model_name
                else:
                    adapter_path = adapter_defaults[mode]

            self.worker = QwenWorkerPolicy(
                model_name=model_name,
                backend=qwen_backend,
                base_url=qwen_base_url,
                adapter_path=adapter_path,
                temperature=self.config.worker_temperature,
            )

            logger.info(
                f"Using Qwen3-VL policy (mode: {mode}, backend: {qwen_backend}, model: {model_name}, adapter: {adapter_path})"
            )
        else:
            # Fallback to basic worker
            self.worker = WorkerPolicy(model=self.config.worker_model, temperature=self.config.worker_temperature)
            logger.warning(f"Unknown mode '{mode}', using default WorkerPolicy")

        self.verifier = ActionVerifier()
        self.loop_detector = LoopDetector()

        # State
        self.step_count = 0
        self.action_history = []
        self.prev_obs: Optional[Dict] = None
        self.goal: str = ""

        # Recovery state
        self.no_progress_count = 0
        self.last_error_action = ""
        self.consecutive_error_count = 0
        self.last_url = ""

        # Stats
        self.invalid_id_count = 0
        self.recovery_action_count = 0

        # Hybrid agent state
        self.ax_signals = None
        self.state_hash = ""
        self.prev_state_hash = None

        # Trajectory logger (set externally)
        self.traj_logger: Optional[TrajectoryLogger] = None

    def reset(self, goal: str = ""):
        """Reset for new episode."""
        self.step_count = 0
        self.action_history = []
        self.prev_obs = None
        self.goal = goal
        self.no_progress_count = 0
        self.last_error_action = ""
        self.consecutive_error_count = 0
        self.last_url = ""
        self.invalid_id_count = 0
        self.recovery_action_count = 0
        self.loop_detector.reset()

    def get_action(self, obs: Dict) -> Tuple[str, Dict[str, Any]]:
        """Get next action from policy.

        Args:
            obs: Current observation

        Returns:
            Tuple of (action_string, info_dict)
        """
        start_time = time.time()
        self.step_count += 1

        # Extract AX signals and compute robust state hash
        self.ax_signals = extract_signals(obs)
        self.state_hash = compute_state_hash(self.ax_signals)

        # Extract goal if not set
        if not self.goal and 'goal_object' in obs:
            goal_obj = obs['goal_object']
            if isinstance(goal_obj, list):
                self.goal = ' '.join(
                    item.get('text', '') for item in goal_obj
                    if isinstance(item, dict) and item.get('type') == 'text'
                )

        # Build state key and page type
        state_key = build_state_key(obs, ax_signals=self.ax_signals)
        page_type = classify_page_type(obs.get('url', ''), obs.get('title', ''))
        obs_hash = get_obs_hash(obs)

        progress_info = score_progress(
            obs,
            goal=self.goal,
            site_id=state_key.site_id,
            page_type=page_type,
        )

        # Update loop detector with robust state hash
        self.loop_detector.add_state(self.state_hash, self.action_history[-1] if self.action_history else "")
        loop_detected = self.loop_detector.is_loop()

        # Extract actionable elements and valid bids
        axtree = obs.get('axtree_object')
        elements = extract_actionable_nodes(axtree, top_k=30) if axtree else []
        valid_bids = {str(e['bid']) for e in elements}

        # Summarize observation
        obs_summary = summarize_observation(obs, self.goal)

        # Get last action error
        last_action_error = obs.get('last_action_error', '')

        # Track progress (URL changes indicate progress)
        current_url = obs.get('url', '')
        made_progress = current_url != self.last_url
        self.last_url = current_url

        if made_progress:
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1

        # Track consecutive errors
        if last_action_error:
            if self.action_history and self.action_history[-1] == self.last_error_action:
                self.consecutive_error_count += 1
            else:
                self.consecutive_error_count = 1
            self.last_error_action = self.action_history[-1] if self.action_history else ""
        else:
            self.consecutive_error_count = 0

        # Decision
        action = None
        action_source = "worker"
        recovery_action = False

        # Check recovery policies first
        if self.config.enable_recovery_policies:
            recovery_result = self._check_recovery_policies(
                loop_detected, last_action_error, obs_summary, valid_bids
            )
            if recovery_result:
                action, action_source = recovery_result
                recovery_action = True
                self.recovery_action_count += 1
                logger.info(f"Recovery action: {action} (source: {action_source})")

        # Get action from worker policy
        if action is None:
            worker_result = self.worker.get_action(
                obs_summary=obs_summary,
                goal=self.goal,
                action_history=self.action_history,
                manager_guidance=None,
                valid_bids=valid_bids,
                last_action_error=last_action_error
            )
            action = worker_result["action"]
            if worker_result.get("was_repaired"):
                self.invalid_id_count += 1

        # Record action
        self.action_history.append(action)
        self.prev_obs = obs.copy()
        self.prev_state_hash = self.state_hash

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        # Log step
        if self.traj_logger:
            self.traj_logger.log_step(
                step_idx=self.step_count,
                url=obs.get('url', ''),
                obs_hash=obs_hash,
                action=action,
                action_source=action_source,
                last_action_error=obs.get('last_action_error', ''),
                elapsed_ms=elapsed_ms,
                manager_called=False,
                cache_hit=False,
                obs_summary=obs_summary,
                actionable_elements=elements,
                valid_bids=sorted(valid_bids),
                progress_score=progress_info.progress_score,
                milestones=progress_info.milestones,
            )

        # Build info
        info = {
            "step": self.step_count,
            "action_source": action_source,
            "cache_hit": False,
            "manager_called": False,
            "page_type": page_type,
            "loop_detected": loop_detected,
            "progress_score": progress_info.progress_score,
            "milestones": progress_info.milestones,
        }

        return action, info

    def _deterministic_fallback(self, sig) -> Optional[Tuple[str, str]]:
        """Deterministic fallback strategy for loops/errors."""
        if not sig:
            return None

        # 1. Dialogs/Alerts - Click confirm/close
        if sig.dialog_count > 0 or sig.alert_count > 0:
            for btn in sig.raw.get('buttons', []):
                if not btn.get('disabled') and btn.get('name'):
                    if re.search(r"(accept|agree|ok|continue|close|allow|confirm|yes|i understand|dismiss)", btn['name'], re.IGNORECASE):
                        if btn.get('bid'):
                            return f'click("{btn["bid"]}")', "fallback_dialog"

        # 2. Required/Invalid fields - Fill them
        if sig.required_field_count > 0 or sig.invalid_field_count > 0:
            targets = sig.raw.get('requiredFields', []) + sig.raw.get('invalidFields', [])
            for field in targets:
                bid = field.get('bid')
                if not bid:
                    continue
                role = field.get('role', '')
                name = field.get('name', '').lower()

                if field.get('value'):
                    continue

                val = "dummy"
                if "email" in name or "email" in role:
                    val = "example@example.com"
                elif "password" in name:
                    val = "password123"
                elif "zip" in name or "postal" in name:
                    val = "90210"
                elif "name" in name:
                    val = "John Doe"
                elif "address" in name:
                    val = "123 Main St"
                elif "phone" in name:
                    val = "555-555-5555"
                elif "search" in name or "search" in role:
                    continue

                return f'fill("{bid}", "{val}")', "fallback_fill"

        # 3. Submit/Continue - Click enabled progress buttons
        for btn in sig.raw.get('buttons', []):
            if not btn.get('disabled') and btn.get('name'):
                if re.search(r"(submit|sign in|log in|login|continue|next|book|place order|send|checkout|pay|proceed)", btn['name'], re.IGNORECASE):
                    if btn.get('bid'):
                        return f'click("{btn["bid"]}")', "fallback_submit"

        # 4. Backtrack - Hard reset
        return "go_back()", "fallback_backtrack"

    def _check_recovery_policies(
        self,
        loop_detected: bool,
        last_action_error: str,
        obs_summary: str,
        valid_bids: set = None
    ) -> Optional[Tuple[str, str]]:
        """Check recovery policies and return action if triggered."""
        valid_bids = valid_bids or set()

        # Priority 0: Obstruction Clearing
        if last_action_error and "intercepts pointer events" in last_action_error:
            for line in last_action_error.split('\n'):
                if "intercepts pointer events" in line:
                    match = re.search(r'bid=["\'](\d+)["\']', line)
                    if match:
                        obstruction_bid = match.group(1)
                        logger.info(f"Obstruction detected (bid {obstruction_bid}), attempting to clear it")
                        return f'click("{obstruction_bid}")', "recovery_obstruction_clearing"

        # Priority 1: Deterministic Fallback on Loop or Stuck
        if loop_detected or self.no_progress_count >= self.config.no_progress_threshold:
            fallback = self._deterministic_fallback(self.ax_signals)
            if fallback:
                action, source = fallback
                if self.action_history and action == self.action_history[-1]:
                    logger.info(f"Fallback loop detected ({action}), escalating to go_back")
                    return "go_back()", "recovery_fallback_loop"
                self.no_progress_count = 0
                return fallback

        # Policy 1: No progress for N steps
        if self.no_progress_count >= self.config.no_progress_threshold:
            self.no_progress_count = 0
            if valid_bids and len(valid_bids) > 3:
                bid_list = sorted(valid_bids)
                idx = self.step_count % len(bid_list)
                return f'click("{bid_list[idx]}")', "recovery_no_progress_click"
            return "go_back()", "recovery_no_progress"

        # Policy 2: Same error repeating
        if self.consecutive_error_count >= 2:
            self.consecutive_error_count = 0

            fallback = self._deterministic_fallback(self.ax_signals)
            if fallback:
                return fallback[0], "recovery_error_fallback"

            if valid_bids:
                bid_list = sorted(valid_bids)
                idx = (self.step_count + 1) % len(bid_list)
                return f'click("{bid_list[idx]}")', "recovery_error_click"
            return "scroll(0, 300)", "recovery_error_repeat"

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            "total_steps": self.step_count,
            "mode": self.mode,
            "invalid_id_count": self.invalid_id_count,
            "recovery_action_count": self.recovery_action_count,
            "invalid_id_rate": self.invalid_id_count / max(1, self.step_count)
        }
        if hasattr(self.worker, 'get_invalid_id_rate'):
            stats["worker_invalid_id_rate"] = self.worker.get_invalid_id_rate()
        return stats
