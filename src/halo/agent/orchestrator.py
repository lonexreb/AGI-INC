"""Orchestrator for HALO Agent.

Implements the decision hierarchy:
1) Macro replay (if active)
2) VAC cache hit
3) Worker action
4) Manager (only when gated)
"""

import os
import re
import time
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from ..obs import (
    summarize_observation, extract_actionable_nodes, get_obs_hash,
    build_state_key, state_key_hash, classify_page_type
)
from ..policy import (
    WorkerPolicy, ManagerPolicy, GatingController, ManagerDecision,
    QwenWorkerPolicy, create_qwen_worker_policy
)
from ..cache import VerifiedActionCache, MacroReplayCache
from ..verify import ActionVerifier, LoopDetector
from ..logging import TrajectoryLogger

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator."""
    use_cache: bool = True
    use_macros: bool = True
    use_manager: bool = True
    worker_model: str = "gpt-4o-mini"
    manager_model: str = "gpt-4o"
    max_steps: int = 70
    # Manager warm start: call manager at step 0 and 1
    manager_warm_start: bool = True
    # Recovery policies
    no_progress_threshold: int = 3  # go_back after N no-progress steps
    enable_recovery_policies: bool = True
    # Manager gating overrides
    always_call_manager: bool = False
    # Qwen worker backend configuration
    qwen_backend: str = "vllm"
    qwen_base_url: str = "http://localhost:8000/v1"


class Orchestrator:
    """Main orchestrator implementing HALO decision hierarchy."""

    def __init__(self, config: OrchestratorConfig = None, mode: str = "halo_cache"):
        self.config = config or OrchestratorConfig()
        self.mode = mode

        # Initialize components based on mode
        # Check if this is a Qwen mode
        if mode.startswith('qwen_worker'):
            qwen_mode = mode.replace('qwen_worker_', '')
            qwen_backend = self.config.qwen_backend or "vllm"
            qwen_base_url = self.config.qwen_base_url or "http://localhost:8000/v1"

            default_model_name = os.environ.get("HALO_WORKER_MODEL") or "Qwen/Qwen2.5-3B-Instruct"
            requested = self.config.worker_model

            if qwen_mode in {"bc", "dpo"}:
                adapter_path = None
                model_name = default_model_name
                if requested and (
                    os.path.exists(requested)
                    or requested.startswith("checkpoints/")
                    or requested.startswith("models/")
                    or requested.endswith("_lora")
                ):
                    adapter_path = requested
                else:
                    model_name = requested or default_model_name
                    adapter_path = (
                        "checkpoints/qwen_bc_lora" if qwen_mode == "bc" else "checkpoints/qwen_dpo_lora"
                    )

                self.worker = QwenWorkerPolicy(
                    model_name=model_name,
                    backend=qwen_backend,
                    base_url=qwen_base_url,
                    adapter_path=adapter_path,
                )
            else:
                self.worker = create_qwen_worker_policy(
                    mode=qwen_mode,
                    model_name=requested or default_model_name,
                    backend=qwen_backend,
                    base_url=qwen_base_url,
                )

            logger.info(
                f"Using Qwen worker policy (mode: {qwen_mode}, backend: {qwen_backend}, model: {self.worker.model_name})"
            )
        else:
            self.worker = WorkerPolicy(model=self.config.worker_model)

        if self.config.use_manager:
            self.manager = ManagerPolicy(model=self.config.manager_model)
        else:
            self.manager = None

        if self.config.use_cache:
            self.vac = VerifiedActionCache()
        else:
            self.vac = None
        
        if self.config.use_macros:
            self.macro_cache = MacroReplayCache()
        else:
            self.macro_cache = None

        self.verifier = ActionVerifier()
        self.loop_detector = LoopDetector()
        self.gating = GatingController()

        # State
        self.step_count = 0
        self.action_history = []
        self.prev_obs: Optional[Dict] = None
        self.current_manager_guidance: Optional[ManagerDecision] = None
        self.goal: str = ""
        
        # Recovery state
        self.no_progress_count = 0
        self.last_error_action = ""
        self.consecutive_error_count = 0
        self.last_url = ""
        
        # Stats
        self.invalid_id_count = 0
        self.recovery_action_count = 0

        # Trajectory logger (set externally)
        self.traj_logger: Optional[TrajectoryLogger] = None

    def reset(self, goal: str = ""):
        """Reset for new episode."""
        self.step_count = 0
        self.action_history = []
        self.prev_obs = None
        self.current_manager_guidance = None
        self.goal = goal
        self.no_progress_count = 0
        self.last_error_action = ""
        self.consecutive_error_count = 0
        self.last_url = ""
        self.invalid_id_count = 0
        self.recovery_action_count = 0
        self.loop_detector.reset()
        self.gating.reset()
        if self.vac:
            self.vac.reset_stats()
        if self.macro_cache:
            self.macro_cache.abort_macro()

    def get_action(self, obs: Dict) -> Tuple[str, Dict[str, Any]]:
        """Get next action following decision hierarchy.

        Args:
            obs: Current observation

        Returns:
            Tuple of (action_string, info_dict)
        """
        start_time = time.time()
        self.step_count += 1

        # Extract goal if not set
        if not self.goal and 'goal_object' in obs:
            goal_obj = obs['goal_object']
            if isinstance(goal_obj, list):
                self.goal = ' '.join(
                    item.get('text', '') for item in goal_obj
                    if isinstance(item, dict) and item.get('type') == 'text'
                )

        # Build state key and page type
        state_key = build_state_key(obs)
        state_key_str = state_key.to_string()
        page_type = classify_page_type(obs.get('url', ''), obs.get('title', ''))
        obs_hash = get_obs_hash(obs)

        # Update loop detector
        self.loop_detector.add_state(state_key_str, self.action_history[-1] if self.action_history else "")
        loop_detected = self.loop_detector.is_loop()

        # Verify previous action if applicable
        if self.prev_obs is not None:
            verification = self.verifier.verify(
                self.prev_obs, obs,
                self.action_history[-1] if self.action_history else "",
                ""
            )
            # Update cache if using
            if self.vac and self.action_history:
                prev_state_key = build_state_key(self.prev_obs).to_string()
                self.vac.put(
                    prev_state_key,
                    self.action_history[-1],
                    "",
                    verified=verification.success
                )

        # Extract actionable elements and valid bids
        axtree = obs.get('axtree_object')
        elements = extract_actionable_nodes(axtree, top_k=30) if axtree else []
        valid_bids = {str(e['bid']) for e in elements}
        
        # Debug logging
        if not axtree:
            logger.warning("No axtree_object in observation!")
        elif not elements:
            # Log axtree structure for debugging
            if isinstance(axtree, dict):
                keys = list(axtree.keys())[:10]
                logger.warning(f"No actionable elements! axtree keys: {keys}")
                # Check if it's a flattened string representation
                if 'nodes' in axtree:
                    logger.warning(f"axtree has 'nodes' key with {len(axtree.get('nodes', []))} items")
            else:
                logger.warning(f"axtree is not a dict: {type(axtree)}")
        else:
            logger.info(f"Extracted {len(elements)} elements, valid_bids: {sorted(valid_bids)[:5]}...")

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

        # Decision hierarchy
        action = None
        action_source = "worker"
        cache_hit = False
        manager_called = False
        recovery_action = False

        # 0. Recovery policies (highest priority)
        if self.config.enable_recovery_policies:
            recovery_result = self._check_recovery_policies(
                loop_detected, last_action_error, obs_summary, valid_bids
            )
            if recovery_result:
                action, action_source = recovery_result
                recovery_action = True
                self.recovery_action_count += 1
                logger.info(f"Recovery action: {action} (source: {action_source})")

        # 1. Manager warm start (step 0 and 1)
        if action is None and self.manager and self.config.manager_warm_start and not self.config.always_call_manager:
            if self.step_count <= 2:  # Steps 1 and 2 (after increment)
                manager_called = True
                self.current_manager_guidance = self.manager.get_decision(
                    obs_summary=obs_summary,
                    goal=self.goal,
                    action_history=self.action_history,
                    last_error=last_action_error,
                    loop_detected=loop_detected,
                    page_type=page_type
                )
                logger.info(f"Manager warm start: {self.current_manager_guidance.subgoal}")

        if action is None and self.manager and self.config.always_call_manager:
            manager_called = True
            self.current_manager_guidance = self.manager.get_decision(
                obs_summary=obs_summary,
                goal=self.goal,
                action_history=self.action_history,
                last_error=last_action_error,
                loop_detected=loop_detected,
                page_type=page_type,
            )
            logger.info(f"Manager always-call: {self.current_manager_guidance.subgoal}")

        # 2. Macro replay (if active)
        if action is None and self.macro_cache and self.macro_cache.is_active():
            macro_result = self.macro_cache.get_next_action(elements)
            if macro_result:
                action, postcondition = macro_result
                action_source = "macro"
                logger.info(f"Macro action: {action}")

        # 3. VAC cache hit (validate action still uses valid element IDs)
        # Skip cache during loops to allow fresh decisions
        if action is None and self.vac and not loop_detected:
            cached_action = self.vac.get(state_key_str)
            if cached_action:
                # Validate cached action uses current valid bids
                bid_match = re.search(r'["\']([^"\']+)["\']', cached_action)
                if bid_match:
                    cached_bid = bid_match.group(1)
                    if cached_bid in valid_bids:
                        # Also check action isn't same as last action (avoid replay loops)
                        if self.action_history and cached_action == self.action_history[-1]:
                            logger.info(f"Cache skip: same as last action {cached_action}")
                        else:
                            action = cached_action
                            action_source = "cache"
                            cache_hit = True
                            logger.info(f"Cache hit: {action}")
                    else:
                        logger.info(f"Cache miss: stale bid {cached_bid} not in current page")
                else:
                    # Action doesn't use element ID (e.g., scroll, go_back)
                    if self.action_history and cached_action == self.action_history[-1]:
                        logger.info(f"Cache skip: same as last action {cached_action}")
                    else:
                        action = cached_action
                        action_source = "cache"
                        cache_hit = True
                        logger.info(f"Cache hit: {action}")
        elif action is None and self.vac and loop_detected:
            logger.info("Cache skip: loop detected, forcing fresh decision")

        # 4. Check gating for manager (errors/loops/high-stakes)
        if action is None and self.manager and not self.config.always_call_manager:
            gate_decision = self.gating.check(
                obs, self.action_history, loop_detected, page_type
            )
            if gate_decision.should_call_manager:
                manager_called = True
                self.current_manager_guidance = self.manager.get_decision(
                    obs_summary=obs_summary,
                    goal=self.goal,
                    action_history=self.action_history,
                    last_error=last_action_error,
                    loop_detected=loop_detected,
                    page_type=page_type
                )
                logger.info(f"Manager decision: {self.current_manager_guidance.subgoal}")

        # 5. Worker action
        if action is None:
            guidance = None
            if self.current_manager_guidance:
                guidance = {
                    "subgoal": self.current_manager_guidance.subgoal,
                    "skill": self.current_manager_guidance.skill
                }

            worker_result = self.worker.get_action(
                obs_summary=obs_summary,
                goal=self.goal,
                action_history=self.action_history,
                manager_guidance=guidance,
                valid_bids=valid_bids,
                last_action_error=last_action_error
            )
            action = worker_result["action"]
            if worker_result.get("was_repaired"):
                self.invalid_id_count += 1
            action_source = "worker" if not manager_called else "worker_guided"

        # Record action
        self.action_history.append(action)
        self.prev_obs = obs.copy()

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
                manager_called=manager_called,
                cache_hit=cache_hit,
                obs_summary=obs_summary,
                actionable_elements=elements,
                valid_bids=sorted(valid_bids),
            )

        # Build info
        info = {
            "step": self.step_count,
            "action_source": action_source,
            "cache_hit": cache_hit,
            "manager_called": manager_called,
            "page_type": page_type,
            "loop_detected": loop_detected
        }

        return action, info

    def _check_recovery_policies(
        self,
        loop_detected: bool,
        last_action_error: str,
        obs_summary: str,
        valid_bids: set = None
    ) -> Optional[Tuple[str, str]]:
        """Check recovery policies and return action if triggered.
        
        Returns:
            Tuple of (action, source) if recovery triggered, None otherwise
        """
        valid_bids = valid_bids or set()
        
        # Policy 1: No progress for N steps -> try clicking something or go_back
        if self.no_progress_count >= self.config.no_progress_threshold:
            self.no_progress_count = 0  # Reset counter
            # Try clicking a random element if available
            if valid_bids and len(valid_bids) > 3:
                # Pick a different element each time
                bid_list = sorted(valid_bids)
                idx = self.step_count % len(bid_list)
                return f'click("{bid_list[idx]}")', "recovery_no_progress_click"
            return "go_back()", "recovery_no_progress"
        
        # Policy 2: Same error repeating -> try a different element
        if self.consecutive_error_count >= 2:
            self.consecutive_error_count = 0  # Reset counter
            if valid_bids:
                bid_list = sorted(valid_bids)
                idx = (self.step_count + 1) % len(bid_list)
                return f'click("{bid_list[idx]}")', "recovery_error_click"
            return "scroll(0, 300)", "recovery_error_repeat"
        
        # Policy 3: Loop detected with many scroll actions -> try clicking
        if loop_detected:
            # Check if recent actions are mostly scrolls
            recent_scrolls = sum(1 for a in self.action_history[-5:] if 'scroll' in a.lower())
            if recent_scrolls >= 3 and valid_bids:
                bid_list = sorted(valid_bids)
                idx = self.step_count % len(bid_list)
                return f'click("{bid_list[idx]}")', "recovery_loop_click"
        
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            "total_steps": self.step_count,
            "manager_calls": self.gating.manager_calls,
            "mode": self.mode,
            "invalid_id_count": self.invalid_id_count,
            "recovery_action_count": self.recovery_action_count,
            "invalid_id_rate": self.invalid_id_count / max(1, self.step_count)
        }
        if self.vac:
            stats["cache"] = self.vac.stats()
        if self.macro_cache:
            stats["macros"] = self.macro_cache.stats()
        if hasattr(self.worker, 'get_invalid_id_rate'):
            stats["worker_invalid_id_rate"] = self.worker.get_invalid_id_rate()
        return stats
