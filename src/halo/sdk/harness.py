"""Wrapper for REAL.harness integration with HALO Agent.

This module provides the harness configuration and execution functions
for running HALO Agent with AGI SDK v0.3.5 on REAL Bench tasks.

CRITICAL: Always use task_version="v2" everywhere.
"""

import dataclasses
from typing import Optional, Dict, Any
from agisdk import REAL
from agisdk.REAL.browsergym.experiments import AbstractAgentArgs

from .agent import HaloAgent
from ..logging import TrajectoryLogger


# Supported modes for evaluation (Qwen3-VL based Online RL)
SUPPORTED_MODES = [
    'gpt4o_baseline',    # GPT-4o for comparison / MCTS critic
    'qwen3vl_base',      # Qwen3-VL-8B base (before training)
    'qwen3vl_grpo',      # Qwen3-VL + Online GRPO LoRA
    'qwen3vl_mcts',      # Qwen3-VL + MCTS-trained LoRA (Agent Q style)
]


def unwrap_single_task_result(result: Any, task_name: str) -> Any:
    """Unwrap REAL.harness results for a single task.

    REAL.harness.run() returns a dict keyed by canonical task name, even when
    running a single task. Our evaluation stack expects a per-task record dict.
    """
    if isinstance(result, dict):
        if task_name in result:
            return result[task_name]
        if len(result) == 1:
            return next(iter(result.values()))
    return result


def derive_task_id(task_name: str) -> str:
    """Derive task_id from canonical v2 task name (e.g., 'v2.omnizon-13' -> 'omnizon-13')."""
    task_name = ensure_v2_task(task_name)
    return task_name.split('.', 1)[1] if '.' in task_name else task_name


@dataclasses.dataclass
class HaloAgentArgs(AbstractAgentArgs):
    """Arguments for the HALO Agent.

    Simplified for Online RL - just policy -> action -> reward.
    Uses Qwen3-VL-8B Vision-Language Model for GUI control.

    Supported modes:
    - gpt4o_baseline: GPT-4o for comparison / MCTS critic
    - qwen3vl_base: Qwen3-VL-8B base (before training)
    - qwen3vl_grpo: Qwen3-VL + Online GRPO LoRA
    - qwen3vl_mcts: Qwen3-VL + MCTS-trained LoRA (Agent Q style)
    """

    agent_name: str = "HaloAgent"

    # Agent configuration
    mode: str = "qwen3vl_base"
    worker_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    worker_temperature: float = 0.0
    max_steps: int = 70  # Default for score-mode; use 25 for speed-mode

    enable_recovery_policies: Optional[bool] = None
    qwen_backend: Optional[str] = None
    qwen_base_url: Optional[str] = None

    # Trajectory logging
    run_id: Optional[str] = None
    task_seed: Optional[int] = None
    log_trajectories: bool = True
    traj_output_dir: str = "data/trajectories"
    traj_group: Optional[str] = None

    def make_agent(self) -> HaloAgent:
        """Create and return an instance of HaloAgent.

        Returns:
            HaloAgent instance configured with the specified parameters
        """
        # Create trajectory logger if enabled
        traj_logger = None
        if self.log_trajectories and self.run_id:
            output_group = self.traj_group or self.mode
            output_dir = f"{self.traj_output_dir}/{output_group}"
            traj_logger = TrajectoryLogger(
                run_id=self.run_id,
                output_dir=output_dir,
                mode=self.mode
            )
            traj_logger.run_metadata = {
                "task_seed": int(self.task_seed) if self.task_seed is not None else None,
            }

        agent = HaloAgent(
            mode=self.mode,
            worker_model=self.worker_model,
            worker_temperature=self.worker_temperature,
            max_steps=self.max_steps,
            enable_recovery_policies=self.enable_recovery_policies,
            qwen_backend=self.qwen_backend,
            qwen_base_url=self.qwen_base_url,
            traj_logger=traj_logger
        )

        return agent


def ensure_v2_task(task_name: str) -> str:
    """Ensure task name has v2. prefix.
    
    CRITICAL: SDK defaults to v1 if version omitted. Always enforce v2.
    """
    if not task_name.startswith('v2.'):
        return f'v2.{task_name}'
    return task_name


def create_harness(
    agent_args: Optional[HaloAgentArgs] = None,
    task_name: str = "v2.omnizon-1",
    headless: bool = True,
    max_steps: int = 70,
    use_axtree: bool = True,
    use_screenshot: bool = True,
    use_html: bool = False,  # CRITICAL: Must be False per constraints
    browser_dimensions: tuple = (1280, 720),  # CRITICAL: Must be 1280x720
    results_dir: Optional[str] = None,
    leaderboard: bool = False,
    run_id: Optional[str] = None,
    mode: str = "qwen3vl_base",
    **kwargs
) -> Any:
    """Create a REAL harness configured for HALO Agent.

    CRITICAL CONSTRAINTS:
    - task_version is always v2 (enforced via task_name prefix)
    - browser_dimensions must be (1280, 720)
    - use_html must be False (AXTree + screenshot only)
    - max_steps default is 70 for score-mode, use 25 for speed-mode

    Args:
        agent_args: HaloAgentArgs instance, creates default if None
        task_name: Name of the task (v2. prefix added if missing)
        headless: Run browser in headless mode (default: True)
        max_steps: Maximum steps per episode (default: 70 for score-mode)
        use_axtree: Include accessibility tree in observations (default: True)
        use_screenshot: Include screenshots in observations (default: True)
        use_html: Include HTML in observations (MUST be False)
        browser_dimensions: Tuple of (width, height) (MUST be 1280x720)
        results_dir: Directory to save results (default: None)
        leaderboard: Submit to leaderboard (default: False)
        run_id: Run ID for identification (default: None)
        mode: Agent mode (default: hierarchy_vac_macros)
        **kwargs: Additional arguments to pass to REAL.harness

    Returns:
        Configured REAL harness instance
    """
    # CRITICAL: Enforce v2 task version
    task_name = ensure_v2_task(task_name)
    
    # CRITICAL: Enforce constraints
    browser_dimensions = (1280, 720)  # Always 1280x720
    use_html = False  # Never use HTML, only AXTree + screenshot
    
    # Create default agent args if not provided
    if agent_args is None:
        agent_args = HaloAgentArgs(
            run_id=run_id, 
            max_steps=max_steps,
            mode=mode
        )
    else:
        if run_id:
            agent_args.run_id = run_id
        agent_args.max_steps = max_steps

    # Build harness configuration
    harness_config = {
        "agentargs": agent_args,
        "task_name": task_name,
        "headless": headless,
        "max_steps": max_steps,
        "use_axtree": use_axtree,
        "use_screenshot": use_screenshot,
        "use_html": use_html,
        "browser_dimensions": browser_dimensions,
    }

    # Add optional parameters
    if results_dir is not None:
        harness_config["results_dir"] = results_dir

    if leaderboard:
        harness_config["leaderboard"] = True
        if run_id is not None:
            harness_config["run_id"] = run_id

    # Merge any additional kwargs
    harness_config.update(kwargs)

    # Create and return the harness
    return REAL.harness(**harness_config)


def run_single_task(
    task_name: str = "v2.omnizon-1",
    mode: str = "qwen3vl_base",
    headless: bool = True,
    max_steps: int = 70,
    task_seed: Optional[int] = None,
    run_id: Optional[str] = None,
    results_dir: Optional[str] = None,
    worker_model: Optional[str] = None,
    worker_temperature: float = 0.0,
    enable_recovery_policies: Optional[bool] = None,
    qwen_backend: Optional[str] = None,
    qwen_base_url: Optional[str] = None,
    traj_group: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run a single task with HALO Agent.

    CRITICAL: Always uses task_version v2 (enforced in create_harness).

    Args:
        task_name: Name of the task (v2. prefix added if missing)
        mode: Agent mode:
            - gpt4o_baseline: GPT-4o for comparison / MCTS critic
            - qwen3vl_base: Qwen3-VL-8B base (before training)
            - qwen3vl_grpo: Qwen3-VL + Online GRPO LoRA
            - qwen3vl_mcts: Qwen3-VL + MCTS-trained LoRA (Agent Q style)
        headless: Run browser in headless mode (default: True)
        max_steps: Maximum steps (default: 70 for score-mode, use 25 for speed-mode)
        run_id: Run ID for logging
        results_dir: Directory to save results
        **kwargs: Additional arguments

    Returns:
        Dictionary with task results
    """
    # Ensure v2 prefix
    task_name = ensure_v2_task(task_name)
    task_id = derive_task_id(task_name)

    agent_args = HaloAgentArgs(
        mode=mode,
        max_steps=max_steps,
        run_id=run_id,
        task_seed=task_seed,
        worker_model=worker_model or "Qwen/Qwen3-VL-8B-Instruct",
        worker_temperature=float(worker_temperature),
        enable_recovery_policies=enable_recovery_policies,
        qwen_backend=qwen_backend,
        qwen_base_url=qwen_base_url,
        traj_group=traj_group,
    )

    harness = create_harness(
        agent_args=agent_args,
        task_name=task_name,
        headless=headless,
        max_steps=max_steps,
        run_id=run_id,
        mode=mode,
        results_dir=results_dir,
        **kwargs
    )

    if task_seed is not None:
        harness.env_args["task_seed"] = int(task_seed)

    results = harness.run()
    record = unwrap_single_task_result(results, task_name)

    if not isinstance(record, dict):
        record = {"raw_result": record}

    record["task_name"] = task_name
    record["task_id"] = task_id

    return record


def run_task_subset(
    task_names: list,
    mode: str = "qwen3vl_base",
    headless: bool = True,
    max_steps: int = 70,
    run_id: Optional[str] = None,
    results_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """Run multiple tasks with HALO Agent.

    CRITICAL: Always uses task_version v2.

    Args:
        task_names: List of task names (v2. prefix added if missing)
        mode: Agent mode (see run_single_task for options)
        headless: Run browser in headless mode (default: True)
        max_steps: Maximum steps (default: 70 for score-mode)
        run_id: Run ID for logging
        results_dir: Directory to save results
        **kwargs: Additional arguments

    Returns:
        Dictionary mapping task names to results
    """
    all_results = {}
    successes = 0
    failures = 0

    for i, task_name in enumerate(task_names):
        # Ensure v2 prefix
        task_name = ensure_v2_task(task_name)
        
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(task_names)}] Running task: {task_name} (mode: {mode})")
        print(f"{'='*60}\n")

        try:
            result = run_single_task(
                task_name=task_name,
                mode=mode,
                headless=headless,
                max_steps=max_steps,
                run_id=run_id,
                results_dir=results_dir,
                **kwargs
            )
            all_results[task_name] = result
            
            success = result.get('success', False)
            if success:
                successes += 1
                print(f"\n✓ Task {task_name} SUCCEEDED")
            else:
                failures += 1
                print(f"\n✗ Task {task_name} FAILED")
                
        except Exception as e:
            print(f"\n✗ Task {task_name} CRASHED: {e}")
            all_results[task_name] = {"success": False, "error": str(e)}
            failures += 1

    # Print summary
    total = len(task_names)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successes}/{total} succeeded ({100*successes/total:.1f}%)")
    print(f"{'='*60}\n")

    return all_results
