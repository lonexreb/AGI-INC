#!/usr/bin/env python3
"""Rollout sampler for collecting diverse trajectories.

Runs N rollouts per task with explicit seed control and optional exploration
(temperature > 0) so that each rollout is saved under a distinct run_id.

Examples:
  python scripts/rollout_sampler.py --task_type gomail --sample_size 5 --rollouts_per_task 3 --seed 42 --task_seed 123 --temperature 0.7

  # Explicit tasks
  python scripts/rollout_sampler.py --tasks v2.omnizon-13,v2.gomail-1 --rollouts_per_task 2 --temperature 0.3
"""

import argparse
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from list_v2_tasks import discover_v2_tasks

SUPPORTED_MODES = [
    "baseline_worker",
    "hierarchy_mgr_gate",
    "hierarchy_vac",
    "hierarchy_vac_macros",
    "expert_rollout",
    "qwen_worker_zero",
    "qwen_worker_bc",
    "qwen_worker_dpo",
    "qwen_worker_grpo",
]


def check_api_key(mode: str) -> None:
    """Check if OPENAI_API_KEY is set when required."""
    # Qwen worker still uses OpenAI manager by default, so keep this strict.
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Please set it in your .env file or environment variables.")
        sys.exit(1)


def make_rollout_run_id(base_run_id: str, task_name: str, rollout_idx: int, task_seed: Optional[int]) -> str:
    safe_task = task_name.replace("/", "_").replace(":", "_")
    seed_part = f"seed{task_seed}" if task_seed is not None else "seedNone"
    return f"{base_run_id}__{safe_task}__r{rollout_idx:03d}__{seed_part}"


def run_single_rollout(
    task_name: str,
    mode: str,
    headless: bool,
    max_steps: int,
    run_id: str,
    results_dir: str,
    task_seed: Optional[int],
    agent_overrides: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    from halo.sdk.harness import ensure_v2_task, run_single_task

    task_name = ensure_v2_task(task_name)

    kwargs: Dict[str, Any] = {}
    if task_seed is not None:
        kwargs["task_seed"] = int(task_seed)

    # Force refresh so repeated rollouts do not accidentally reuse cached results.
    kwargs["force_refresh"] = True

    result = run_single_task(
        task_name=task_name,
        mode=mode,
        headless=headless,
        max_steps=max_steps,
        run_id=run_id,
        results_dir=results_dir,
        **(agent_overrides or {}),
        **kwargs,
    )

    if debug:
        print(
            f"  [DEBUG] {task_name} -> success={result.get('success', False)} "
            f"reward={result.get('cum_reward', result.get('reward', 0))} steps={result.get('n_steps', 0)}"
        )

    return result


def pick_tasks(
    tasks_csv: Optional[str],
    task_type: Optional[str],
    sample_size: int,
    seed: Optional[int],
) -> Tuple[List[str], str]:
    if tasks_csv:
        tasks = [t.strip() for t in tasks_csv.split(",") if t.strip()]
        return tasks, "explicit"

    all_tasks, source = discover_v2_tasks()
    if not all_tasks:
        raise RuntimeError("No v2 tasks discovered")

    if task_type:
        all_tasks = [t for t in all_tasks if task_type in t]

    if seed is not None:
        random.seed(seed)

    if len(all_tasks) > sample_size:
        tasks = random.sample(all_tasks, sample_size)
    else:
        tasks = all_tasks

    return tasks, source


def load_experiments_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_experiment_from_config(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    defaults = config.get("defaults", {})
    experiments = config.get("experiments", [])
    for exp in experiments:
        if exp.get("name") == experiment_name:
            return {**defaults, **exp}
    available = [e.get("name") for e in experiments if isinstance(e, dict) and e.get("name")]
    raise ValueError(f"Unknown experiment '{experiment_name}'. Available: {sorted(available)}")


def get_tasks_from_subset(config: Dict[str, Any], subset_name: str) -> Tuple[List[str], str]:
    subsets = config.get("task_subsets", {})
    subset = subsets.get(subset_name)
    if not isinstance(subset, dict):
        available = [k for k in subsets.keys()]
        raise ValueError(f"Unknown subset '{subset_name}'. Available: {sorted(available)}")

    if "tasks" in subset and subset["tasks"]:
        tasks = [str(t).strip() for t in subset["tasks"] if str(t).strip()]
        return tasks, subset_name

    task_types = subset.get("task_types", [])
    if task_types == "all" or not task_types:
        tasks, source = discover_v2_tasks()
        return tasks, f"{subset_name}:{source}"

    if isinstance(task_types, str):
        task_types = [task_types]

    all_tasks, source = discover_v2_tasks()
    filtered = [t for t in all_tasks if any(tt in t for tt in task_types)]
    return filtered, f"{subset_name}:{source}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N rollouts per task with seed control and exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments.yaml",
        help="Experiments config YAML (default: configs/experiments.yaml)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name from experiments.yaml (sets mode + agent overrides)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Task subset name from experiments.yaml (e.g., quick_test, shopping, full)",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task IDs (v2.*)",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Filter tasks by site/type (e.g., omnizon, gomail, gocalendar)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of tasks to sample if --tasks not provided (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for task sampling (reproducibility)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=SUPPORTED_MODES,
        help="Agent mode to run (default: qwen_worker_zero)",
    )
    parser.add_argument(
        "--headless",
        type=str,
        default=None,
        help="Run headless browser (default: true)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per rollout (default: 70)",
    )

    parser.add_argument(
        "--rollouts_per_task",
        type=int,
        default=3,
        help="Number of rollouts per task (default: 3)",
    )
    parser.add_argument(
        "--task_seed",
        type=int,
        default=None,
        help=(
            "Base task_seed. If provided, each rollout uses task_seed + rollout_idx "
            "(and task index) for reproducibility. If omitted, task_seed=None (SDK random)."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Worker sampling temperature for exploration (default: 0.7)",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Base run ID. Each rollout appends task/seed suffix (default: timestamp)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to store harness results (default: results/<base_run_id>)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    config: Dict[str, Any] = {}
    exp_cfg: Optional[Dict[str, Any]] = None
    if args.experiment or args.subset:
        config = load_experiments_config(args.config)
        if args.experiment:
            exp_cfg = get_experiment_from_config(config, args.experiment)

    mode = args.mode or "qwen_worker_zero"
    if exp_cfg is not None and args.mode is None:
        mode = str(exp_cfg.get("mode", mode))

    check_api_key(mode)

    headless = True
    if args.headless is not None:
        headless = args.headless.lower() in ("true", "1", "yes")
    elif exp_cfg is not None and "headless" in exp_cfg:
        headless = bool(exp_cfg.get("headless"))

    max_steps = int(args.max_steps) if args.max_steps is not None else 70
    if exp_cfg is not None and args.max_steps is None and exp_cfg.get("max_steps") is not None:
        max_steps = int(exp_cfg.get("max_steps"))

    base_run_id = args.run_id or f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    repo_root = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else (repo_root / "results" / base_run_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[str]
    source: str
    if args.tasks:
        tasks, source = pick_tasks(args.tasks, None, 0, None)
    elif args.subset:
        tasks, source = get_tasks_from_subset(config, args.subset)
    else:
        sample_size = int(args.sample_size) if args.sample_size is not None else 50
        tasks, source = pick_tasks(args.tasks, args.task_type, sample_size, args.seed)

    if args.sample_size is not None and args.subset and len(tasks) > int(args.sample_size):
        if args.seed is not None:
            random.seed(args.seed)
        tasks = random.sample(tasks, int(args.sample_size))

    # Save task list for reproducibility
    with open(results_dir / "tasks.txt", "w") as f:
        for t in sorted(tasks):
            f.write(t + "\n")

    print("=" * 70)
    print("ROLLOUT SAMPLER")
    print("=" * 70)
    print(f"Base Run ID: {base_run_id}")
    if args.experiment:
        print(f"Experiment: {args.experiment}")
    print(f"Mode: {mode}")
    print(f"Tasks: {len(tasks)} (source: {source})")
    print(f"Rollouts/task: {args.rollouts_per_task}")
    print(f"Max Steps: {max_steps}")
    print(f"Headless: {headless}")
    print(f"Temperature: {args.temperature}")
    print(f"Base task_seed: {args.task_seed}")
    print(f"Results dir: {results_dir}")
    print()

    agent_overrides: Dict[str, Any] = {}
    if exp_cfg is not None:
        for key in (
            "worker_model",
            "manager_model",
            "use_manager",
            "use_cache",
            "use_macros",
            "manager_warm_start",
            "enable_recovery_policies",
            "always_call_manager",
            "qwen_backend",
            "qwen_base_url",
        ):
            if key in exp_cfg and exp_cfg.get(key) is not None:
                agent_overrides[key] = exp_cfg.get(key)

    agent_overrides["worker_temperature"] = float(args.temperature)

    # Track results (one record per rollout)
    summary_path = results_dir / "rollout_summary.jsonl"

    total = 0
    successes = 0

    for task_i, task_name in enumerate(tasks):
        for r in range(args.rollouts_per_task):
            total += 1

            rollout_seed = None
            if args.task_seed is not None:
                # Deterministic mapping from (task_i, r) -> seed
                rollout_seed = int(args.task_seed) + (task_i * 1000) + r

            rollout_run_id = make_rollout_run_id(
                base_run_id=base_run_id,
                task_name=task_name,
                rollout_idx=r,
                task_seed=rollout_seed,
            )

            print(f"\n[{total}/{len(tasks) * args.rollouts_per_task}] {task_name}  rollout={r+1}/{args.rollouts_per_task}")
            print(f"  run_id={rollout_run_id}")
            if rollout_seed is not None:
                print(f"  task_seed={rollout_seed}")

            try:
                result = run_single_rollout(
                    task_name=task_name,
                    mode=mode,
                    headless=headless,
                    max_steps=max_steps,
                    run_id=rollout_run_id,
                    results_dir=str(results_dir / "harness"),
                    task_seed=rollout_seed,
                    agent_overrides=agent_overrides,
                    debug=args.debug,
                )

                success = bool(result.get("success", False))
                if success:
                    successes += 1

                record = {
                    "timestamp": datetime.now().isoformat(),
                    "base_run_id": base_run_id,
                    "run_id": rollout_run_id,
                    "task_name": task_name,
                    "mode": mode,
                    "experiment": args.experiment,
                    "subset": args.subset,
                    "headless": headless,
                    "max_steps": max_steps,
                    "task_seed": rollout_seed,
                    "temperature": args.temperature,
                    "success": success,
                    "reward": result.get("cum_reward", result.get("reward", 0.0)),
                    "n_steps": result.get("n_steps", 0),
                    "result": result,
                }

                with open(summary_path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")

                status = "✅" if success else "❌"
                print(f"  {status} done (reward={record['reward']}, steps={record['n_steps']})")

            except Exception as e:
                tb = traceback.format_exc()
                print(f"  ⚠ ERROR: {type(e).__name__}: {e}")

                record = {
                    "timestamp": datetime.now().isoformat(),
                    "base_run_id": base_run_id,
                    "run_id": rollout_run_id,
                    "task_name": task_name,
                    "mode": mode,
                    "experiment": args.experiment,
                    "subset": args.subset,
                    "headless": headless,
                    "max_steps": max_steps,
                    "task_seed": rollout_seed,
                    "temperature": args.temperature,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": tb,
                }

                with open(summary_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

    print("\n" + "=" * 70)
    print("SAMPLING SUMMARY")
    print("=" * 70)
    print(f"Total rollouts: {total}")
    print(f"Successes: {successes} ({(100.0 * successes / max(1, total)):.1f}%)")
    print(f"Summary: {summary_path}")
    print(f"Trajectories: data/trajectories/{mode}/<run_id>/")


if __name__ == "__main__":
    main()
