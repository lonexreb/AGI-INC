#!/usr/bin/env python3
"""
Evaluation script for running HALO-Agent on a task subset.

Discovers valid v2 tasks dynamically and runs evaluation with proper error handling.

CRITICAL CONSTRAINTS:
- Always uses task_version v2 (SDK defaults to v1 if omitted)
- Browser viewport: 1280x720
- Observations: AXTree + Screenshot only (no HTML)

Usage:
    python scripts/eval_subset.py --mode baseline_worker
    python scripts/eval_subset.py --tasks v2.omnizon-13,v2.gomail-1 --mode hierarchy_vac_macros
    python scripts/eval_subset.py --max_steps 25 --debug
"""

import argparse
import json
import os
import random
import statistics
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


def check_api_key():
    """Check if OPENAI_API_KEY is set, exit with error if not."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Please set it in your .env file or environment variables.")
        print("Run 'python scripts/check_env.py' for diagnostics.")
        sys.exit(1)


def load_experiments_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f) if config_path.endswith(".json") else __import__("yaml").safe_load(f) or {}


def get_experiment_from_config(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    defaults = config.get("defaults", {})
    experiments = config.get("experiments", [])
    for exp in experiments:
        if isinstance(exp, dict) and exp.get("name") == experiment_name:
            return {**defaults, **exp}
    available = [e.get("name") for e in experiments if isinstance(e, dict) and e.get("name")]
    raise ValueError(f"Unknown experiment '{experiment_name}'. Available: {sorted(available)}")


def should_require_openai_key(modes: List[str], exp_cfg: Optional[Dict[str, Any]]) -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return False
    if exp_cfg is None:
        return True
    if exp_cfg.get("use_manager") is not False:
        return True
    return not all(str(m).startswith("qwen_worker") for m in (modes or []))


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from halo.sdk.task_registry import load_real_task_names

SUPPORTED_MODES = [
    'baseline_worker',
    'hierarchy_mgr_gate',
    'hierarchy_vac',
    'hierarchy_vac_macros',
    # Qwen-based modes
    'qwen_worker_zero',
    'qwen_worker_bc',
    'qwen_worker_dpo',
    'qwen_worker_grpo',
]


def log_error(error_file: Path, task_id: str, error: Exception, tb: str):
    """Log error to JSONL file."""
    error_record = {
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": tb
    }
    with open(error_file, 'a') as f:
        f.write(json.dumps(error_record) + "\n")


def run_single_task_safe(
    task_name: str,
    mode: str,
    headless: bool,
    max_steps: int,
    run_id: str,
    results_dir: str,
    error_file: Path,
    agent_overrides: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """Run a single task with error handling.
    
    Returns:
        Result dict with keys: success, error, steps, wall_time, etc.
    """
    from halo.sdk.harness import run_single_task, ensure_v2_task
    
    task_name = ensure_v2_task(task_name)
    start_time = datetime.now()
    
    try:
        if debug:
            print(f"  [DEBUG] Starting task: {task_name}")
        
        result = run_single_task(
            task_name=task_name,
            mode=mode,
            headless=headless,
            max_steps=max_steps,
            run_id=run_id,
            results_dir=results_dir,
            **(agent_overrides or {})
        )
        
        wall_time = (datetime.now() - start_time).total_seconds()
        result['wall_time'] = wall_time
        result['init_error'] = False
        
        if debug:
            print(f"  [DEBUG] Completed: success={result.get('success', False)}, steps={result.get('n_steps', 0)}")
        
        return result
        
    except Exception as e:
        wall_time = (datetime.now() - start_time).total_seconds()
        tb = traceback.format_exc()
        
        log_error(error_file, task_name, e, tb)
        
        if debug:
            print(f"  [DEBUG] Error: {type(e).__name__}: {e}")
            print(tb)
        
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "init_error": True,
            "wall_time": wall_time,
            "n_steps": 0
        }


def run_mode(
    mode: str,
    tasks: List[str],
    run_id: str,
    headless: bool = True,
    max_steps: int = 70,
    results_dir: str = None,
    run_name: Optional[str] = None,
    agent_overrides: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """Run all tasks in a specific mode with error handling."""
    from halo.sdk.harness import ensure_v2_task
    
    label = run_name or mode
    mode_run_id = f"{run_id}_{label}"
    run_results_dir = Path(results_dir) if results_dir else Path("results") / run_id
    run_results_dir.mkdir(parents=True, exist_ok=True)
    
    error_file = run_results_dir / "errors.jsonl"
    
    print(f"\n{'='*70}")
    print(f"Running mode: {label} ({len(tasks)} tasks)")
    print(f"Run ID: {mode_run_id}")
    print(f"Max Steps: {max_steps}")
    print(f"Error log: {error_file}")
    print(f"{'='*70}")
    
    tasks = [ensure_v2_task(t) for t in tasks]
    results = {}
    
    for i, task_name in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task_name}")
        
        result = run_single_task_safe(
            task_name=task_name,
            mode=mode,
            headless=headless,
            max_steps=max_steps,
            run_id=mode_run_id,
            results_dir=str(run_results_dir / label),
            error_file=error_file,
            agent_overrides=agent_overrides,
            debug=debug
        )
        
        results[task_name] = result
        
        status = "✓" if result.get('success') else ("⚠ ERROR" if result.get('init_error') else "✗")
        print(f"  {status} ({result.get('wall_time', 0):.1f}s, {result.get('n_steps', 0)} steps)")
    
    return results


def compute_metrics(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from results.
    
    Computes:
    - valid_tasks: Tasks that ran without init/agent errors
    - init_or_agent_errors: Tasks that crashed
    - successes: Valid tasks with reward > 0
    - failures: Valid tasks with reward = 0
    - success_rate: successes / valid_tasks
    - median_steps/time across ALL valid tasks
    """
    valid_tasks = 0
    init_or_agent_errors = 0
    successes = 0
    failures = 0
    steps = []
    wall_times = []
    invalid_actions = []
    manager_calls = []
    cache_hits = []
    cache_misses = []
    
    for task_name, result in results.items():
        if result.get('init_error') or result.get('error_type'):
            init_or_agent_errors += 1
            continue
        
        valid_tasks += 1
        success = result.get('success', False)
        
        if success:
            successes += 1
        else:
            failures += 1
        
        if 'n_steps' in result:
            steps.append(result['n_steps'])
        elif 'steps' in result:
            steps.append(result['steps'])
        
        if 'wall_time' in result:
            wall_times.append(result['wall_time'])
        
        agent_stats = result.get('agent_stats', {})
        if 'invalid_action_rate' in agent_stats:
            invalid_actions.append(agent_stats['invalid_action_rate'])
        if 'manager_calls' in agent_stats:
            manager_calls.append(agent_stats['manager_calls'])
        if 'cache' in agent_stats:
            cache_stats = agent_stats['cache']
            cache_hits.append(cache_stats.get('hits', 0))
            cache_misses.append(cache_stats.get('misses', 0))
    
    total_cache = sum(cache_hits) + sum(cache_misses)
    
    return {
        "total_tasks": len(results),
        "valid_tasks": valid_tasks,
        "init_or_agent_errors": init_or_agent_errors,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / valid_tasks if valid_tasks > 0 else 0,
        "median_steps": statistics.median(steps) if steps else 0,
        "mean_steps": statistics.mean(steps) if steps else 0,
        "median_wall_time": statistics.median(wall_times) if wall_times else 0,
        "mean_wall_time": statistics.mean(wall_times) if wall_times else 0,
        "invalid_action_rate": statistics.mean(invalid_actions) if invalid_actions else 0,
        "manager_call_rate": sum(manager_calls) / max(1, sum(steps)) if manager_calls else 0,
        "cache_hit_rate": sum(cache_hits) / total_cache if total_cache > 0 else 0,
    }


def save_results(
    run_id: str,
    mode: str,
    results: Dict[str, Dict],
    metrics: Dict[str, Any],
    results_dir: Path
):
    """Save results to JSON file."""
    mode_dir = results_dir / f"{run_id}_{mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "run_id": run_id,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "tasks": list(results.keys())
    }
    
    with open(mode_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(mode_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved results to {mode_dir}")


def generate_readme(
    run_id: str,
    all_metrics: Dict[str, Dict],
    results_dir: Path,
    max_steps: int = 70
):
    """Generate comparison README.md."""
    readme_path = results_dir / "README.md"
    
    lines = [
        "# HALO-Agent Evaluation Results",
        "",
        f"**Run ID:** {run_id}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Comparison Table",
        "",
        "| Mode | Valid Tasks | Errors | Successes | Failures | Success Rate | Median Steps | Median Time |",
        "|------|-------------|--------|-----------|----------|--------------|--------------|-------------|",
    ]
    
    mode_order = ['baseline_worker', 'hierarchy_mgr_gate', 'hierarchy_vac', 'hierarchy_vac_macros']
    for mode in mode_order:
        if mode in all_metrics:
            m = all_metrics[mode]
            lines.append(
                f"| {mode} | {m['valid_tasks']} | {m['init_or_agent_errors']} | "
                f"{m['successes']} | {m['failures']} | {m['success_rate']:.1%} | "
                f"{m['median_steps']:.1f} | {m['median_wall_time']:.1f}s |"
            )
    
    lines.extend([
        "",
        "## Metrics Definitions",
        "",
        "- **Valid Tasks**: Tasks that ran without initialization or agent errors",
        "- **Errors**: Tasks that crashed during init or execution",
        "- **Successes**: Valid tasks that achieved reward > 0",
        "- **Failures**: Valid tasks that achieved reward = 0",
        "- **Success Rate**: successes / valid_tasks",
        "",
        "## Configuration",
        "",
        "- **Task Version:** v2 (CRITICAL: SDK defaults to v1 if omitted)",
        f"- **Max Steps:** {max_steps}",
        "- **Browser:** 1280x720",
        "- **Observations:** AXTree + Screenshot (no HTML)",
        "",
    ])
    
    with open(readme_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nGenerated {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run HALO-Agent evaluation on task subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run baseline_worker on 50 random v2 tasks
    python eval_subset.py --mode baseline_worker
    
    # Run specific tasks
    python eval_subset.py --tasks v2.omnizon-13,v2.gomail-1 --mode hierarchy_vac_macros
    
    # Run with debug output
    python eval_subset.py --mode baseline_worker --debug --max_steps 10
    
    # Run with visible browser
    python eval_subset.py --mode baseline_worker --headless false
"""
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiments config YAML (default: none)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name from experiments.yaml (sets agent overrides)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task IDs (default: sample from discovered v2 tasks)"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Filter tasks by site/type (e.g., omnizon, gomail)"
    )
    parser.add_argument(
        "--task_version",
        type=str,
        default="v2",
        help="Task version (default: v2). WARNING: SDK defaults to v1 if omitted!"
    )
    parser.add_argument(
        "--task_registry",
        type=str,
        default=None,
        help=(
            "Path to REAL task registry snapshot JSON. If omitted, uses configs/real_<task_version>_task_registry.json "
            "when present, otherwise falls back to dynamic discovery."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs='+',
        default=["baseline_worker"],
        choices=SUPPORTED_MODES,
        help="Mode(s) to run (default: baseline_worker)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=70,
        help="Max steps per task (default: 70)"
    )
    parser.add_argument(
        "--headless",
        type=str,
        default="true",
        help="Run headless browser (default: true)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of tasks to sample if --tasks not provided (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for task sampling (for reproducibility)"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Custom run ID (default: timestamp)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )
    
    args = parser.parse_args()

    config: Optional[Dict[str, Any]] = None
    exp_cfg: Optional[Dict[str, Any]] = None
    if args.config and args.experiment:
        config = load_experiments_config(args.config)
        exp_cfg = get_experiment_from_config(config, args.experiment)

    if should_require_openai_key(args.mode, exp_cfg):
        check_api_key()
    
    if args.task_version != "v2":
        print(f"WARNING: task_version={args.task_version} specified. SDK defaults to v1 if omitted!")
        print("This evaluation is designed for v2 tasks only.")
    
    headless = args.headless.lower() in ('true', '1', 'yes')
    
    run_id = args.run_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        all_tasks, source = load_real_task_names(task_version="v2", task_registry=args.task_registry)
        if not all_tasks:
            print("ERROR: No v2 tasks discovered!")
            sys.exit(1)
        
        print(f"Discovered {len(all_tasks)} v2 tasks from {source}")
        
        if args.task_type:
            all_tasks = [t for t in all_tasks if args.task_type in t]
            print(f"Filtered to {len(all_tasks)} tasks matching '{args.task_type}'")
        
        if len(all_tasks) > args.sample_size:
            if args.seed is not None:
                random.seed(args.seed)
                print(f"Using random seed: {args.seed}")
            tasks = random.sample(all_tasks, args.sample_size)
            print(f"Sampled {args.sample_size} tasks")
        else:
            tasks = all_tasks
    
    # Save task list for reproducibility
    tasks_file = results_dir / "tasks.txt"
    with open(tasks_file, 'w') as f:
        for task in sorted(tasks):
            f.write(task + "\n")
    print(f"Task list saved to: {tasks_file}")
    
    print(f"\n{'='*70}")
    print(f"HALO-Agent Evaluation")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Tasks: {len(tasks)}")
    print(f"Modes: {args.mode}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Headless: {headless}")
    print(f"Debug: {args.debug}")
    print(f"Results: {results_dir}")
    print()
    
    if args.dry_run:
        print("DRY RUN - Would execute:")
        for mode in args.mode:
            print(f"  Mode: {mode}")
        print(f"\nTasks ({len(tasks)}):")
        for task in tasks[:10]:
            print(f"  - {task}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return
    
    all_results = {}
    all_metrics = {}
    
    for mode in args.mode:
        try:
            agent_overrides = None
            run_name = None
            if exp_cfg is not None:
                run_name = exp_cfg.get("name") or mode
                agent_overrides = {
                    k: exp_cfg[k]
                    for k in [
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
                    ]
                    if k in exp_cfg
                }
                if mode.startswith("qwen_worker") and exp_cfg.get("use_manager") is False:
                    agent_overrides["traj_group"] = str(exp_cfg.get("name") or args.experiment)

            results = run_mode(
                mode=mode,
                tasks=tasks,
                run_id=run_id,
                headless=headless,
                max_steps=args.max_steps,
                results_dir=str(results_dir),
                run_name=run_name,
                agent_overrides=agent_overrides,
                debug=args.debug
            )
            
            metrics = compute_metrics(results)
            all_results[mode] = results
            all_metrics[mode] = metrics
            
            save_results(run_id, mode, results, metrics, results_dir)
            
            print(f"\n{mode} Summary:")
            print(f"  Valid Tasks: {metrics['valid_tasks']}/{metrics['total_tasks']}")
            print(f"  Errors: {metrics['init_or_agent_errors']}")
            print(f"  Successes: {metrics['successes']}")
            print(f"  Failures: {metrics['failures']}")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            print(f"  Median Steps: {metrics['median_steps']:.1f}")
            print(f"  Median Time: {metrics['median_wall_time']:.1f}s")
            
        except Exception as e:
            print(f"\nFATAL Error running mode {mode}: {e}")
            traceback.print_exc()
    
    if all_metrics:
        generate_readme(run_id, all_metrics, results_dir, args.max_steps)
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete!")
    print(f"Results: {results_dir}")
    print(f"Errors: {results_dir}/errors.jsonl")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
