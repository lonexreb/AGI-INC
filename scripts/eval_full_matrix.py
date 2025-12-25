#!/usr/bin/env python3
"""
Full benchmark evaluation matrix for HALO-Agent.

Runs FULL REAL v2 benchmark for each technique/mode and outputs comparison.

CRITICAL CONSTRAINTS:
- Always uses task_version v2 (SDK defaults to v1 if omitted)
- Browser viewport: 1280x720
- Observations: AXTree + Screenshot only (no HTML)

Outputs:
- results/<run_id>/matrix.csv - Raw metrics for all modes
- results/<run_id>/matrix.md - Formatted comparison with deltas vs baseline
"""

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import yaml
from dotenv import load_dotenv

load_dotenv()


def check_api_key():
    """Check if OPENAI_API_KEY is set, exit with error if not."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Please set it in your .env file or environment variables.")
        print("Run 'python scripts/check_env.py' for diagnostics.")
        sys.exit(1)


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from list_v2_tasks import get_v2_tasks, discover_v2_tasks
from eval_subset import (
    SUPPORTED_MODES,
    run_mode,
    compute_metrics,
    save_results,
    log_error
)


def load_experiments_config(config_path: str) -> Dict[str, Any]:
    """Load experiments configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dict with 'defaults', 'experiments', 'task_subsets', 'metrics'
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_experiments_from_config(
    config: Dict[str, Any],
    experiment_names: List[str] = None
) -> List[Dict[str, Any]]:
    """Get experiment configs, optionally filtered by name.
    
    Args:
        config: Full config dict
        experiment_names: Optional list of experiment names to filter
        
    Returns:
        List of experiment dicts with defaults applied
    """
    defaults = config.get('defaults', {})
    experiments = config.get('experiments', [])
    
    if experiment_names:
        experiments = [e for e in experiments if e.get('name') in experiment_names]
    
    # Apply defaults to each experiment
    result = []
    for exp in experiments:
        merged = {**defaults, **exp}
        result.append(merged)
    
    return result


def get_tasks_from_config(
    config: Dict[str, Any],
    subset_name: str = None
) -> List[str]:
    """Get task list from config subset.
    
    Args:
        config: Full config dict
        subset_name: Name of task subset (e.g., 'quick_test', 'shopping')
        
    Returns:
        List of task IDs, or None to use all discovered tasks
    """
    if not subset_name:
        return None
    
    subsets = config.get('task_subsets', {})
    subset = subsets.get(subset_name, {})
    
    if 'tasks' in subset:
        return subset['tasks']
    
    task_types = subset.get('task_types', [])
    if task_types == 'all' or not task_types:
        return None
    
    # Filter discovered tasks by type
    all_tasks, _ = discover_v2_tasks()
    return [t for t in all_tasks if any(tt in t for tt in task_types)]


def compute_usefulness_label(
    mode_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any]
) -> str:
    """Compute usefulness label for a mode compared to baseline."""
    if not baseline_metrics or baseline_metrics.get('valid_tasks', 0) == 0:
        return "N/A"
    
    mode_score = mode_metrics['success_rate'] * 100
    baseline_score = baseline_metrics['success_rate'] * 100
    score_delta = mode_score - baseline_score
    
    mode_steps = mode_metrics['median_steps']
    baseline_steps = baseline_metrics['median_steps']
    
    if baseline_steps > 0:
        step_reduction = (baseline_steps - mode_steps) / baseline_steps * 100
    else:
        step_reduction = 0
    
    if score_delta >= 2.0:
        return "Useful"
    elif abs(score_delta) <= 1.0 and step_reduction >= 10.0:
        return "Useful-for-speed"
    else:
        return "Not useful"


def generate_matrix_csv(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: Path,
    run_id: str
):
    """Generate CSV matrix of results."""
    fieldnames = [
        'mode', 'total_tasks', 'valid_tasks', 'init_or_agent_errors',
        'successes', 'failures', 'success_rate', 'median_steps', 'mean_steps',
        'median_wall_time', 'mean_wall_time', 'invalid_action_rate',
        'manager_call_rate', 'cache_hit_rate', 'usefulness_label', 'score_delta'
    ]
    
    baseline_metrics = all_metrics.get('baseline_worker', {})
    baseline_score = baseline_metrics.get('success_rate', 0) * 100
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for mode in SUPPORTED_MODES:
            if mode not in all_metrics:
                continue
            
            m = all_metrics[mode]
            mode_score = m['success_rate'] * 100
            
            row = {
                'mode': mode,
                'total_tasks': m['total_tasks'],
                'valid_tasks': m['valid_tasks'],
                'init_or_agent_errors': m['init_or_agent_errors'],
                'successes': m['successes'],
                'failures': m['failures'],
                'success_rate': f"{m['success_rate']:.4f}",
                'median_steps': f"{m['median_steps']:.1f}",
                'mean_steps': f"{m['mean_steps']:.1f}",
                'median_wall_time': f"{m['median_wall_time']:.1f}",
                'mean_wall_time': f"{m['mean_wall_time']:.1f}",
                'invalid_action_rate': f"{m['invalid_action_rate']:.4f}",
                'manager_call_rate': f"{m['manager_call_rate']:.4f}",
                'cache_hit_rate': f"{m['cache_hit_rate']:.4f}",
                'usefulness_label': compute_usefulness_label(m, baseline_metrics),
                'score_delta': f"{mode_score - baseline_score:+.1f}"
            }
            writer.writerow(row)
    
    print(f"Generated {output_path}")


def generate_matrix_md(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: Path,
    run_id: str,
    max_steps: int
):
    """Generate Markdown matrix with deltas vs baseline."""
    baseline_metrics = all_metrics.get('baseline_worker', {})
    baseline_score = baseline_metrics.get('success_rate', 0) * 100
    baseline_steps = baseline_metrics.get('median_steps', 0)
    
    lines = [
        "# HALO-Agent Full Benchmark Matrix",
        "",
        f"**Run ID:** {run_id}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Max Steps:** {max_steps}",
        "",
        "## Results Matrix",
        "",
        "| Mode | Valid/Total | Errors | Success Rate | Δ Score | Median Steps | Usefulness |",
        "|------|-------------|--------|--------------|---------|--------------|------------|",
    ]
    
    for mode in SUPPORTED_MODES:
        if mode not in all_metrics:
            continue
        
        m = all_metrics[mode]
        mode_score = m['success_rate'] * 100
        mode_steps = m['median_steps']
        
        score_delta = mode_score - baseline_score
        steps_delta = mode_steps - baseline_steps
        
        usefulness = compute_usefulness_label(m, baseline_metrics)
        
        score_indicator = "🟢" if score_delta >= 2 else ("🟡" if score_delta >= 0 else "🔴")
        
        lines.append(
            f"| {mode} | {m['valid_tasks']}/{m['total_tasks']} | {m['init_or_agent_errors']} | "
            f"{m['success_rate']:.1%} | {score_indicator} {score_delta:+.1f}pp | "
            f"{mode_steps:.1f} | {usefulness} |"
        )
    
    lines.extend([
        "",
        "## Detailed Metrics",
        "",
        "| Mode | Successes | Failures | Invalid Action Rate | Manager Call Rate | Cache Hit Rate |",
        "|------|-----------|----------|---------------------|-------------------|----------------|",
    ])
    
    for mode in SUPPORTED_MODES:
        if mode not in all_metrics:
            continue
        m = all_metrics[mode]
        lines.append(
            f"| {mode} | {m['successes']} | {m['failures']} | {m['invalid_action_rate']:.1%} | "
            f"{m['manager_call_rate']:.1%} | {m['cache_hit_rate']:.1%} |"
        )
    
    lines.extend([
        "",
        "## Usefulness Labels",
        "",
        "- **Useful**: Score improves >= +2 percentage points vs baseline",
        "- **Useful-for-speed**: Score within ±1pp AND median steps decreases >= 10%",
        "- **Not useful**: Does not meet above criteria",
        "",
        "## Configuration",
        "",
        "- **Task Version:** v2 (CRITICAL: SDK defaults to v1 if omitted)",
        f"- **Max Steps:** {max_steps}",
        "- **Browser:** 1280x720",
        "- **Observations:** AXTree + Screenshot (no HTML)",
        "",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated {output_path}")


def main():
    # Check API key before doing anything
    check_api_key()
    
    parser = argparse.ArgumentParser(
        description="Run full REAL v2 benchmark matrix evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full matrix with all modes
    python eval_full_matrix.py
    
    # Run specific modes only
    python eval_full_matrix.py --mode baseline_worker hierarchy_vac_macros
    
    # Run specific tasks
    python eval_full_matrix.py --tasks v2.omnizon-13,v2.gomail-1
    
    # Run with debug output
    python eval_full_matrix.py --debug --max_steps 10
"""
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of task IDs (default: all discovered v2 tasks)"
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
        "--mode",
        type=str,
        nargs='+',
        default=SUPPORTED_MODES,
        choices=SUPPORTED_MODES,
        help="Mode(s) to run (default: all modes)"
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiments YAML config file"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        nargs='+',
        default=None,
        help="Run specific experiment(s) from config by name"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Task subset from config (e.g., quick_test, shopping)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        import random
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    if args.task_version != "v2":
        print(f"WARNING: task_version={args.task_version} specified. SDK defaults to v1 if omitted!")
    
    headless = args.headless.lower() in ('true', '1', 'yes')
    
    run_id = args.run_id or f"matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config if provided
    config = None
    experiments_to_run = None
    if args.config:
        config = load_experiments_config(args.config)
        experiments_to_run = get_experiments_from_config(config, args.experiment)
        print(f"Loaded {len(experiments_to_run)} experiments from {args.config}")
    
    # Determine tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    elif config and args.subset:
        tasks = get_tasks_from_config(config, args.subset)
        if tasks:
            print(f"Using {len(tasks)} tasks from subset '{args.subset}'")
        else:
            all_tasks, source = discover_v2_tasks()
            tasks = all_tasks
            print(f"Subset '{args.subset}' returned all {len(tasks)} discovered tasks")
    else:
        all_tasks, source = discover_v2_tasks()
        if not all_tasks:
            print("ERROR: No v2 tasks discovered!")
            sys.exit(1)
        
        print(f"Discovered {len(all_tasks)} v2 tasks from {source}")
        
        if args.task_type:
            all_tasks = [t for t in all_tasks if args.task_type in t]
            print(f"Filtered to {len(all_tasks)} tasks matching '{args.task_type}'")
        
        tasks = all_tasks
    
    # Determine modes to run
    if experiments_to_run:
        modes_to_run = [exp['mode'] for exp in experiments_to_run]
    else:
        modes_to_run = args.mode
    
    print(f"\n{'='*70}")
    print(f"HALO-Agent Full Benchmark Matrix")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Tasks: {len(tasks)}")
    print(f"Modes: {modes_to_run}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Headless: {headless}")
    print(f"Results: {results_dir}")
    if args.config:
        print(f"Config: {args.config}")
    print()
    
    if args.dry_run:
        print("DRY RUN - Would execute:")
        for mode in modes_to_run:
            print(f"  Mode: {mode}")
        print(f"\nTasks ({len(tasks)}):")
        for task in tasks[:10]:
            print(f"  - {task}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return
    
    all_results = {}
    all_metrics = {}
    
    for mode in modes_to_run:
        try:
            exp_cfg = None
            if experiments_to_run:
                for exp in experiments_to_run:
                    if exp.get('mode') == mode:
                        exp_cfg = exp
                        break

            agent_overrides = None
            run_name = None
            if exp_cfg:
                run_name = exp_cfg.get('name') or mode
                agent_overrides = {
                    k: exp_cfg[k]
                    for k in [
                        'worker_model',
                        'manager_model',
                        'use_manager',
                        'use_cache',
                        'use_macros',
                        'manager_warm_start',
                        'enable_recovery_policies',
                        'always_call_manager',
                        'qwen_backend',
                        'qwen_base_url',
                    ]
                    if k in exp_cfg
                }

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
            key = run_name or mode
            all_results[key] = results
            all_metrics[key] = metrics
            
            save_results(run_id, key, results, metrics, results_dir)
            
            print(f"\n{key} Summary:")
            print(f"  Valid Tasks: {metrics['valid_tasks']}/{metrics['total_tasks']}")
            print(f"  Errors: {metrics['init_or_agent_errors']}")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            
        except Exception as e:
            print(f"\nFATAL Error running mode {mode}: {e}")
            traceback.print_exc()
    
    if all_metrics:
        csv_path = results_dir / "matrix.csv"
        md_path = results_dir / "matrix.md"
        
        generate_matrix_csv(all_metrics, csv_path, run_id)
        generate_matrix_md(all_metrics, md_path, run_id, args.max_steps)
        
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        
        baseline_metrics = all_metrics.get('baseline_worker', {})
        baseline_score = baseline_metrics.get('success_rate', 0) * 100
        
        for mode in modes_to_run:
            if mode not in all_metrics:
                continue
            m = all_metrics[mode]
            mode_score = m['success_rate'] * 100
            delta = mode_score - baseline_score
            usefulness = compute_usefulness_label(m, baseline_metrics)
            
            print(f"\n{mode}:")
            print(f"  Score: {mode_score:.1f}% ({delta:+.1f}pp vs baseline)")
            print(f"  Usefulness: {usefulness}")
    
    print(f"\n{'='*70}")
    print(f"Matrix evaluation complete!")
    print(f"Results: {results_dir / 'matrix.csv'}")
    print(f"Report: {results_dir / 'matrix.md'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
