#!/usr/bin/env python3
"""
Single-task debug runner for HALO-Agent.

Runs one task with detailed step-by-step output for debugging.

Usage:
    python scripts/run_one_debug.py --task v2.omnizon-13
    python scripts/run_one_debug.py --task v2.gomail-1 --mode hierarchy_vac_macros
    python scripts/run_one_debug.py --task v2.omnizon-13 --headless false --max_steps 10
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import os
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

SUPPORTED_MODES = [
    'baseline_worker',
    'hierarchy_mgr_gate',
    'hierarchy_vac',
    'hierarchy_vac_macros',
    'qwen_worker_zero',
    'qwen_worker_bc',
    'qwen_worker_dpo',
    'qwen_worker_grpo',
]


class DebugStepLogger:
    """Logger that prints detailed step information during execution."""
    
    def __init__(self):
        self.steps = []
        self.current_step = 0
    
    def log_step(
        self,
        step_index: int,
        url: str,
        action: str,
        action_source: str,
        last_action_error: Optional[str],
        page_type: str,
        cache_hit: bool,
        manager_called: bool
    ):
        """Log a single step with all debug information."""
        step_info = {
            "step": step_index,
            "url": url,
            "action": action,
            "action_source": action_source,
            "last_action_error": last_action_error,
            "page_type": page_type,
            "cache_hit": cache_hit,
            "manager_called": manager_called
        }
        self.steps.append(step_info)
        
        print(f"\n{'─'*60}")
        print(f"STEP {step_index}")
        print(f"{'─'*60}")
        print(f"  URL: {url[:80]}{'...' if len(url) > 80 else ''}")
        print(f"  Page Type: {page_type}")
        print(f"  Action: {action}")
        print(f"  Source: {action_source}")
        print(f"  Cache Hit: {'✓' if cache_hit else '✗'}")
        print(f"  Manager Called: {'✓' if manager_called else '✗'}")
        if last_action_error:
            print(f"  Last Error: {last_action_error}")


def dump_observation(obs: Dict, step: int, output_dir: Path, obs_summary: str = ""):
    """Dump observation to debug files.
    
    Writes:
    - debug_obs_step_<t>.json: Raw obs dict (excluding large screenshot bytes)
    - debug_summary_step_<t>.txt: Summarized prompt
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a copy without screenshot bytes (too large)
    obs_copy = {}
    for k, v in obs.items():
        if k == 'screenshot' and isinstance(v, bytes):
            obs_copy[k] = f"<{len(v)} bytes>"
        elif k == 'screenshot' and isinstance(v, str) and len(v) > 1000:
            obs_copy[k] = f"<base64 string, {len(v)} chars>"
        else:
            try:
                json.dumps(v)  # Test if serializable
                obs_copy[k] = v
            except (TypeError, ValueError):
                obs_copy[k] = str(v)[:500]
    
    # Write raw obs
    obs_file = output_dir / f"debug_obs_step_{step}.json"
    with open(obs_file, 'w') as f:
        json.dump(obs_copy, f, indent=2, default=str)
    
    # Write summary
    if obs_summary:
        summary_file = output_dir / f"debug_summary_step_{step}.txt"
        with open(summary_file, 'w') as f:
            f.write(obs_summary)
    
    return obs_file


def run_debug_task(
    task_name: str,
    mode: str,
    headless: bool,
    max_steps: int,
    dry_run: bool = False,
    dump_obs: bool = False
) -> Dict[str, Any]:
    """Run a single task with debug output.
    
    Args:
        task_name: Task ID (v2. prefix added if missing)
        mode: Agent mode
        headless: Run headless browser
        max_steps: Maximum steps
        dry_run: If True, don't make LLM calls
        dump_obs: If True, dump observations to debug files
        
    Returns:
        Result dictionary
    """
    from halo.sdk.harness import ensure_v2_task, HaloAgentArgs, create_harness
    from halo.sdk.agent import HaloAgent
    from halo.logging import TrajectoryLogger
    
    task_name = ensure_v2_task(task_name)
    
    print(f"\n{'='*60}")
    print(f"DEBUG RUN: {task_name}")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Max Steps: {max_steps}")
    print(f"Headless: {headless}")
    print(f"Dry Run: {dry_run}")
    print()
    
    if dry_run:
        print("DRY RUN MODE - No LLM calls will be made")
        print("Simulating agent initialization...")
        
        try:
            agent = HaloAgent(mode=mode, max_steps=max_steps)
            print(f"  ✓ Agent created successfully")
            print(f"  ✓ Mode: {agent.mode}")
            print(f"  ✓ Orchestrator initialized")
            
            if hasattr(agent.orchestrator, 'manager') and agent.orchestrator.manager:
                print(f"  ✓ Manager policy: enabled")
            else:
                print(f"  ✗ Manager policy: disabled")
            
            if hasattr(agent.orchestrator, 'vac') and agent.orchestrator.vac:
                print(f"  ✓ VAC: enabled")
            else:
                print(f"  ✗ VAC: disabled")
            
            if hasattr(agent.orchestrator, 'macro_cache') and agent.orchestrator.macro_cache:
                print(f"  ✓ Macro cache: enabled")
            else:
                print(f"  ✗ Macro cache: disabled")
            
            print(f"\n✓ Dry run completed successfully")
            return {"success": True, "dry_run": True, "mode": mode}
            
        except Exception as e:
            print(f"\n✗ Dry run failed: {e}")
            traceback.print_exc()
            return {"success": False, "dry_run": True, "error": str(e)}
    
    run_id = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    debug_logger = DebugStepLogger()
    
    # Setup output directory for dumps
    repo_root = Path(__file__).parent.parent
    debug_output_dir = repo_root / "results" / run_id
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    
    if dump_obs:
        print(f"Observation dumps will be saved to: {debug_output_dir}")
    
    try:
        traj_logger = TrajectoryLogger(
            run_id=run_id,
            output_dir=f"data/trajectories/debug",
            mode=mode
        )
        
        agent_args = HaloAgentArgs(
            mode=mode,
            max_steps=max_steps,
            run_id=run_id,
            log_trajectories=True
        )
        
        print(f"Creating harness for task: {task_name}")
        
        harness = create_harness(
            agent_args=agent_args,
            task_name=task_name,
            headless=headless,
            max_steps=max_steps,
            run_id=run_id,
            mode=mode
        )
        
        print(f"Starting task execution...")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        results = harness.run()
        if isinstance(results, dict):
            if task_name in results:
                result = results[task_name]
            elif len(results) == 1:
                result = next(iter(results.values()))
            else:
                result = results
        else:
            result = results

        if not isinstance(result, dict):
            result = {"raw_result": result}

        result["task_name"] = task_name
        result["task_id"] = task_name.split('.', 1)[1] if '.' in task_name else task_name
        end_time = datetime.now()
        
        wall_time = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"EXECUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Success: {'✓' if result.get('success') else '✗'}")
        print(f"Steps: {result.get('n_steps', 0)}")
        print(f"Wall Time: {wall_time:.1f}s")
        print(f"Reward: {result.get('cum_reward', 0)}")
        
        if result.get('error'):
            print(f"Error: {result.get('error')}")
        
        result['wall_time'] = wall_time
        result['debug_steps'] = debug_logger.steps
        
        return result
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"EXECUTION FAILED")
        print(f"{'='*60}")
        print(f"Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "debug_steps": debug_logger.steps
        }


def main():
    # Check API key before doing anything
    check_api_key()
    
    parser = argparse.ArgumentParser(
        description="Run a single HALO-Agent task with debug output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run a task with visible browser
    python run_one_debug.py --task v2.omnizon-13
    
    # Run with specific mode
    python run_one_debug.py --task v2.gomail-1 --mode hierarchy_vac_macros
    
    # Run headless with limited steps
    python run_one_debug.py --task v2.omnizon-13 --headless true --max_steps 10
    
    # Dry run (no LLM calls, just verify setup)
    python run_one_debug.py --task v2.omnizon-13 --dry-run
"""
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task ID to run (e.g., v2.omnizon-13 or omnizon-13)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline_worker",
        choices=SUPPORTED_MODES,
        help="Agent mode (default: baseline_worker)"
    )
    parser.add_argument(
        "--headless",
        type=str,
        default="false",
        help="Run headless browser (default: false for debugging)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Max steps (default: 20 for debugging)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - verify setup without LLM calls"
    )
    parser.add_argument(
        "--dump_obs",
        action="store_true",
        help="Dump observations to results/<run_id>/debug_obs_step_<t>.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    headless = args.headless.lower() in ('true', '1', 'yes')
    
    result = run_debug_task(
        task_name=args.task,
        mode=args.mode,
        headless=headless,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        dump_obs=args.dump_obs
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    
    sys.exit(0 if result.get('success') else 1)


if __name__ == "__main__":
    main()
