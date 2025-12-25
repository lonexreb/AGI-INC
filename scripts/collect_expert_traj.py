#!/usr/bin/env python3
"""Collect expert trajectories for RL training (BC/DPO).

This script uses the expert_rollout mode with:
- gpt-4o worker (stronger model)
- Manager + VAC + Macros enabled
- Higher max_steps (100-140) for data collection

The trajectories are saved to data/trajectories/ for processing into BC/DPO datasets.

IMPORTANT: This is NOT an ablation experiment - purely for data collection.

Usage:
    python scripts/collect_expert_traj.py --tasks v2.gomail-1,v2.gomail-2 --max_steps 120
    python scripts/collect_expert_traj.py --domain gomail --num_tasks 5
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Expert rollout settings
EXPERT_MODE = "expert_rollout"
EXPERT_MAX_STEPS = 120  # Higher for data collection
EXPERT_WORKER_MODEL = "gpt-4o"  # Stronger model


def discover_v2_tasks():
    """Discover available v2 tasks from the SDK."""
    try:
        from agisdk.REAL.browsergym.webarena import ALL_WEBARENA_TASK_IDS
        v2_tasks = sorted([t for t in ALL_WEBARENA_TASK_IDS if t.startswith("v2.")])
        return v2_tasks
    except Exception as e:
        print(f"Warning: Could not discover tasks: {e}")
        return []


def run_expert_rollout(task_id: str, max_steps: int, headless: bool, output_dir: Path):
    """Run a single expert rollout and save trajectory."""
    from halo.sdk.harness import ensure_v2_task, HaloAgentArgs, create_harness
    from halo.logging import TrajectoryLogger
    
    task_id = ensure_v2_task(task_id)
    run_id = f"expert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n{'='*60}")
    print(f"Expert Rollout: {task_id}")
    print(f"Mode: {EXPERT_MODE} (gpt-4o worker)")
    print(f"Max Steps: {max_steps}")
    print(f"{'='*60}\n")
    
    try:
        # Setup trajectory logger
        traj_logger = TrajectoryLogger(
            run_id=run_id,
            output_dir=str(output_dir),
            mode=EXPERT_MODE
        )
        
        # Create agent args
        agent_args = HaloAgentArgs(
            mode=EXPERT_MODE,
            max_steps=max_steps,
            run_id=run_id,
            log_trajectories=True
        )
        
        # Create and run harness
        harness = create_harness(
            agent_args=agent_args,
            task_name=task_id,
            headless=headless,
            max_steps=max_steps,
            run_id=run_id,
            mode=EXPERT_MODE
        )
        
        start_time = datetime.now()
        results = harness.run()
        if isinstance(results, dict):
            if task_id in results:
                result = results[task_id]
            elif len(results) == 1:
                result = next(iter(results.values()))
            else:
                result = results
        else:
            result = results

        if not isinstance(result, dict):
            result = {"raw_result": result}
        end_time = datetime.now()
        wall_time = (end_time - start_time).total_seconds()
        
        # Extract results
        reward = result.get("cum_reward", 0)
        success = reward > 0
        steps = result.get("n_steps", 0)
        
        print(f"\n{'='*60}")
        print(f"Result: {'✅ SUCCESS' if success else '❌ Failed'}")
        print(f"Reward: {reward}, Steps: {steps}, Time: {wall_time:.1f}s")
        print(f"{'='*60}\n")
        
        # Save result metadata
        result_file = output_dir / f"{task_id.replace('.', '_')}_result.json"
        with open(result_file, 'w') as f:
            json.dump({
                "task_id": task_id,
                "mode": EXPERT_MODE,
                "worker_model": EXPERT_WORKER_MODEL,
                "success": success,
                "reward": reward,
                "steps": steps,
                "wall_time": wall_time,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        
        return success, steps, wall_time, reward
        
    except Exception as e:
        print(f"Error running {task_id}: {e}")
        traceback.print_exc()
        return False, 0, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Collect expert trajectories for RL")
    parser.add_argument("--tasks", type=str, default=None,
                       help="Comma-separated task IDs (default: auto-discover)")
    parser.add_argument("--max_steps", type=int, default=EXPERT_MAX_STEPS,
                       help=f"Max steps per episode (default: {EXPERT_MAX_STEPS})")
    parser.add_argument("--headless", type=str, default="true",
                       help="Run headless (default: true)")
    parser.add_argument("--num_tasks", type=int, default=10,
                       help="Number of tasks to run (default: 10)")
    parser.add_argument("--domain", type=str, default=None,
                       help="Filter by domain (gomail, gcalendar, omnizon)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "trajectories" / f"expert_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headless = args.headless.lower() in ('true', '1', 'yes')
    
    # Get tasks
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        tasks = discover_v2_tasks()
        if args.domain:
            tasks = [t for t in tasks if args.domain in t]
        tasks = tasks[:args.num_tasks]
    
    if not tasks:
        print("ERROR: No tasks found to run")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("EXPERT TRAJECTORY COLLECTION")
    print(f"{'='*60}")
    print(f"Mode: {EXPERT_MODE}")
    print(f"Worker Model: {EXPERT_WORKER_MODEL}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Track results
    successes = []
    failures = []
    
    for i, task_id in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Running {task_id}...")
        
        success, steps, wall_time, reward = run_expert_rollout(
            task_id=task_id,
            max_steps=args.max_steps,
            headless=headless,
            output_dir=output_dir
        )
        
        if success:
            successes.append((task_id, steps, wall_time, reward))
            print(f"✅ SUCCESS: {task_id} (reward={reward}, {steps} steps, {wall_time:.1f}s)")
        else:
            failures.append(task_id)
            print(f"❌ FAILED: {task_id}")
    
    # Summary
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(tasks)}")
    print(f"Successes: {len(successes)} ({100*len(successes)/len(tasks):.1f}%)")
    print(f"Failures: {len(failures)}")
    
    if successes:
        print(f"\nSuccessful trajectories:")
        for task_id, steps, wall_time, reward in successes:
            print(f"  - {task_id}: reward={reward}, {steps} steps, {wall_time:.1f}s")
        print(f"\nTrajectories saved to: {output_dir}")
        print("\nNext step: Process into BC/DPO datasets:")
        print(f"  python scripts/collect_traj.py --input_dir {output_dir} --output_dir data/datasets/")
    else:
        print("\n⚠️  No successful trajectories collected.")
        print("Consider:")
        print("  - Trying different tasks (email/calendar often easier)")
        print("  - Increasing max_steps")
        print("  - Running with headless=false to observe agent")
    
    # Save summary
    summary_file = output_dir / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "mode": EXPERT_MODE,
            "worker_model": EXPERT_WORKER_MODEL,
            "max_steps": args.max_steps,
            "total_tasks": len(tasks),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(tasks) if tasks else 0,
            "successful_tasks": [t[0] for t in successes],
            "failed_tasks": failures,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    return len(successes)


if __name__ == "__main__":
    main()
