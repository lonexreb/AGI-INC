#!/usr/bin/env python3
"""Online GRPO Training Script for HALO Agent.

This script trains the Qwen3-VL-30B-A3B model using Online GRPO (Group Relative
Policy Optimization) on REAL Benchmark tasks.

Prerequisites:
1. Start vLLM server:
   vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8000

2. Set up environment:
   pip install agisdk==0.3.5
   playwright install --force

Usage:
    # Dry run (no actual training)
    python scripts/train_online_grpo.py --dry-run

    # Train on single task
    python scripts/train_online_grpo.py --tasks v2.omnizon-1 --episodes 10

    # Train on multiple tasks
    python scripts/train_online_grpo.py \\
        --tasks v2.omnizon-1 v2.gomail-1 v2.calendar-1 \\
        --episodes 30 \\
        --num-generations 8 \\
        --temperature 0.7

    # Resume from checkpoint
    python scripts/train_online_grpo.py --resume checkpoints/qwen3vl_grpo_lora
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HALO Agent using Online GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Tasks
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["v2.omnizon-1"],
        help="Task names to train on (default: v2.omnizon-1)",
    )
    parser.add_argument(
        "--task-file",
        type=str,
        help="JSON file with list of tasks (alternative to --tasks)",
    )

    # Training
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to train (default: 100)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Number of actions to sample per step (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--loss-type",
        choices=["grpo", "dr_grpo"],
        default="dr_grpo",
        help="Loss type: grpo or dr_grpo (default: dr_grpo)",
    )

    # Reward shaping
    parser.add_argument(
        "--progress-weight",
        type=float,
        default=1.0,
        help="Weight for progress rewards (default: 1.0)",
    )
    parser.add_argument(
        "--novelty-bonus",
        type=float,
        default=0.2,
        help="Bonus for visiting new states (default: 0.2)",
    )
    parser.add_argument(
        "--success-bonus",
        type=float,
        default=1.0,
        help="Bonus for task completion (default: 1.0)",
    )

    # Environment
    parser.add_argument(
        "--max-steps",
        type=int,
        default=70,
        help="Maximum steps per episode (default: 70)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Run browser with visible window",
    )

    # vLLM
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Model name (default: Qwen/Qwen3-VL-30B-A3B-Instruct)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/qwen3vl_grpo_lora",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N episodes (default: 10)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint directory",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/grpo_training",
        help="Directory to save results",
    )

    # Debug
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: check config and exit without training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def setup_logging(level: str):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def check_vllm_health(url: str, model: str) -> bool:
    """Check if vLLM server is running and serving the model."""
    try:
        from halo.policy import VLLMPolicyClient
        client = VLLMPolicyClient(base_url=url, model=model)
        return client.health_check()
    except Exception as e:
        logging.error(f"vLLM health check failed: {e}")
        return False


def load_tasks_from_file(filepath: str) -> list:
    """Load task names from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("tasks", [])


def ensure_v2_task(task_name: str) -> str:
    """Ensure task name has v2. prefix."""
    if not task_name.startswith("v2."):
        return f"v2.{task_name}"
    return task_name


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Banner
    print("=" * 60)
    print("HALO Agent - Online GRPO Training")
    print("=" * 60)
    print()

    # Load tasks
    if args.task_file:
        tasks = load_tasks_from_file(args.task_file)
    else:
        tasks = args.tasks

    # Ensure v2 prefix
    tasks = [ensure_v2_task(t) for t in tasks]

    # Print configuration
    print("Configuration:")
    print(f"  Tasks: {tasks}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  vLLM URL: {args.vllm_url}")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print()

    # Dry run check
    if args.dry_run:
        print("Dry run mode - checking configuration...")

        # Check vLLM
        print("\nChecking vLLM server...")
        if check_vllm_health(args.vllm_url, args.model):
            print("  vLLM server is running and model is available")
        else:
            print("  WARNING: vLLM server not available or model not found")
            print(f"  Start with: vllm serve {args.model} --port 8000")

        # Check imports
        print("\nChecking imports...")
        try:
            from halo.rl import OnlineGRPOTrainer, GRPOConfig
            from halo.policy import VLLMPolicyClient
            from agisdk import REAL
            print("  All imports successful")
        except ImportError as e:
            print(f"  Import error: {e}")
            return 1

        print("\nDry run complete. Configuration looks good!")
        print("Remove --dry-run to start training.")
        return 0

    # Check vLLM server
    logger.info("Checking vLLM server...")
    if not check_vllm_health(args.vllm_url, args.model):
        logger.error(
            f"vLLM server not available at {args.vllm_url}\n"
            f"Start with: vllm serve {args.model} --port 8000"
        )
        return 1
    logger.info("vLLM server is ready")

    # Import training modules
    from halo.rl import OnlineGRPOTrainer, GRPOConfig
    from halo.sdk import create_harness, HaloAgentArgs

    # Create config
    config = GRPOConfig(
        num_generations=args.num_generations,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        progress_weight=args.progress_weight,
        novelty_bonus=args.novelty_bonus,
        success_bonus=args.success_bonus,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        vllm_base_url=args.vllm_url,
        model_name=args.model,
    )

    # Create trainer
    trainer = OnlineGRPOTrainer(config=config)

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save config
    config_file = results_dir / f"config_{run_id}.json"
    with open(config_file, "w") as f:
        json.dump({
            "tasks": tasks,
            "episodes": args.episodes,
            "config": {
                "num_generations": config.num_generations,
                "temperature": config.temperature,
                "learning_rate": config.learning_rate,
                "loss_type": config.loss_type,
                "progress_weight": config.progress_weight,
                "novelty_bonus": config.novelty_bonus,
                "success_bonus": config.success_bonus,
                "max_steps": config.max_steps,
            },
            "run_id": run_id,
        }, f, indent=2)

    logger.info(f"Config saved to {config_file}")

    # Create environment for each task
    # Note: REAL.harness creates a new env per task, so we iterate
    successes = 0
    failures = 0
    all_metrics = []

    for ep_idx in range(args.episodes):
        task_name = tasks[ep_idx % len(tasks)]
        task_seed = args.seed + ep_idx if args.seed else None

        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {ep_idx + 1}/{args.episodes}: {task_name}")
        logger.info(f"{'='*60}")

        try:
            # Create harness for this task
            from agisdk import REAL

            harness = REAL.harness(
                agentargs=HaloAgentArgs(
                    mode="qwen3vl_base",
                    max_steps=args.max_steps,
                    run_id=run_id,
                    qwen_backend="vllm",
                    qwen_base_url=args.vllm_url,
                ),
                task_name=task_name,
                headless=args.headless,
                max_steps=args.max_steps,
                use_axtree=True,
                use_screenshot=True,
                use_html=False,
                browser_dimensions=(1280, 720),
            )

            if task_seed is not None:
                harness.env_args["task_seed"] = task_seed

            # Run episode
            # Note: For training, we need direct env access
            # The harness runs the agent automatically, so we use it differently
            result = harness.run()

            # Parse result
            if isinstance(result, dict):
                if task_name in result:
                    record = result[task_name]
                elif len(result) == 1:
                    record = next(iter(result.values()))
                else:
                    record = result
            else:
                record = {"raw_result": result}

            success = record.get("success", False)
            if success:
                successes += 1
                logger.info(f"SUCCESS")
            else:
                failures += 1
                logger.info(f"FAILED")

            all_metrics.append({
                "episode": ep_idx + 1,
                "task": task_name,
                "success": success,
                "steps": record.get("n_steps", 0),
                "reward": record.get("cum_reward", 0),
            })

        except Exception as e:
            logger.error(f"Episode failed: {e}", exc_info=True)
            failures += 1

    # Save final results
    results_file = results_dir / f"results_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump({
            "run_id": run_id,
            "total_episodes": args.episodes,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / args.episodes if args.episodes > 0 else 0,
            "metrics": all_metrics,
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {args.episodes}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success rate: {successes / args.episodes * 100:.1f}%")
    print(f"Results saved to: {results_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
