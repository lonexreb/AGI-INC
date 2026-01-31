#!/usr/bin/env python3
"""Tinker-based Online RL Training for HALO Agent.

Uses the Tinker API for distributed training of Qwen3-VL-30B-A3B-Instruct
on REAL benchmark browser automation tasks.

Prerequisites:
    1. Sign up at https://auth.thinkingmachines.ai/sign-up
    2. Set TINKER_API_KEY environment variable
    3. pip install tinker tinker-cookbook
    4. pip install agisdk==0.3.5 && playwright install --force

Usage:
    # Train on a single task
    python scripts/train_tinker.py --tasks v2.omnizon-1

    # Train on multiple tasks with custom hyperparameters
    python scripts/train_tinker.py \\
        --tasks v2.omnizon-1 v2.gomail-1 v2.gocalendar-4 \\
        --num-envs 8 \\
        --learning-rate 1e-6 \\
        --loss-fn importance_sampling

    # Use task registry file
    python scripts/train_tinker.py \\
        --task-file configs/real_v2_task_registry.json

    # Resume from checkpoint
    python scripts/train_tinker.py \\
        --tasks v2.omnizon-1 \\
        --load-checkpoint tinker://path/to/checkpoint
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HALO Agent using Tinker RL API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Model name on Tinker (default: Qwen/Qwen3-VL-30B-A3B-Instruct)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
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
        help="JSON file with task list (alternative to --tasks)",
    )
    parser.add_argument(
        "--test-tasks",
        nargs="+",
        default=None,
        help="Task names for evaluation (optional)",
    )

    # GRPO / Training
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Environments per group = GRPO group size (default: 8)",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="importance_sampling",
        choices=["importance_sampling", "ppo", "cispo", "dro"],
        help="RL loss function (default: importance_sampling)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--num-substeps",
        type=int,
        default=1,
        help="Optimizer updates per sampling iteration (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Max tokens per model response (default: 300)",
    )

    # Reward shaping
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--novelty-bonus", type=float, default=0.2)
    parser.add_argument("--loop-penalty", type=float, default=-0.5)
    parser.add_argument("--action-error-penalty", type=float, default=-0.2)
    parser.add_argument("--success-bonus", type=float, default=1.0)

    # Environment
    parser.add_argument(
        "--max-steps",
        type=int,
        default=70,
        help="Max steps per episode (default: 70 for score-mode)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser headless (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
    )

    # Tinker
    parser.add_argument(
        "--tinker-base-url",
        type=str,
        default=None,
        help="Tinker API base URL (default: uses TINKER_API_KEY env var)",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Tinker checkpoint path to resume from",
    )

    # Logging
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Directory for training logs (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=20,
        help="Run evaluation every N batches (default: 20, 0=disabled)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save checkpoint every N batches (default: 20)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser.parse_args()


def load_tasks_from_file(filepath: str) -> list[str]:
    """Load task names from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if "tasks" in data:
        tasks = data["tasks"]
        if tasks and isinstance(tasks[0], dict):
            return [t["task_name"] for t in tasks]
        return tasks
    return []


def ensure_v2(tasks: list[str]) -> list[str]:
    return [t if t.startswith("v2.") else f"v2.{t}" for t in tasks]


async def main_async(args):
    from datetime import datetime

    from tinker_cookbook.rl.train import Config, main as tinker_main

    from halo.tinker.dataset import BrowserDatasetBuilder

    # Resolve tasks
    if args.task_file:
        tasks = load_tasks_from_file(args.task_file)
    else:
        tasks = args.tasks
    tasks = ensure_v2(tasks)

    test_tasks = ensure_v2(args.test_tasks) if args.test_tasks else []

    # Log path
    if args.log_path is None:
        model_short = args.model.replace("/", "-")
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.log_path = f"results/tinker_rl/{model_short}-{ts}"

    # Build dataset builder
    dataset_builder = BrowserDatasetBuilder(
        train_tasks=tasks,
        test_tasks=test_tasks,
        model_name_for_tokenizer=args.model,
        num_envs_per_group=args.num_envs,
        max_steps=args.max_steps,
        headless=args.headless,
        seed=args.seed,
        progress_weight=args.progress_weight,
        novelty_bonus=args.novelty_bonus,
        loop_penalty=args.loop_penalty,
        action_error_penalty=args.action_error_penalty,
        success_bonus=args.success_bonus,
    )

    # Build Tinker training config
    cfg = Config(
        model_name=args.model,
        learning_rate=args.learning_rate,
        loss_fn=args.loss_fn,
        num_substeps=args.num_substeps,
        lora_rank=args.lora_rank,
        dataset_builder=dataset_builder,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        log_path=args.log_path,
        wandb_project=args.wandb_project,
        eval_every=args.eval_every,
        save_every=args.save_every,
        base_url=args.tinker_base_url,
        load_checkpoint_path=args.load_checkpoint,
    )

    # Print configuration
    print("=" * 60)
    print("HALO Agent - Tinker RL Training")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  Tasks:          {tasks}")
    print(f"  Test tasks:     {test_tasks or '(none)'}")
    print(f"  Group size:     {args.num_envs}")
    print(f"  Loss fn:        {args.loss_fn}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Max steps/ep:   {args.max_steps}")
    print(f"  Log path:       {args.log_path}")
    print("=" * 60)
    print()

    # Run Tinker training loop
    await tinker_main(cfg)


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Check API key
    if not os.environ.get("TINKER_API_KEY"):
        print("ERROR: TINKER_API_KEY environment variable not set.")
        print("Sign up at https://auth.thinkingmachines.ai/sign-up")
        print("Then: export TINKER_API_KEY=your_key_here")
        return 1

    asyncio.run(main_async(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
