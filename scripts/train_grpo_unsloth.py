#!/usr/bin/env python3
"""Group Relative Policy Optimization (GRPO) training script using Unsloth + TRL.

Trains a LoRA adapter on Qwen for browser automation with GRPO-style updates.

This script builds a prompt dataset from trajectory logs and uses a reward function that
scores generated actions based on how well they match actions observed in higher-progress
episodes.

Usage:
  python scripts/train_grpo_unsloth.py \
    --input_dir data/trajectories/qwen_worker_zero \
    --output_dir checkpoints/qwen_grpo_lora
"""

import argparse
import inspect
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GrpoPromptExample:
    prompt: str
    reward_actions: List[str]
    reward_values: List[float]
    task_id: str
    site_id: str


def extract_site_id(task_id: str) -> str:
    task = task_id.replace("v2.", "")
    parts = task.split("-")
    return parts[0] if parts else "unknown"


def load_trajectory_files(input_dir: Path) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    jsonl_files = list(input_dir.rglob("*.jsonl"))
    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        record["_source_file"] = str(jsonl_path)
                        all_records.append(record)
        except Exception as e:
            print(f"Warning: Failed to load {jsonl_path}: {e}")
    return all_records


def group_by_episode(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}

    by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_file[record.get("_source_file", "unknown")].append(record)

    for source_file, file_records in by_file.items():
        has_boundaries = any(r.get("type") in {"episode_start", "episode_end", "episode"} for r in file_records)

        if has_boundaries:
            current = {"steps": [], "episode": None, "episode_start": None}
            episode_idx = 0
            prev_step_idx = None

            def finalize_current() -> None:
                nonlocal episode_idx, current, prev_step_idx
                if not (current["steps"] or current["episode"] or current["episode_start"]):
                    return
                episodes[f"{source_file}::episode_{episode_idx}"] = current
                episode_idx += 1
                current = {"steps": [], "episode": None, "episode_start": None}
                prev_step_idx = None

            for record in file_records:
                record_type = record.get("type", "step")
                if record_type == "episode_start":
                    if current["steps"] or current["episode"] or current["episode_start"]:
                        finalize_current()
                    current["episode_start"] = record
                    continue

                if record_type == "step":
                    step_idx = record.get("step_idx")
                    if isinstance(step_idx, int) and isinstance(prev_step_idx, int) and step_idx <= prev_step_idx:
                        finalize_current()
                    current["steps"].append(record)
                    if isinstance(step_idx, int):
                        prev_step_idx = step_idx
                    continue

                if record_type in {"episode", "episode_end"}:
                    current["episode"] = record
                    finalize_current()
                    continue

            finalize_current()
            continue

        current_steps: List[Dict[str, Any]] = []
        episode_idx = 0
        prev_step_idx = None

        for record in file_records:
            record_type = record.get("type", "step")
            if record_type != "step":
                continue
            step_idx = record.get("step_idx")
            if isinstance(step_idx, int) and isinstance(prev_step_idx, int) and step_idx <= prev_step_idx:
                if current_steps:
                    episodes[f"{source_file}::episode_{episode_idx}"] = {
                        "steps": current_steps,
                        "episode": None,
                        "episode_start": None,
                    }
                    episode_idx += 1
                current_steps = []
            current_steps.append(record)
            if isinstance(step_idx, int):
                prev_step_idx = step_idx

        if current_steps:
            episodes[f"{source_file}::episode_{episode_idx}"] = {
                "steps": current_steps,
                "episode": None,
                "episode_start": None,
            }

    for ep_key in episodes:
        episodes[ep_key]["steps"].sort(key=lambda x: x.get("step_idx", 0))

    return episodes


def build_prompt_from_step(step: Dict[str, Any]) -> str:
    obs_summary = step.get("obs_summary")
    if isinstance(obs_summary, str) and obs_summary.strip():
        return obs_summary

    parts: List[str] = []

    url = step.get("url")
    if isinstance(url, str) and url:
        parts.append(f"# Current URL\n{url}")

    last_action_error = step.get("last_action_error")
    if isinstance(last_action_error, str) and last_action_error:
        parts.append(f"# Last Action Error\n{last_action_error}")

    obs_hash = step.get("obs_hash")
    if isinstance(obs_hash, str) and obs_hash:
        parts.append(f"# State Hash: {obs_hash}")

    if parts:
        return "\n\n".join(parts)

    return f"Step {step.get('step_idx', 0)}"


def _episode_task_id(ep_data: Dict[str, Any]) -> str:
    episode_info = ep_data.get("episode") or {}
    episode_start = ep_data.get("episode_start") or {}
    steps = ep_data.get("steps", [])

    task_id = episode_info.get("task_id", "")
    if not task_id:
        task_id = episode_start.get("task_id", "")
    if not task_id and steps:
        task_id = steps[0].get("task_id", "")
    return task_id


def _episode_max_progress_score(ep_data: Dict[str, Any]) -> float:
    episode_info = ep_data.get("episode") or {}
    max_progress = episode_info.get("max_progress_score", None)
    if isinstance(max_progress, (int, float)):
        return float(max_progress)

    steps = ep_data.get("steps", [])
    best = 0.0
    for step in steps:
        score = step.get("progress_score", 0.0)
        if isinstance(score, (int, float)):
            best = max(best, float(score))
    return best


def _episode_total_steps(ep_data: Dict[str, Any]) -> int:
    episode_info = ep_data.get("episode") or {}
    total = episode_info.get("total_steps", None)
    if isinstance(total, int):
        return int(total)
    return int(len(ep_data.get("steps", [])))


def _episode_recovery_action_count(ep_data: Dict[str, Any]) -> int:
    count = 0
    for step in ep_data.get("steps", []):
        src = step.get("action_source")
        if isinstance(src, str) and src.startswith("recovery_"):
            count += 1
    return count


def _rank_episodes_by_task(episodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep_key, ep_data in episodes.items():
        task_id = _episode_task_id(ep_data)
        if not task_id:
            continue
        grouped[task_id].append(
            {
                "ep_key": ep_key,
                "ep_data": ep_data,
                "max_progress_score": _episode_max_progress_score(ep_data),
                "total_steps": _episode_total_steps(ep_data),
                "recovery_action_count": _episode_recovery_action_count(ep_data),
            }
        )

    for task_id, items in grouped.items():
        items.sort(
            key=lambda x: (
                -float(x["max_progress_score"]),
                int(x["total_steps"]),
                int(x["recovery_action_count"]),
                str(x["ep_key"]),
            )
        )

    return grouped


def build_qwen_prompt(obs_summary: str) -> str:
    action_grammar = """Available actions (use exact syntax):
- click(\"bid\") - Click element with browser ID
- fill(\"bid\", \"text\") - Fill text input with value
- select_option(\"bid\", \"option\") - Select dropdown option
- scroll(x, y) - Scroll page by x,y pixels
- go_back() - Navigate back
- go_forward() - Navigate forward
- goto(\"url\") - Navigate to URL
- send_msg_to_user(\"message\") - Send message to complete task
- noop() - Do nothing this step
"""

    system_prompt = (
        "You are a browser automation agent. Your goal is to complete the given task.\n\n"
        + action_grammar
        + "\nCRITICAL RULES:\n"
        + "1. You MUST ONLY use element IDs (bid values) from the Actionable Elements list if present.\n"
        + "2. NEVER invent element IDs.\n"
        + "3. Do NOT call send_msg_to_user unless you see clear confirmation.\n\n"
        + "Output ONLY valid JSON:\n"
        + '{"action": "your_action_here", "rationale": "brief explanation", "confidence": 0.0-1.0}'
        + "\n"
    )

    return (
        "<|im_start|>system\n"
        + system_prompt
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + (obs_summary or "")
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def build_grpo_dataset_from_trajectories(
    input_dir: str,
    top_percent: float = 0.2,
    min_action_support: int = 1,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[GrpoPromptExample]:
    if not (0.0 < float(top_percent) <= 1.0):
        raise ValueError(f"top_percent must be in (0, 1], got {top_percent}")

    records = load_trajectory_files(Path(input_dir))
    episodes = group_by_episode(records)
    ranked = _rank_episodes_by_task(episodes)

    rng = random.Random(int(seed))

    examples: List[GrpoPromptExample] = []

    for task_id, items in ranked.items():
        if not items:
            continue

        k = int(math.ceil(len(items) * float(top_percent)))
        k = max(1, min(k, len(items)))
        chosen_items = items[:k]

        state_to_obs: Dict[str, str] = {}
        state_to_action_rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
        state_to_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for item in chosen_items:
            ep_data = item["ep_data"]
            ep_reward = float(item["max_progress_score"])

            for step in ep_data.get("steps", []):
                if step.get("last_action_error"):
                    continue

                obs_hash = step.get("obs_hash")
                action = step.get("action")

                if not isinstance(obs_hash, str) or not obs_hash:
                    continue
                if not isinstance(action, str) or not action:
                    continue

                if obs_hash not in state_to_obs:
                    state_to_obs[obs_hash] = build_prompt_from_step(step)

                state_to_action_counts[obs_hash][action] += 1
                prev = state_to_action_rewards[obs_hash].get(action)
                if prev is None:
                    state_to_action_rewards[obs_hash][action] = ep_reward
                else:
                    state_to_action_rewards[obs_hash][action] = max(prev, ep_reward)

        site_id = extract_site_id(task_id)
        state_keys = list(state_to_action_rewards.keys())
        rng.shuffle(state_keys)

        for obs_hash in state_keys:
            action_rewards = state_to_action_rewards.get(obs_hash, {})
            if not action_rewards:
                continue

            filtered: Dict[str, float] = {}
            for action, rew in action_rewards.items():
                if int(state_to_action_counts[obs_hash].get(action, 0)) >= int(min_action_support):
                    filtered[action] = float(rew)
            if not filtered:
                continue

            sorted_pairs = sorted(filtered.items(), key=lambda x: (-float(x[1]), str(x[0])))
            reward_actions = [a for a, _ in sorted_pairs]
            reward_values = [float(r) for _, r in sorted_pairs]

            obs_summary = state_to_obs.get(obs_hash, f"# State Hash: {obs_hash}")
            prompt = build_qwen_prompt(obs_summary)
            examples.append(
                GrpoPromptExample(
                    prompt=prompt,
                    reward_actions=reward_actions,
                    reward_values=reward_values,
                    task_id=task_id,
                    site_id=site_id,
                )
            )

            if max_examples is not None and len(examples) >= int(max_examples):
                return examples

    return examples


def _extract_action_from_completion(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    try:
        match = None
        for m in [
            *(__import__("re").finditer(r"\{[^}]+\}", text)),
        ]:
            match = m
            break
        if match is not None:
            obj = json.loads(match.group())
            action = obj.get("action")
            return action.strip() if isinstance(action, str) else ""
    except Exception:
        pass

    try:
        import re

        m = re.search(
            r"(click|fill|select_option|scroll|go_back|go_forward|goto|send_msg_to_user|noop)\([^)]*\)",
            text,
        )
        return m.group(0).strip() if m else ""
    except Exception:
        return ""


def make_reward_func():
    def reward_func(
        prompts: List[str],
        completions: List[str],
        completion_ids: List[List[int]],
        reward_actions: List[List[str]],
        reward_values: List[List[float]],
        trainer_state=None,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []
        for comp, actions, values in zip(completions, reward_actions, reward_values, strict=True):
            action = _extract_action_from_completion(comp)
            if not action:
                rewards.append(-1.0)
                continue

            ar: Dict[str, float] = {}
            if isinstance(actions, list) and isinstance(values, list):
                for a, v in zip(actions, values):
                    if isinstance(a, str) and isinstance(v, (int, float)):
                        ar[a] = float(v)

            rewards.append(float(ar.get(action, 0.0)))
        return rewards

    return reward_func


def train_grpo(
    base_model: str,
    input_dir: str,
    output_dir: str,
    top_percent: float = 0.2,
    max_examples: Optional[int] = None,
    max_seq_len: int = 2048,
    max_completion_length: int = 128,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    epochs: int = 1,
    num_generations: int = 8,
    beta: float = 0.001,
    temperature: float = 1.0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    warmup_ratio: float = 0.03,
    save_steps: int = 100,
    logging_steps: int = 10,
    seed: int = 42,
):
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOTrainer, GRPOConfig
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: Missing required packages: {e}")
        print("Install with: pip install unsloth trl datasets")
        sys.exit(1)

    print("=" * 60)
    print("HALO-Agent GRPO Training with Unsloth + TRL")
    print("=" * 60)
    print(f"Base Model: {base_model}")
    print(f"Trajectories: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"top_percent: {top_percent}")
    print(f"num_generations: {num_generations}")
    print(f"beta (KL): {beta}")
    print(f"max_completion_length: {max_completion_length}")
    print()

    print("Building GRPO dataset from trajectories...")
    raw = build_grpo_dataset_from_trajectories(
        input_dir=input_dir,
        top_percent=top_percent,
        max_examples=max_examples,
        seed=seed,
    )
    if not raw:
        print("ERROR: No training examples produced from trajectories.")
        sys.exit(1)

    dataset = Dataset.from_list(
        [
            {
                "prompt": ex.prompt,
                "reward_actions": ex.reward_actions,
                "reward_values": ex.reward_values,
            }
            for ex in raw
        ]
    )
    print(f"Dataset ready: {len(dataset)} prompts")

    print("\nLoading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(seed),
    )

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        optim="adamw_8bit",
        seed=int(seed),
        report_to="none",
        remove_unused_columns=False,
        num_generations=int(num_generations),
        max_completion_length=int(max_completion_length),
        beta=float(beta),
        temperature=float(temperature),
    )

    reward_func = make_reward_func()

    print("\nStarting GRPO training...")
    trainer_init = inspect.signature(GRPOTrainer.__init__)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": grpo_config,
        "train_dataset": dataset,
    }
    if "reward_funcs" in trainer_init.parameters:
        trainer_kwargs["reward_funcs"] = reward_func
    elif "reward_func" in trainer_init.parameters:
        trainer_kwargs["reward_func"] = reward_func
    else:
        raise TypeError("GRPOTrainer missing reward function parameter")

    if "processing_class" in trainer_init.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    if "ref_model" in trainer_init.parameters:
        trainer_kwargs["ref_model"] = None

    trainer = GRPOTrainer(**trainer_kwargs)

    trainer.train()

    print(f"\nSaving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("=" * 60)
    print("GRPO Training Complete!")
    print(f"Adapter saved to: {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO model using Unsloth + TRL")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/trajectories/qwen_worker_zero",
        help="Trajectory directory to build prompts from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/qwen_grpo_lora",
        help="Output directory for adapter",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=0.2,
        help="Top percent episodes per task to keep (progress-ranked)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of prompt examples",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"ERROR: input_dir not found: {args.input_dir}")
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_grpo(
        base_model=args.base_model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        top_percent=args.top_percent,
        max_examples=args.max_examples,
        max_seq_len=args.max_seq_len,
        max_completion_length=args.max_completion_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        epochs=args.epochs,
        num_generations=args.num_generations,
        beta=args.beta,
        temperature=args.temperature,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
