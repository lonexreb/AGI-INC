#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from halo.sdk.harness import ensure_v2_task, run_single_task


def check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)


def safe_task_name(task_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name)


def temperature_tag(temp: float) -> str:
    return str(temp).replace(".", "p")


def find_episode_file(mode: str, run_id: str, task_name: str) -> Path:
    repo_root = Path(__file__).parent.parent
    run_dir = repo_root / "data" / "trajectories" / mode / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Trajectory run dir not found: {run_dir}")

    safe_task = safe_task_name(task_name)
    candidates = list(run_dir.glob(f"{safe_task}__attempt_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No trajectory files matching {safe_task}__attempt_*.jsonl in {run_dir}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_episode_start_record(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "episode_start":
                return rec
    raise ValueError(f"No episode_start record found in {path}")


def load_actions(path: Path) -> List[str]:
    actions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "step":
                action = rec.get("action")
                if isinstance(action, str):
                    actions.append(action.strip())
                else:
                    actions.append(str(action))
    return actions


def first_divergence_idx(actions_a: List[str], actions_b: List[str], max_steps: int) -> Optional[int]:
    n = min(max_steps, len(actions_a), len(actions_b))
    for i in range(n):
        if actions_a[i] != actions_b[i]:
            return i

    if max_steps > n and len(actions_a) != len(actions_b):
        return n

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same task twice with the same task_seed but different worker_temperature, "
            "then verify trajectories diverge. Exits nonzero if no divergence."
        )
    )
    parser.add_argument("--task", type=str, default="v2.gomail-1")
    parser.add_argument("--mode", type=str, default="baseline_worker")
    parser.add_argument("--task_seed", type=int, default=123)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--headless", type=str, default="true")
    parser.add_argument("--temperature_a", type=float, default=0.0)
    parser.add_argument("--temperature_b", type=float, default=0.7)
    parser.add_argument("--compare_steps", type=int, default=10)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)

    args = parser.parse_args()

    check_api_key()

    task_name = ensure_v2_task(args.task)
    headless = args.headless.lower() in ("true", "1", "yes")

    base_run_id = args.run_id or f"verify_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    repo_root = Path(__file__).parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else (repo_root / "results" / base_run_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id_a = f"{base_run_id}__t{temperature_tag(args.temperature_a)}"
    run_id_b = f"{base_run_id}__t{temperature_tag(args.temperature_b)}"

    print("=" * 70)
    print("VERIFY EXPLORATION")
    print("=" * 70)
    print(f"Task: {task_name}")
    print(f"Mode: {args.mode}")
    print(f"Task seed: {args.task_seed}")
    print(f"Temps: A={args.temperature_a}  B={args.temperature_b}")
    print(f"Compare steps: {args.compare_steps}")
    print(f"Results dir: {results_dir}")

    try:
        print(f"\n[RUN A] run_id={run_id_a}")
        run_single_task(
            task_name=task_name,
            mode=args.mode,
            headless=headless,
            max_steps=args.max_steps,
            run_id=run_id_a,
            results_dir=str(results_dir / "harness"),
            task_seed=int(args.task_seed),
            worker_temperature=float(args.temperature_a),
            force_refresh=True,
        )

        print(f"\n[RUN B] run_id={run_id_b}")
        run_single_task(
            task_name=task_name,
            mode=args.mode,
            headless=headless,
            max_steps=args.max_steps,
            run_id=run_id_b,
            results_dir=str(results_dir / "harness"),
            task_seed=int(args.task_seed),
            worker_temperature=float(args.temperature_b),
            force_refresh=True,
        )
    except Exception as e:
        print(f"ERROR: failed to run tasks: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        sys.exit(1)

    traj_a = find_episode_file(args.mode, run_id_a, task_name)
    traj_b = find_episode_file(args.mode, run_id_b, task_name)

    start_a = load_episode_start_record(traj_a)
    start_b = load_episode_start_record(traj_b)

    logged_temp_a = start_a.get("worker_temperature")
    logged_temp_b = start_b.get("worker_temperature")

    if logged_temp_a is None or logged_temp_b is None:
        print("ERROR: worker_temperature missing from episode_start record(s)")
        print(f"  traj_a={traj_a}")
        print(f"  traj_b={traj_b}")
        sys.exit(3)

    if abs(float(logged_temp_a) - float(args.temperature_a)) > 1e-6:
        print("ERROR: logged worker_temperature for run A does not match expected")
        print(f"  expected={args.temperature_a} logged={logged_temp_a}")
        sys.exit(3)

    if abs(float(logged_temp_b) - float(args.temperature_b)) > 1e-6:
        print("ERROR: logged worker_temperature for run B does not match expected")
        print(f"  expected={args.temperature_b} logged={logged_temp_b}")
        sys.exit(3)

    actions_a = load_actions(traj_a)
    actions_b = load_actions(traj_b)

    if not actions_a or not actions_b:
        print("ERROR: missing step actions in one or both trajectories")
        print(f"  traj_a={traj_a} steps={len(actions_a)}")
        print(f"  traj_b={traj_b} steps={len(actions_b)}")
        sys.exit(1)

    idx = first_divergence_idx(actions_a, actions_b, int(args.compare_steps))

    if idx is None:
        print("\nERROR: No divergence detected within compare window")
        n = min(int(args.compare_steps), len(actions_a), len(actions_b))
        for i in range(n):
            print(f"  step {i+1}: {actions_a[i]}")
        print(f"\nTraj A: {traj_a}")
        print(f"Traj B: {traj_b}")
        sys.exit(2)

    print("\nOK: Divergence detected")
    if idx < min(len(actions_a), len(actions_b)):
        print(f"  first divergent step: {idx+1}")
        print(f"  A: {actions_a[idx]}")
        print(f"  B: {actions_b[idx]}")
    else:
        print("  divergence due to episode length mismatch")
        print(f"  len(A)={len(actions_a)} len(B)={len(actions_b)}")

    print(f"\nTraj A: {traj_a}")
    print(f"Traj B: {traj_b}")
    sys.exit(0)


if __name__ == "__main__":
    main()
