#!/usr/bin/env python3
"""
Trajectory collection and dataset export for HALO-Agent.
Processes trajectory JSONL files into training datasets for RL.

Outputs:
- data/datasets/bc.jsonl: Behavioral cloning dataset
  Fields: {prompt, action, task_id, site_id}
  
- data/datasets/dpo.jsonl: Direct Preference Optimization dataset
  Fields: {prompt, chosen, rejected, task_id, site_id}

DPO Pairing Rules:
1. StateKey grouping: Same state_key, action with verifier_pass=true -> chosen,
   action with verifier_pass=false or last_action_error -> rejected
2. Trajectory-level fallback: If no step-level pairs, use successful episode
   actions as chosen, failed episode actions as rejected for same task_id
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BCExample:
    """Behavioral cloning training example."""
    prompt: str  # obs_summary
    action: str  # action_str
    task_id: str
    site_id: str
    step_idx: int
    action_source: str


@dataclass
class DPOExample:
    """Direct Preference Optimization training example."""
    prompt: str  # obs_summary (same for both)
    chosen: str  # action that succeeded
    rejected: str  # action that failed
    task_id: str
    site_id: str
    state_key: str  # for deduplication


def extract_site_id(task_id: str) -> str:
    """Extract site ID from task ID (e.g., 'v2.omnizon-1' -> 'omnizon')."""
    # Remove v2. prefix if present
    task = task_id.replace('v2.', '')
    # Extract site name (before the dash and number)
    parts = task.split('-')
    if parts:
        return parts[0]
    return 'unknown'


def load_trajectory_files(input_dir: Path) -> List[Dict[str, Any]]:
    """Load all trajectory JSONL files from directory.
    
    Args:
        input_dir: Directory containing .jsonl files
        
    Returns:
        List of all records from all files
    """
    all_records = []
    
    # Find all JSONL files recursively
    jsonl_files = list(input_dir.rglob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} trajectory files")
    
    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        # Add source file info
                        record['_source_file'] = str(jsonl_path)
                        all_records.append(record)
        except Exception as e:
            print(f"Warning: Failed to load {jsonl_path}: {e}")
    
    print(f"Loaded {len(all_records)} total records")
    return all_records


def group_by_episode(records: List[Dict]) -> Dict[str, Dict]:
    """Group records by episode (task_id + run).
    
    Returns:
        Dict mapping episode_key -> {steps: [...], episode: {...}}
    """
    episodes = {}

    # First, group records by the file they came from. New logging emits one file per episode.
    by_file = defaultdict(list)
    for record in records:
        by_file[record.get('_source_file', 'unknown')].append(record)

    for source_file, file_records in by_file.items():
        # If we have explicit boundaries, prefer them.
        has_boundaries = any(
            r.get('type') in {'episode_start', 'episode_end', 'episode'}
            for r in file_records
        )

        if has_boundaries:
            current = {'steps': [], 'episode': None, 'episode_start': None}
            episode_idx = 0
            prev_step_idx = None

            def finalize_current():
                nonlocal episode_idx, current, prev_step_idx
                if not (current['steps'] or current['episode'] or current['episode_start']):
                    return
                episodes[f"{source_file}::episode_{episode_idx}"] = current
                episode_idx += 1
                current = {'steps': [], 'episode': None, 'episode_start': None}
                prev_step_idx = None

            for record in file_records:
                record_type = record.get('type', 'step')
                if record_type == 'episode_start':
                    if current['steps'] or current['episode'] or current['episode_start']:
                        finalize_current()
                    current['episode_start'] = record
                    continue

                if record_type == 'step':
                    step_idx = record.get('step_idx')
                    if (
                        isinstance(step_idx, int)
                        and isinstance(prev_step_idx, int)
                        and step_idx <= prev_step_idx
                    ):
                        finalize_current()
                    current['steps'].append(record)
                    if isinstance(step_idx, int):
                        prev_step_idx = step_idx
                    continue

                if record_type in {'episode', 'episode_end'}:
                    current['episode'] = record
                    finalize_current()
                    continue

            finalize_current()
            continue

        # Fallback for legacy single-file logs without boundaries: split on step_idx resets.
        current_steps: List[Dict[str, Any]] = []
        episode_idx = 0
        prev_step_idx = None

        for record in file_records:
            record_type = record.get('type', 'step')
            if record_type != 'step':
                continue

            step_idx = record.get('step_idx')
            if (
                isinstance(step_idx, int)
                and isinstance(prev_step_idx, int)
                and step_idx <= prev_step_idx
            ):
                if current_steps:
                    episodes[f"{source_file}::episode_{episode_idx}"] = {
                        'steps': current_steps,
                        'episode': None,
                        'episode_start': None,
                    }
                    episode_idx += 1
                current_steps = []
            current_steps.append(record)
            if isinstance(step_idx, int):
                prev_step_idx = step_idx

        if current_steps:
            episodes[f"{source_file}::episode_{episode_idx}"] = {
                'steps': current_steps,
                'episode': None,
                'episode_start': None,
            }

    # Sort steps by step_idx
    for ep_key in episodes:
        episodes[ep_key]['steps'].sort(key=lambda x: x.get('step_idx', 0))

    return episodes


def build_prompt_from_step(step: Dict) -> str:
    """Build prompt string from step record.
    
    The prompt is the obs_summary that was used for the action decision.
    If not available, construct from available fields.
    """
    obs_summary = step.get('obs_summary')
    if isinstance(obs_summary, str) and obs_summary.strip():
        return obs_summary

    # If obs_hash is available, use it as a proxy for state
    parts = []
    
    if step.get('url'):
        parts.append(f"# Current URL\n{step['url']}")
    
    if step.get('last_action_error'):
        parts.append(f"# Last Action Error\n{step['last_action_error']}")
    
    # Add obs_hash as state identifier
    if step.get('obs_hash'):
        parts.append(f"# State Hash: {step['obs_hash']}")
    
    return "\n\n".join(parts) if parts else f"Step {step.get('step_idx', 0)}"


def generate_bc_dataset(
    episodes: Dict[str, Dict],
    successful_only: bool = True
) -> List[BCExample]:
    """Generate Behavioral Cloning dataset.
    
    Args:
        episodes: Grouped episode data
        successful_only: Only include steps from successful episodes
        
    Returns:
        List of BCExample instances
    """
    bc_examples = []
    
    for ep_key, ep_data in episodes.items():
        episode_info = ep_data.get('episode') or {}
        steps = ep_data.get('steps', [])
        
        # Check if episode was successful
        is_success = episode_info.get('success', False)
        
        if successful_only and not is_success:
            continue
        
        task_id = episode_info.get('task_id', '')
        if not task_id and steps:
            task_id = steps[0].get('task_id', '')
        site_id = extract_site_id(task_id)
        
        for step in steps:
            # Skip steps with errors if we want clean data
            if step.get('last_action_error') and successful_only:
                continue
            
            action = step.get('action', '')
            if not action:
                continue
            
            prompt = build_prompt_from_step(step)
            
            bc_examples.append(BCExample(
                prompt=prompt,
                action=action,
                task_id=task_id,
                site_id=site_id,
                step_idx=step.get('step_idx', 0),
                action_source=step.get('action_source', 'unknown')
            ))
    
    return bc_examples


def generate_dpo_dataset(
    episodes: Dict[str, Dict]
) -> List[DPOExample]:
    """Generate DPO preference pairs dataset.
    
    Pairing rules:
    1. StateKey grouping: For same obs_hash, if an action has no error -> chosen,
       if action has error -> rejected
    2. Trajectory-level fallback: successful episode actions vs failed episode actions
    
    Args:
        episodes: Grouped episode data
        
    Returns:
        List of DPOExample instances
    """
    dpo_examples = []
    
    # Group steps by state (obs_hash) for step-level pairing
    state_actions = defaultdict(list)  # obs_hash -> [(action, success, task_id, step)]
    
    for ep_key, ep_data in episodes.items():
        episode_info = ep_data.get('episode') or {}
        steps = ep_data.get('steps', [])
        task_id = episode_info.get('task_id', '')
        
        for step in steps:
            obs_hash = step.get('obs_hash', '')
            action = step.get('action', '')
            has_error = bool(step.get('last_action_error', ''))
            step_task_id = step.get('task_id', '') or task_id
            
            if obs_hash and action:
                state_actions[obs_hash].append({
                    'action': action,
                    'success': not has_error,
                    'task_id': step_task_id,
                    'step': step
                })
    
    # Generate step-level DPO pairs
    seen_pairs = set()
    
    for obs_hash, actions in state_actions.items():
        successful = [a for a in actions if a['success']]
        failed = [a for a in actions if not a['success']]
        
        # Create pairs: each successful action paired with each failed action
        for succ in successful:
            for fail in failed:
                # Create unique pair key to avoid duplicates
                pair_key = f"{obs_hash}_{succ['action']}_{fail['action']}"
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                prompt = build_prompt_from_step(succ['step'])
                task_id = succ['task_id'] or fail['task_id']
                site_id = extract_site_id(task_id)
                
                dpo_examples.append(DPOExample(
                    prompt=prompt,
                    chosen=succ['action'],
                    rejected=fail['action'],
                    task_id=task_id,
                    site_id=site_id,
                    state_key=obs_hash
                ))
    
    # Trajectory-level fallback: pair successful vs failed episodes for same task
    task_episodes = defaultdict(lambda: {'success': [], 'fail': []})
    
    for ep_key, ep_data in episodes.items():
        episode_info = ep_data.get('episode') or {}
        steps = ep_data.get('steps', [])
        task_id = episode_info.get('task_id', '')
        if not task_id and steps:
            task_id = steps[0].get('task_id', '')
        is_success = episode_info.get('success', False)
        
        if is_success:
            task_episodes[task_id]['success'].append(ep_data)
        else:
            task_episodes[task_id]['fail'].append(ep_data)
    
    # Generate trajectory-level pairs if no step-level pairs exist
    for task_id, task_data in task_episodes.items():
        if not task_data['success'] or not task_data['fail']:
            continue
        
        site_id = extract_site_id(task_id)
        
        # Take first successful and first failed episode
        succ_ep = task_data['success'][0]
        fail_ep = task_data['fail'][0]
        
        succ_steps = succ_ep.get('steps', [])
        fail_steps = fail_ep.get('steps', [])
        
        # Pair corresponding steps (by index)
        min_len = min(len(succ_steps), len(fail_steps))
        
        for i in range(min_len):
            succ_step = succ_steps[i]
            fail_step = fail_steps[i]
            
            succ_action = succ_step.get('action', '')
            fail_action = fail_step.get('action', '')
            
            if not succ_action or not fail_action:
                continue
            
            # Skip if same action
            if succ_action == fail_action:
                continue
            
            prompt = build_prompt_from_step(succ_step)
            state_key = f"traj_{task_id}_{i}"
            
            # Check if we already have this pair from step-level
            pair_key = f"{state_key}_{succ_action}_{fail_action}"
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            dpo_examples.append(DPOExample(
                prompt=prompt,
                chosen=succ_action,
                rejected=fail_action,
                task_id=task_id,
                site_id=site_id,
                state_key=state_key
            ))
    
    return dpo_examples


def _episode_task_id(ep_data: Dict[str, Any]) -> str:
    episode_info = ep_data.get('episode') or {}
    episode_start = ep_data.get('episode_start') or {}
    steps = ep_data.get('steps', [])

    task_id = episode_info.get('task_id', '')
    if not task_id:
        task_id = episode_start.get('task_id', '')
    if not task_id and steps:
        task_id = steps[0].get('task_id', '')
    return task_id


def _episode_max_progress_score(ep_data: Dict[str, Any]) -> float:
    episode_info = ep_data.get('episode') or {}
    max_progress = episode_info.get('max_progress_score', None)
    if isinstance(max_progress, (int, float)):
        return float(max_progress)

    steps = ep_data.get('steps', [])
    best = 0.0
    for step in steps:
        score = step.get('progress_score', 0.0)
        if isinstance(score, (int, float)):
            best = max(best, float(score))
    return best


def _episode_total_steps(ep_data: Dict[str, Any]) -> int:
    episode_info = ep_data.get('episode') or {}
    total = episode_info.get('total_steps', None)
    if isinstance(total, int):
        return int(total)
    steps = ep_data.get('steps', [])
    return int(len(steps))


def _episode_recovery_action_count(ep_data: Dict[str, Any]) -> int:
    steps = ep_data.get('steps', [])
    count = 0
    for step in steps:
        src = step.get('action_source')
        if isinstance(src, str) and src.startswith('recovery_'):
            count += 1
    return count


def _rank_episodes_by_task(episodes: Dict[str, Dict]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep_key, ep_data in episodes.items():
        task_id = _episode_task_id(ep_data)
        if not task_id:
            continue
        grouped[task_id].append(
            {
                'ep_key': ep_key,
                'ep_data': ep_data,
                'max_progress_score': _episode_max_progress_score(ep_data),
                'total_steps': _episode_total_steps(ep_data),
                'recovery_action_count': _episode_recovery_action_count(ep_data),
            }
        )

    for task_id, items in grouped.items():
        items.sort(
            key=lambda x: (
                -float(x['max_progress_score']),
                int(x['total_steps']),
                int(x['recovery_action_count']),
                str(x['ep_key']),
            )
        )
    return grouped


def generate_bc_dataset_progress_ranked(
    episodes: Dict[str, Dict],
    top_percent: float = 0.2,
) -> List[BCExample]:
    if not (0.0 < float(top_percent) <= 1.0):
        raise ValueError(f"top_percent must be in (0, 1], got {top_percent}")

    ranked = _rank_episodes_by_task(episodes)
    bc_examples: List[BCExample] = []

    for task_id, items in ranked.items():
        if not items:
            continue

        k = int(math.ceil(len(items) * float(top_percent)))
        k = max(1, min(k, len(items)))
        chosen_items = items[:k]
        site_id = extract_site_id(task_id)

        for item in chosen_items:
            ep_data = item['ep_data']
            steps = ep_data.get('steps', [])
            for step in steps:
                if step.get('last_action_error'):
                    continue

                action = step.get('action', '')
                if not action:
                    continue

                prompt = build_prompt_from_step(step)

                bc_examples.append(
                    BCExample(
                        prompt=prompt,
                        action=action,
                        task_id=task_id,
                        site_id=site_id,
                        step_idx=step.get('step_idx', 0),
                        action_source=step.get('action_source', 'unknown'),
                    )
                )

    return bc_examples


def generate_dpo_dataset_progress_ranked(episodes: Dict[str, Dict]) -> List[DPOExample]:
    ranked = _rank_episodes_by_task(episodes)
    dpo_examples: List[DPOExample] = []
    seen_pairs = set()

    for task_id, items in ranked.items():
        if len(items) < 2:
            continue

        best = items[0]['ep_data']
        worst = items[-1]['ep_data']

        best_steps = best.get('steps', [])
        worst_steps = worst.get('steps', [])
        if not best_steps or not worst_steps:
            continue

        site_id = extract_site_id(task_id)
        min_len = min(len(best_steps), len(worst_steps))
        for i in range(min_len):
            best_step = best_steps[i]
            worst_step = worst_steps[i]

            chosen = best_step.get('action', '')
            rejected = worst_step.get('action', '')
            if not chosen or not rejected:
                continue
            if chosen == rejected:
                continue

            state_key = f"progress_ranked_{task_id}_{i}"
            pair_key = f"{state_key}_{chosen}_{rejected}"
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            prompt = build_prompt_from_step(best_step)

            dpo_examples.append(
                DPOExample(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    task_id=task_id,
                    site_id=site_id,
                    state_key=state_key,
                )
            )

    return dpo_examples


def save_bc_dataset(examples: List[BCExample], output_path: Path):
    """Save BC dataset to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + '\n')
    
    print(f"Saved {len(examples)} BC examples to {output_path}")


def save_dpo_dataset(examples: List[DPOExample], output_path: Path):
    """Save DPO dataset to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + '\n')
    
    print(f"Saved {len(examples)} DPO examples to {output_path}")


def print_stats(bc_examples: List[BCExample], dpo_examples: List[DPOExample]):
    """Print dataset statistics."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    # BC stats
    print(f"\nBC Dataset: {len(bc_examples)} examples")
    if bc_examples:
        sites = defaultdict(int)
        sources = defaultdict(int)
        for ex in bc_examples:
            sites[ex.site_id] += 1
            sources[ex.action_source] += 1
        
        print("  By site:")
        for site, count in sorted(sites.items()):
            print(f"    {site}: {count}")
        print("  By action source:")
        for source, count in sorted(sources.items()):
            print(f"    {source}: {count}")
    
    # DPO stats
    print(f"\nDPO Dataset: {len(dpo_examples)} pairs")
    if dpo_examples:
        sites = defaultdict(int)
        for ex in dpo_examples:
            sites[ex.site_id] += 1
        
        print("  By site:")
        for site, count in sorted(sites.items()):
            print(f"    {site}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Process HALO-Agent trajectories into training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate both BC and DPO datasets
  python collect_traj.py
  
  # Generate only BC dataset
  python collect_traj.py --format bc
  
  # Generate only DPO dataset  
  python collect_traj.py --format dpo
  
  # Use custom input/output directories
  python collect_traj.py --input_dir data/trajectories/hierarchy_vac_macros --output_dir data/datasets/vac_macros
"""
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/trajectories",
        help="Directory containing trajectory JSONL files (default: data/trajectories)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets",
        help="Output directory for processed datasets (default: data/datasets)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["bc", "dpo", "all"],
        default="all",
        help="Output format: bc, dpo, or all (default: all)"
    )
    parser.add_argument(
        "--pairing_strategy",
        type=str,
        choices=["default", "progress_ranked"],
        default="default",
        help="Pairing strategy for dataset generation (default: default)",
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=0.2,
        help="Top percent episodes per task to keep for BC when pairing_strategy=progress_ranked (default: 0.2)",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include steps from failed episodes in BC dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and analyze data without saving"
    )
    # Legacy argument support
    parser.add_argument("--input-dir", dest="legacy_input_dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", dest="legacy_output_dir", type=str, default=None, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle legacy arguments
    if getattr(args, 'legacy_input_dir', None):
        args.input_dir = getattr(args, 'legacy_input_dir')
    if getattr(args, 'legacy_output_dir', None):
        args.output_dir = getattr(args, 'legacy_output_dir')
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    input_dir = repo_root / args.input_dir
    output_dir = repo_root / args.output_dir
    
    print(f"{'='*50}")
    print("HALO-Agent Trajectory Collection")
    print(f"{'='*50}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Format: {args.format}")
    print(f"Pairing strategy: {args.pairing_strategy}")
    print()
    
    # Check input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print("Run evaluation first to generate trajectories.")
        sys.exit(1)
    
    # Load trajectory files
    records = load_trajectory_files(input_dir)
    
    if not records:
        print("No trajectory records found.")
        sys.exit(0)
    
    # Group by episode
    episodes = group_by_episode(records)
    print(f"Found {len(episodes)} episodes")
    
    # Count successful/failed
    n_success = sum(1 for ep in episodes.values() if (ep.get('episode') or {}).get('success', False))
    n_fail = len(episodes) - n_success
    print(f"  Successful: {n_success}")
    print(f"  Failed: {n_fail}")
    
    # Generate datasets
    bc_examples = []
    dpo_examples = []
    
    if args.format in ['bc', 'all']:
        if args.pairing_strategy == "progress_ranked":
            bc_examples = generate_bc_dataset_progress_ranked(
                episodes,
                top_percent=float(args.top_percent),
            )
        else:
            bc_examples = generate_bc_dataset(
                episodes,
                successful_only=not args.include_failed
            )
    
    if args.format in ['dpo', 'all']:
        if args.pairing_strategy == "progress_ranked":
            dpo_examples = generate_dpo_dataset_progress_ranked(episodes)
        else:
            dpo_examples = generate_dpo_dataset(episodes)
    
    # Print statistics
    print_stats(bc_examples, dpo_examples)
    
    # Save datasets
    if not args.dry_run:
        if bc_examples:
            save_bc_dataset(bc_examples, output_dir / "bc.jsonl")
        
        if dpo_examples:
            save_dpo_dataset(dpo_examples, output_dir / "dpo.jsonl")
        
        print(f"\nDatasets saved to: {output_dir}")
    else:
        print("\nDRY RUN - No files saved")
    
    print(f"\n{'='*50}")
    print("Collection complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
