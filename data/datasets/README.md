# Datasets

This folder contains JSONL datasets exported from HALO-Agent trajectory logs for training a Qwen worker policy (BC / DPO).

## Inputs: Trajectories

Trajectories are written under:

- `data/trajectories/<mode>/<run_id>/`

Each episode is a single JSONL file containing records of type:

- `episode_start`
- `step`
- `episode_end`

### Progress fields

Progress is logged on every `step`:

- `progress_score`: `float`
- `milestones`: `list[str] | null`

Progress aggregates are logged on `episode_end`:

- `max_progress_score`: `float`
- `final_progress_score`: `float`

### Provenance fields (episode_start)

`episode_start` includes run/task provenance used for debugging and reproducibility.
Newer runs include:

- `task_seed`: `int | null`
- `worker_temperature`: `float`
- `worker_model`: `str`
- `manager_model`: `str`
- `use_manager`: `bool`
- `use_cache`: `bool`
- `use_macros`: `bool`
- `max_steps`: `int`
- `manager_warm_start`: `bool`
- `enable_recovery_policies`: `bool`
- `always_call_manager`: `bool`
- `qwen_backend`: `str`
- `qwen_base_url`: `str`

Older trajectory files may not contain all of these fields.

## Export: collect_traj.py

Use `scripts/collect_traj.py` to export datasets from trajectories:

```bash
python scripts/collect_traj.py \
  --input_dir data/trajectories/qwen_worker_zero \
  --output_dir data/datasets/qwen_worker_zero \
  --format all
```

Outputs:

- `bc.jsonl`
- `dpo.jsonl`

## Dataset schemas

### BC (`bc.jsonl`)

One line per action:

```json
{
  "prompt": "...",
  "action": "click(\"a1\")",
  "task_id": "v2.omnizon-13",
  "site_id": "omnizon",
  "step_idx": 7,
  "action_source": "worker"
}
```

Default behavior:

- Only uses *successful* episodes.
- Skips steps with `last_action_error`.

To include failed episodes:

```bash
python scripts/collect_traj.py --include-failed
```

### DPO (`dpo.jsonl`)

One line per preference pair:

```json
{
  "prompt": "...",
  "chosen": "click(\"a1\")",
  "rejected": "click(\"b2\")",
  "task_id": "v2.omnizon-13",
  "site_id": "omnizon",
  "state_key": "..."
}
```

## Pairing strategies

### default

- Step-level pairing: group by `obs_hash` and pair a no-error action (chosen) against an error action (rejected).
- Fallback: pair steps from a successful episode against steps from a failed episode for the same task.

### progress_ranked

Enable with:

```bash
python scripts/collect_traj.py \
  --input_dir data/trajectories/qwen_worker_zero \
  --output_dir data/datasets/qwen_worker_zero_progress_ranked \
  --pairing_strategy progress_ranked \
  --top_percent 0.2 \
  --format all
```

Behavior:

- Episodes are ranked *per task* by:
  - `max_progress_score` (descending)
  - `total_steps` (ascending)
  - number of recovery actions (ascending; `action_source` starts with `recovery_`)

BC:

- Keeps the top `top_percent` episodes per task (ceil, minimum 1).
- Does **not** require success.
- Skips steps with `last_action_error`.

DPO:

- For each task, pairs the best episode vs the worst episode.
- Pairs steps by index (`i`) and emits `{chosen=best[i], rejected=worst[i]}`.
- Does **not** require success.
