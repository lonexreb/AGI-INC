# HALO-Agent

**Hierarchical Worker+Manager Browser Agent with RL Training Pipeline for AGI Inc REAL Bench**

HALO-Agent is a hierarchical browser automation agent designed for AGI Inc's REAL Bench evaluation suite. It implements a worker-manager architecture with verified action caching, macro skill replay, and a complete RL training pipeline (BC → DPO → GRPO) using Qwen models.

## Overview

HALO-Agent combines multiple decision-making strategies in a hierarchical pipeline:

1. **Macro Replay Cache** — Reuse learned skill sequences
2. **Verified Action Cache (VAC)** — Cache verified state-action pairs
3. **Worker Policy** — Fast, lightweight decision-making (OpenAI gpt-4o-mini or Qwen)
4. **Manager Policy** — High-stakes oversight and error recovery (gpt-4o)

### RL Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HALO RL Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. Rollout Collection  →  2. Dataset Export  →  3. Model Training  │
│     rollout_sampler.py      collect_traj.py       train_*.py        │
├─────────────────────────────────────────────────────────────────────┤
│  Training Methods: BC (Behavioral Cloning)                          │
│                    DPO (Direct Preference Optimization)             │
│                    GRPO (Group Relative Policy Optimization)        │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Create virtual environment (Python 3.10+ required)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install Playwright browsers
playwright install --force

# Copy environment template
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run Smoke Test

```bash
python scripts/smoke_test.py
```

### Run Evaluation

```bash
# Run on 50 random v2 tasks
python scripts/eval_subset.py --mode baseline_worker

# Run specific tasks
python scripts/eval_subset.py --tasks v2.omnizon-13,v2.gomail-1 --mode hierarchy_vac_macros

# Debug a single task with visible browser
python scripts/run_one_debug.py --task v2.omnizon-13
```

### List Available Tasks

```bash
python scripts/list_v2_tasks.py
```

## Agent Modes

| Mode | Worker | Manager | VAC | Macros |
|------|--------|---------|-----|--------|
| `baseline_worker` | gpt-4o-mini | ✗ | ✗ | ✗ |
| `hierarchy_mgr_gate` | gpt-4o-mini | gpt-4o | ✗ | ✗ |
| `hierarchy_vac` | gpt-4o-mini | gpt-4o | ✓ | ✗ |
| `hierarchy_vac_macros` | gpt-4o-mini | gpt-4o | ✓ | ✓ |
| `qwen_worker_zero` | Qwen (zero-shot) | gpt-4o | ✓ | ✓ |
| `qwen_worker_bc` | Qwen (BC LoRA) | gpt-4o | ✓ | ✓ |
| `qwen_worker_dpo` | Qwen (DPO LoRA) | gpt-4o | ✓ | ✓ |
| `qwen_worker_grpo` | Qwen (GRPO LoRA) | gpt-4o | ✓ | ✓ |

## Project Structure

```
HALO-Agent/
├── src/halo/              # Core agent implementation
│   ├── agent/             # Orchestrator and routing logic
│   ├── policy/            # Worker (OpenAI/Qwen), Manager, Gating
│   ├── cache/             # VAC and macro cache
│   ├── verify/            # Postcondition verification, loop detection
│   ├── obs/               # Observation summarizer, fingerprinting
│   ├── logging/           # Structured JSONL logging
│   ├── rl/                # RL utilities
│   └── sdk/               # AGI SDK wrappers
├── scripts/
│   ├── eval_subset.py         # Subset evaluation runner
│   ├── eval_full_matrix.py    # Full benchmark matrix
│   ├── run_one_debug.py       # Single-task debug runner
│   ├── rollout_sampler.py     # RL data collection
│   ├── collect_traj.py        # Export BC/DPO datasets
│   ├── collect_expert_traj.py # Expert trajectory collection
│   ├── train_bc_unsloth.py    # BC training with Unsloth
│   ├── train_dpo_unsloth.py   # DPO training with Unsloth
│   ├── train_grpo_unsloth.py  # GRPO training with Unsloth
│   ├── list_v2_tasks.py       # Task discovery utility
│   └── smoke_test.py          # Environment smoke test
├── configs/
│   └── experiments.yaml   # Ablation experiment definitions
├── data/
│   ├── trajectories/      # Raw rollout logs
│   ├── datasets/          # Exported BC/DPO training data
│   └── cache/             # VAC cache data
├── results/               # Evaluation results
├── tests/                 # Unit and integration tests
└── docs/
    ├── RUNBOOK.md         # Detailed setup and operations
    └── RL_TRACK.md        # RL training documentation
```

## RL Training Pipeline

### 1. Collect Rollouts

```bash
python scripts/rollout_sampler.py \
    --config configs/experiments.yaml \
    --experiment hierarchy_vac_macros \
    --subset shopping \
    --rollouts_per_task 3 \
    --temperature 0.7
```

### 2. Export Datasets

```bash
python scripts/collect_traj.py \
    --input_dir data/trajectories \
    --output_dir data/datasets \
    --format all \
    --pairing_strategy progress_ranked
```

### 3. Train Models

```bash
# Behavioral Cloning
python scripts/train_bc_unsloth.py \
    --dataset_path data/datasets/bc.jsonl \
    --output_dir checkpoints/qwen_bc_lora

# Direct Preference Optimization
python scripts/train_dpo_unsloth.py \
    --dataset_path data/datasets/dpo.jsonl \
    --output_dir checkpoints/qwen_dpo_lora

# Group Relative Policy Optimization
python scripts/train_grpo_unsloth.py \
    --dataset_path data/datasets/bc.jsonl \
    --output_dir checkpoints/qwen_grpo_lora
```

### 4. Serve with vLLM

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --enable-lora \
    --lora-modules qwen_bc=checkpoints/qwen_bc_lora
```

### 5. Evaluate

```bash
export HALO_WORKER_BACKEND=vllm
export HALO_VLLM_URL=http://localhost:8000/v1
HALO_WORKER_MODEL=qwen_bc python scripts/eval_subset.py --mode qwen_worker_bc
```

## Requirements

- **Python 3.10+** (required for `match` statements in AGI SDK)
- AGI SDK 0.3.5 (pinned)
- Playwright
- OpenAI API key (for Manager and baseline Worker)
- *Optional*: GPU + vLLM for Qwen inference

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `HALO_WORKER_BACKEND` | No | `vllm` or `local` for Qwen |
| `HALO_VLLM_URL` | No | vLLM server URL |
| `HALO_WORKER_MODEL` | No | Model ID for Qwen worker |

## Critical Constraints

⚠️ **Always use `task_version="v2"`** — SDK defaults to v1 if omitted!

- Use `v2.` prefix for task IDs (e.g., `v2.omnizon-13`)
- Browser viewport: 1280×720
- Observations: AXTree + Screenshot (no HTML)
- Max steps: 70 (score mode), 25 (speed mode)

## Documentation

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — Detailed setup, CLI options, troubleshooting
- [`docs/RL_TRACK.md`](docs/RL_TRACK.md) — RL training pipeline documentation
- [`CLAUDE.md`](CLAUDE.md) — Project constitution and development guidelines

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# Unit tests only
python -m pytest tests/unit/ -v
```

## License

Internal project for AGI Inc.
