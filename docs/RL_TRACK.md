# HALO-Agent RL Track Documentation

This document describes how to train and deploy Qwen-based worker policies using Behavioral Cloning (BC) and Direct Preference Optimization (DPO).

## Overview

The RL track replaces the OpenAI-based Worker policy with a Qwen model that can be:
1. **Zero-shot**: Use Qwen directly without finetuning
2. **BC finetuned**: Train on successful trajectories
3. **DPO finetuned**: Train on preference pairs (successful vs failed actions)

The Manager policy remains OpenAI-based (gpt-4o) for stability and strategic planning.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HALO Agent                             │
├─────────────────────────────────────────────────────────────┤
│  Worker Policy (Qwen)          │  Manager Policy (OpenAI)   │
│  - Zero-shot                   │  - gpt-4o                  │
│  - BC finetuned (LoRA)         │  - Strategic decisions     │
│  - DPO finetuned (LoRA)        │  - Error recovery          │
├─────────────────────────────────────────────────────────────┤
│  Orchestrator: VAC + Macros + Recovery Policies             │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Collect Rollouts

First, use `rollout_sampler.py` to collect **multiple rollouts per task** with explicit `task_seed` control and exploration (`--temperature`).

```bash
# Example: 3 rollouts per task on a configured subset
python scripts/rollout_sampler.py \
    --config configs/experiments.yaml \
    --experiment hierarchy_vac_macros \
    --subset shopping \
    --sample_size 50 \
    --rollouts_per_task 3 \
    --seed 42 \
    --task_seed 123 \
    --temperature 0.7
```

Trajectories are saved one JSONL file per episode:
- `data/trajectories/<mode>/<run_id>/<task>__attempt_<attempt_idx>.jsonl`

Each episode starts with an `episode_start` record that includes provenance like `task_seed` and `worker_temperature`.

Optional sanity check: verify changing `worker_temperature` changes the action sequence (same seed):

```bash
python scripts/verify_exploration.py \
    --task v2.gomail-1 \
    --mode baseline_worker \
    --task_seed 123 \
    --temperature_a 0.0 \
    --temperature_b 0.7 \
    --compare_steps 10
```

## Step 2: Export Datasets

Convert trajectories to training datasets:

```bash
# Export BC and DPO datasets
python scripts/collect_traj.py \
    --input_dir data/trajectories \
    --output_dir data/datasets \
    --format all

# Export with progress-ranked pairing (ranks episodes per task by progress metrics)
python scripts/collect_traj.py \
    --input_dir data/trajectories \
    --output_dir data/datasets \
    --format all \
    --pairing_strategy progress_ranked \
    --top_percent 0.2

# This creates:
# - data/datasets/bc.jsonl
# - data/datasets/dpo.jsonl
```

### Dataset Formats

**BC Dataset (bc.jsonl)**:
```json
{"prompt": "<observation summary>", "action": "click(\"123\")", "task_id": "v2.omnizon-13", "site_id": "omnizon", "step_idx": 5, "action_source": "worker"}
```

**DPO Dataset (dpo.jsonl)**:
```json
{"prompt": "<observation summary>", "chosen": "click(\"123\")", "rejected": "click(\"456\")", "task_id": "v2.omnizon-13", "site_id": "omnizon", "state_key": "progress_ranked_v2.omnizon-13_5"}
```

## Step 3: Train BC Model

Train a LoRA adapter using Behavioral Cloning:

```bash
# Basic training
python scripts/train_bc_unsloth.py \
    --dataset_path data/datasets/bc.jsonl \
    --output_dir checkpoints/qwen_bc_lora

# With custom settings (for H100/A100)
python scripts/train_bc_unsloth.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --dataset_path data/datasets/bc.jsonl \
    --output_dir checkpoints/qwen_bc_lora \
    --batch_size 8 \
    --epochs 3 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32

# For smaller GPUs (4090)
python scripts/train_bc_unsloth.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8
```

## Step 4: Train DPO Model

Train a LoRA adapter using Direct Preference Optimization:

```bash
# Basic training
python scripts/train_dpo_unsloth.py \
    --dataset_path data/datasets/dpo.jsonl \
    --output_dir checkpoints/qwen_dpo_lora

# With custom settings
python scripts/train_dpo_unsloth.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --dataset_path data/datasets/dpo.jsonl \
    --output_dir checkpoints/qwen_dpo_lora \
    --batch_size 2 \
    --epochs 2 \
    --beta 0.1 \
    --lr 5e-5
```

## Step 5: Serve Qwen via vLLM

For fast inference, serve the model using vLLM:

```bash
# Install vLLM
pip install vllm

# Serve base model (zero-shot)
vllm serve Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000

# Serve with LoRA adapters (BC and/or DPO)
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-lora \
    --lora-modules qwen_bc=checkpoints/qwen_bc_lora qwen_dpo=checkpoints/qwen_dpo_lora
```

## Step 6: Run Evaluation with Qwen

### Configure Environment

Set the worker backend in your environment or code:

- For base model inference, set `HALO_WORKER_MODEL` to the base model ID (e.g., `Qwen/Qwen2.5-3B-Instruct`)
- For vLLM+LoRA, set `HALO_WORKER_MODEL` to the adapter name you passed to `--lora-modules` (e.g., `qwen_bc` / `qwen_dpo`)

```bash
export HALO_WORKER_BACKEND=vllm
export HALO_VLLM_URL=http://localhost:8000/v1
```

### Run Evaluation

```bash
# Zero-shot Qwen
HALO_WORKER_MODEL=Qwen/Qwen2.5-3B-Instruct \
python scripts/eval_subset.py \
    --mode qwen_worker_zero \
    --sample_size 30 \
    --seed 42

# BC finetuned Qwen
HALO_WORKER_MODEL=qwen_bc \
python scripts/eval_subset.py \
    --mode qwen_worker_bc \
    --sample_size 30 \
    --seed 42

# DPO finetuned Qwen
HALO_WORKER_MODEL=qwen_dpo \
python scripts/eval_subset.py \
    --mode qwen_worker_dpo \
    --sample_size 30 \
    --seed 42

# Full matrix comparison
python scripts/eval_full_matrix.py \
    --mode baseline_worker qwen_worker_zero qwen_worker_bc qwen_worker_dpo
```

## Supported Modes

| Mode | Worker | Manager | VAC | Macros |
|------|--------|---------|-----|--------|
| `baseline_worker` | OpenAI gpt-4o-mini | ✗ | ✗ | ✗ |
| `hierarchy_vac_macros` | OpenAI gpt-4o-mini | OpenAI gpt-4o | ✓ | ✓ |
| `qwen_worker_zero` | Qwen (zero-shot) | OpenAI gpt-4o | ✓ | ✓ |
| `qwen_worker_bc` | Qwen (BC LoRA) | OpenAI gpt-4o | ✓ | ✓ |
| `qwen_worker_dpo` | Qwen (DPO LoRA) | OpenAI gpt-4o | ✓ | ✓ |

## Hardware Requirements

### Training

| GPU | Batch Size | Gradient Accum | Notes |
|-----|------------|----------------|-------|
| H100 80GB | 8 | 2 | Full speed |
| A100 40GB | 4 | 4 | Good speed |
| 4090 24GB | 2 | 8 | Slower but works |
| 3090 24GB | 1 | 16 | Very slow |

### Inference

- **vLLM**: Recommended for production. Requires ~8GB VRAM for 3B model.
- **Local Transformers**: Works anywhere but slower. Requires ~6GB VRAM with 4-bit quantization.

## Troubleshooting

### Out of Memory during Training

```bash
# Reduce batch size and increase gradient accumulation
python scripts/train_bc_unsloth.py --batch_size 1 --gradient_accumulation_steps 16
```

### vLLM Server Not Responding

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Restart with more memory
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --gpu-memory-utilization 0.9
```

### LoRA Adapter Not Loading

```bash
# Verify adapter files exist
ls checkpoints/qwen_bc_lora/

# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
```

## Expected Results

Based on preliminary experiments:

| Mode | Success Rate | Median Steps | Notes |
|------|-------------|--------------|-------|
| baseline_worker | ~5-10% | 40-50 | OpenAI only |
| qwen_worker_zero | ~3-8% | 45-55 | No finetuning |
| qwen_worker_bc | ~8-15% | 35-45 | BC improves |
| qwen_worker_dpo | ~10-18% | 30-40 | DPO best |

*Note: Results vary by task type and dataset quality.*

## File Structure

```
HALO-Agent/
├── src/halo/policy/
│   ├── worker.py           # OpenAI worker
│   ├── qwen_worker.py      # Qwen worker
│   └── manager.py          # OpenAI manager
├── scripts/
│   ├── collect_traj.py     # Trajectory collection
│   ├── train_bc_unsloth.py # BC training
│   └── train_dpo_unsloth.py# DPO training
├── checkpoints/
│   ├── qwen_bc_lora/       # BC adapter
│   └── qwen_dpo_lora/      # DPO adapter
└── data/
    ├── trajectories/       # Raw rollouts (one file per episode)
    │   └── <mode>/<run_id>/<task>__attempt_<attempt_idx>.jsonl
    └── datasets/           # Training data
        ├── bc.jsonl
        └── dpo.jsonl
```

---

Last updated: 2025-12-26
