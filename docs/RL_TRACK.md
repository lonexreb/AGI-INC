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
│                      HALO Agent                              │
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

First, run evaluations to collect trajectories:

```bash
# Run evaluation with OpenAI worker to collect trajectories
python scripts/eval_subset.py \
    --mode hierarchy_vac_macros \
    --sample_size 50 \
    --seed 42 \
    --debug

# Trajectories are saved to data/trajectories/<mode>/<run_id>.jsonl
```

## Step 2: Export Datasets

Convert trajectories to training datasets:

```bash
# Export BC and DPO datasets
python scripts/collect_traj.py \
    --input_dir data/trajectories \
    --output_dir data/datasets

# This creates:
# - data/datasets/bc.jsonl     (prompt, action pairs)
# - data/datasets/dpo.jsonl    (prompt, chosen, rejected triples)
```

### Dataset Formats

**BC Dataset (bc.jsonl)**:
```json
{"prompt": "<observation summary>", "action": "click(\"123\")", "task_id": "v2.omnizon-13", "step": 5}
```

**DPO Dataset (dpo.jsonl)**:
```json
{"prompt": "<observation summary>", "chosen": "click(\"123\")", "rejected": "click(\"456\")", "task_id": "v2.omnizon-13", "step": 5}
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
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000

# Serve with LoRA adapter (BC or DPO)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --enable-lora \
    --lora-modules qwen_bc=checkpoints/qwen_bc_lora \
    --port 8000
```

## Step 6: Run Evaluation with Qwen

### Configure Environment

Set the worker backend in your environment or code:

```bash
export HALO_WORKER_BACKEND=vllm
export HALO_WORKER_MODEL=Qwen/Qwen2.5-3B-Instruct
export HALO_VLLM_URL=http://localhost:8000/v1
```

### Run Evaluation

```bash
# Zero-shot Qwen
python scripts/eval_subset.py \
    --mode qwen_worker_zero \
    --sample_size 30 \
    --seed 42

# BC finetuned Qwen
python scripts/eval_subset.py \
    --mode qwen_worker_bc \
    --sample_size 30 \
    --seed 42

# DPO finetuned Qwen
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
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
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
    ├── trajectories/       # Raw rollouts
    └── datasets/           # Training data
        ├── bc.jsonl
        └── dpo.jsonl
```

---

Last updated: 2025-12-23
