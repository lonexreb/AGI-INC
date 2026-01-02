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

For reproducibility across machines and SDK updates, snapshot the REAL v2 task registry and use it for rollouts/evals.

```bash
python scripts/snapshot_real_tasks.py --task_version v2 --out configs/real_v2_task_registry.json
```

First, use `rollout_sampler.py` to collect **multiple rollouts per task** with explicit `task_seed` control and exploration (`--temperature`).

```bash
# Example: 3 rollouts per task on a configured subset
python scripts/rollout_sampler.py \
    --config configs/experiments.yaml \
    --experiment hierarchy_vac_macros \
    --subset shopping \
    --task_registry configs/real_v2_task_registry.json \
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
    --task_registry configs/real_v2_task_registry.json \
    --seed 42

# BC finetuned Qwen
HALO_WORKER_MODEL=qwen_bc \
python scripts/eval_subset.py \
    --mode qwen_worker_bc \
    --sample_size 30 \
    --task_registry configs/real_v2_task_registry.json \
    --seed 42

# DPO finetuned Qwen
HALO_WORKER_MODEL=qwen_dpo \
python scripts/eval_subset.py \
    --mode qwen_worker_dpo \
    --sample_size 30 \
    --task_registry configs/real_v2_task_registry.json \
    --seed 42

# Full matrix comparison
python scripts/eval_full_matrix.py \
    --mode baseline_worker qwen_worker_zero qwen_worker_bc qwen_worker_dpo
    --task_registry configs/real_v2_task_registry.json
```

## TensorDock H100: Qwen + vLLM + GRPO + REAL v2

This section is an end-to-end minimal RL-only pipeline (no manager, no VAC, no macros) using vLLM for serving.

### 0) Paths

```bash
sudo mkdir -p /opt/halo/models
sudo mkdir -p /opt/halo/lora
```

### 1) Snapshot REAL v2 task registry

```bash
python scripts/snapshot_real_tasks.py --task_version v2 --out configs/real_v2_task_registry.json
```

### 2) Download Qwen (base model)

```bash
python -m pip install -U huggingface_hub

huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /opt/halo/models/Qwen3-4B-Instruct-2507 \
  --local-dir-use-symlinks False
```

### 3) Run vLLM (base model) on port 8000

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -v /opt/halo/models:/models \
  vllm/vllm-openai:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-4B-Instruct-2507 \
    --served-model-name Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype auto
```

In another shell:

```bash
export HALO_WORKER_BACKEND=vllm
export HALO_VLLM_URL=http://localhost:8000/v1
```

### 4) Collect rollouts (pure Qwen)

```bash
python scripts/rollout_sampler.py \
  --config configs/experiments.yaml \
  --experiment qwen_pure_zero \
  --subset shopping \
  --task_registry configs/real_v2_task_registry.json \
  --sample_size 50 \
  --rollouts_per_task 3 \
  --seed 42 \
  --task_seed 123 \
  --temperature 0.8
```

Trajectories will be written under:

```text
data/trajectories/qwen_pure_zero/<run_id>/*.jsonl
```

### 5) Train GRPO LoRA adapter

```bash
python scripts/train_grpo_unsloth.py \
  --base_model /opt/halo/models/Qwen3-4B-Instruct-2507 \
  --input_dir data/trajectories/qwen_pure_zero \
  --output_dir /opt/halo/lora/qwen3_4b_grpo_lora \
  --top_percent 0.2
```

### 6) Restart vLLM with the GRPO adapter enabled

Stop the prior vLLM container, then:

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -v /opt/halo/models:/models \
  -v /opt/halo/lora:/lora \
  vllm/vllm-openai:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-4B-Instruct-2507 \
    --served-model-name Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype auto \
    --enable-lora \
    --lora-modules qwen_grpo=/lora/qwen3_4b_grpo_lora
```

### 7) Evaluate (REAL v2, snapshot registry)

These experiments do not require `OPENAI_API_KEY` because `use_manager=false`.

```bash
python scripts/eval_full_matrix.py \
  --config configs/experiments.yaml \
  --experiment qwen_pure_zero qwen_pure_grpo \
  --subset shopping \
  --task_registry configs/real_v2_task_registry.json
```

---

## Docker-based Training (Recommended)

For easier deployment on GPU machines, use the provided Docker setup.

### Prerequisites

```bash
# On the GPU host (e.g., TensorDock H100)
sudo mkdir -p /opt/halo/{models,lora,data}

# Download Qwen model
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /opt/halo/models/Qwen3-4B-Instruct-2507 \
  --local-dir-use-symlinks False
```

### Build the training image

```bash
docker build -f Dockerfile.train -t halo-train:latest .
```

### Start vLLM for rollout collection

```bash
docker-compose -f docker-compose.train.yml up -d vllm
```

### Run training interactively

```bash
docker-compose -f docker-compose.train.yml run --rm train bash

# Inside container:
python scripts/rollout_sampler.py \
  --config configs/experiments.yaml \
  --experiment qwen_pure_zero \
  --task_registry configs/real_v2_task_registry.json \
  --sample_size 50 --rollouts_per_task 3

python scripts/train_grpo_unsloth.py \
  --base_model /models/Qwen3-4B-Instruct-2507 \
  --input_dir data/trajectories/qwen_pure_zero \
  --output_dir /lora/qwen3_4b_grpo_lora \
  --top_percent 0.2
```

### Start vLLM with LoRA for evaluation

```bash
docker-compose -f docker-compose.train.yml --profile lora up -d vllm-lora

# Evaluate on port 8001
HALO_VLLM_URL=http://localhost:8001/v1 python scripts/eval_full_matrix.py \
  --config configs/experiments.yaml \
  --experiment qwen_pure_grpo \
  --task_registry configs/real_v2_task_registry.json
```

### Install RL dependencies locally (alternative)

```bash
pip install -e ".[rl]"
# or
pip install -r requirements-rl.txt
```

---

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
│   ├── qwen_worker.py      # Qwen worker (vLLM/local)
│   └── manager.py          # OpenAI manager
├── scripts/
│   ├── rollout_sampler.py      # Rollout collection with seed control
│   ├── collect_traj.py         # Export BC/DPO datasets
│   ├── train_bc_unsloth.py     # BC training
│   ├── train_dpo_unsloth.py    # DPO training
│   ├── train_grpo_unsloth.py   # GRPO training (from trajectories)
│   └── snapshot_real_tasks.py  # Task registry snapshot
├── configs/
│   ├── experiments.yaml            # Experiment definitions
│   └── real_v2_task_registry.json  # Snapshotted task list
├── Dockerfile.train            # GPU training image
├── docker-compose.train.yml    # Training orchestration
├── requirements-rl.txt         # RL dependencies
└── data/
    ├── trajectories/       # Raw rollouts (one file per episode)
    │   └── <mode>/<run_id>/<task>__attempt_<attempt_idx>.jsonl
    └── datasets/           # Training data
        ├── bc.jsonl
        └── dpo.jsonl

# On GPU host (TensorDock/Runpod):
/opt/halo/
├── models/                 # Base models (Qwen3-4B-Instruct-2507)
├── lora/                   # Trained LoRA adapters
└── data/                   # Persistent trajectory data
```

---

## TensorDock Deployment Checklist (Split Setup)

This setup runs **vLLM + training on TensorDock** and **rollout collection locally** (where Playwright works).

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Split Setup Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│  LOCAL MACHINE                    │  TENSORDOCK H100                │
│  (has Playwright)                 │  (has GPU)                      │
│                                   │                                 │
│  1. SSH tunnel to vLLM ──────────────► vLLM (port 8000)             │
│  2. Collect rollouts              │                                 │
│  3. SCP trajectories ────────────────► /opt/halo/data/trajectories  │
│                                   │  4. GRPO training (Docker)      │
│                                   │  5. vLLM + LoRA eval            │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 1: TensorDock Setup

```bash
# SSH into TensorDock
ssh user@tensordock-ip

# Verify GPU
nvidia-smi

# Create directories
sudo mkdir -p /opt/halo/{models,lora,data/trajectories,data/datasets}

# Clone repo
git clone <repo-url> HALO-Agent && cd HALO-Agent

# Download Qwen model
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /opt/halo/models/Qwen3-4B-Instruct-2507

# Snapshot task registry (for reproducibility)
pip install -e .
python scripts/snapshot_real_tasks.py --out configs/real_v2_task_registry.json

# Start vLLM
docker-compose -f docker-compose.train.yml up -d vllm

# Wait for healthy (takes ~2 min for model load)
docker-compose -f docker-compose.train.yml ps
curl http://localhost:8000/v1/models
```

### Phase 2: Local Rollout Collection

On your **local machine** (where Playwright is installed):

```bash
# Terminal 1: SSH tunnel to TensorDock vLLM
ssh -L 8000:localhost:8000 user@tensordock-ip

# Terminal 2: Collect rollouts (Playwright runs locally)
cd HALO-Agent
export HALO_WORKER_BACKEND=vllm
export HALO_VLLM_URL=http://localhost:8000/v1

python scripts/rollout_sampler.py \
  --experiment qwen_pure_zero \
  --task_registry configs/real_v2_task_registry.json \
  --sample_size 50 \
  --rollouts_per_task 3 \
  --seed 42 \
  --task_seed 123 \
  --temperature 0.8

# Trajectories saved to: data/trajectories/qwen_worker_zero/<run_id>/
```

### Phase 3: Upload Trajectories to TensorDock

```bash
# From local machine
scp -r data/trajectories/ user@tensordock-ip:/opt/halo/data/trajectories/
```

### Phase 4: GRPO Training on TensorDock

```bash
# SSH into TensorDock
ssh user@tensordock-ip
cd HALO-Agent

# Run training container
docker-compose -f docker-compose.train.yml run --rm train bash

# Inside container - trajectories are mounted at data/trajectories/
python scripts/train_grpo_unsloth.py \
  --base_model /models/Qwen3-4B-Instruct-2507 \
  --input_dir data/trajectories/qwen_worker_zero \
  --output_dir /lora/qwen3_4b_grpo_lora \
  --top_percent 0.2

# Exit container when done
exit
```

### Phase 5: Evaluation with LoRA

```bash
# Stop base vLLM
docker-compose -f docker-compose.train.yml down

# Start vLLM with LoRA adapter
docker-compose -f docker-compose.train.yml --profile lora up -d vllm-lora

# Verify LoRA loaded
curl http://localhost:8001/v1/models
# Should show: qwen_grpo

# Run eval (inside training container or directly)
docker-compose -f docker-compose.train.yml run --rm train \
  python scripts/eval_full_matrix.py \
    --experiment qwen_pure_zero qwen_pure_grpo \
    --task_registry configs/real_v2_task_registry.json
```

### Phase 6: Teardown & Backup

```bash
# Copy results to local
scp -r user@tensordock-ip:~/HALO-Agent/results/ ./results-backup/
scp -r user@tensordock-ip:/opt/halo/lora/ ./lora-backup/

# Stop containers
ssh user@tensordock-ip "cd HALO-Agent && docker-compose -f docker-compose.train.yml down"
```

### Quick Reference

| Phase | Where | Key Command |
|-------|-------|-------------|
| vLLM | TensorDock | `docker-compose up -d vllm` |
| Rollouts | Local + SSH tunnel | `ssh -L 8000:localhost:8000 ...` then `rollout_sampler.py` |
| Upload | Local → TensorDock | `scp -r data/trajectories/ ...` |
| Training | TensorDock | `docker-compose run --rm train` |
| Eval | TensorDock | `--profile lora up -d vllm-lora` |

---

Last updated: 2025-12-29
