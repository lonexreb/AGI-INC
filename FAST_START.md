# ⚡ Fast Start: HALO Training Pipeline

**Prerequisite:** You must have an `OPENAI_API_KEY` in `.env` to collect the initial training data (we use GPT-4o as the "Expert" teacher).

## 1. Local Data Collection (The "Hands")
Run this on your local machine (Mac/Linux) where Playwright can launch a browser.

```bash
# A. Install Dependencies
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
playwright install --force

# B. Collect "Expert" Trajectories (Crucial Step)
# We need ~10-50 successful runs to teach the local model.
# This runs the 'hierarchy_vac_macros' mode (GPT-4o) on 'omnizon' tasks.
python scripts/rollout_sampler.py \
    --config configs/experiments.yaml \
    --experiment hierarchy_vac_macros \
    --task_type omnizon \
    --sample_size 20 \
    --rollouts_per_task 1

# C. Verify Data Exists
ls -R data/trajectories/hierarchy_vac_macros

# D. Export for Training
python scripts/collect_traj.py \
    --input_dir data/trajectories \
    --output_dir data/datasets \
    --format all
```

## 2. Remote Training (The "Brain")
Run this on your GPU machine (TensorDock/H100).

```bash
# A. Prepare Directories
sudo mkdir -p /opt/halo/{models,lora,data}
sudo chmod -R 777 /opt/halo

# B. Upload Data (Run this FROM your local machine)
# scp -r data/datasets/bc.jsonl user@gpu-ip:/opt/halo/data/

# C. Start Training Container
docker-compose -f docker-compose.train.yml run --rm train bash

# D. Run Training (Inside Docker)
# This trains a LoRA adapter using Behavioral Cloning (BC)
python scripts/train_bc_unsloth.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --dataset_path /opt/halo/data/bc.jsonl \
    --output_dir /opt/halo/lora/qwen_bc_v1 \
    --max_steps 100
```

## 3. Evaluation
Run the trained model on a subset of tasks to verify performance.

```bash
# Run evaluation with the new local model
python scripts/eval_subset.py \
    --mode qwen_worker_bc \
    --experiment qwen_worker_bc \
    --task_type omnizon \
    --sample_size 10
```
