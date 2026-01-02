# START: HALO-Agent Quick Start Guide

Get HALO-Agent running in 5 minutes with Online RL using Qwen3-VL-8B.

---

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM (for vLLM with Qwen3-VL-8B)
- `OPENAI_API_KEY` in `.env` (only needed for `gpt4o_baseline` mode)

---

## 1. Install Dependencies

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install HALO-Agent
pip install -e .

# Install browser automation
playwright install --force
```

---

## 2. Run Smoke Test

Verify everything is set up correctly:

```bash
python scripts/smoke_test.py
```

Expected output:
```
✓ agisdk imported successfully
✓ playwright imported successfully
✓ All HALO modules imported successfully
✓ Supported modes: ['gpt4o_baseline', 'qwen3vl_base', 'qwen3vl_grpo', 'qwen3vl_mcts']
✓ Created agent with mode: qwen3vl_base
SUCCESS: All smoke tests passed!
```

---

## 3. Start vLLM Server

On your GPU machine, start vLLM serving Qwen3-VL-8B:

```bash
# Install vLLM (requires >= 0.11.0 for VLM support)
pip install vllm>=0.11.0

# Start the server
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

Verify it's running:
```bash
curl http://localhost:8000/v1/models
```

---

## 4. Run Evaluation

Test the agent on a subset of REAL Benchmark tasks:

```bash
# Dry run (check config without running)
python scripts/eval_subset.py --mode qwen3vl_base --dry-run

# Run on 5 tasks
python scripts/eval_subset.py --mode qwen3vl_base --subset_size 5

# Run on specific tasks
python scripts/eval_subset.py \
    --mode qwen3vl_base \
    --tasks v2.omnizon-1 v2.gomail-1 v2.gocalendar-4
```

---

## 5. Run Online GRPO Training

Train the agent using Online Reinforcement Learning:

```bash
# Dry run (validate config)
python scripts/train_online_grpo.py --dry-run

# Train on single task (10 episodes)
python scripts/train_online_grpo.py \
    --tasks v2.omnizon-1 \
    --episodes 10

# Full training run
python scripts/train_online_grpo.py \
    --tasks v2.omnizon-1 v2.gomail-1 v2.gocalendar-4 \
    --episodes 30 \
    --num-generations 8 \
    --temperature 0.7 \
    --checkpoint-dir checkpoints/qwen3vl_grpo_lora
```

Results are saved to `results/grpo_training/`.

---

## Agent Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `gpt4o_baseline` | GPT-4o policy | Baseline comparison, MCTS critic |
| `qwen3vl_base` | Qwen3-VL-8B (no training) | Pre-training baseline |
| `qwen3vl_grpo` | Qwen3-VL + GRPO LoRA | After Online RL training |
| `qwen3vl_mcts` | Qwen3-VL + MCTS LoRA | Agent Q style training |

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/smoke_test.py` | Verify setup |
| `scripts/eval_subset.py` | Run evaluation |
| `scripts/train_online_grpo.py` | Online RL training |
| `scripts/list_v2_tasks.py` | List available tasks |
| `src/halo/policy/vllm_client.py` | VLM policy client |
| `src/halo/rl/online_grpo.py` | GRPO trainer |

---

## Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Required for gpt4o_baseline mode
OPENAI_API_KEY=sk-...

# Optional: Override vLLM URL
HALO_VLLM_URL=http://localhost:8000/v1

# Optional: Override worker model
HALO_WORKER_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

---

## Troubleshooting

### vLLM not connecting
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Check logs
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 2>&1 | tee vllm.log
```

### Import errors
```bash
# Reinstall in development mode
pip install -e .

# Verify PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Playwright issues
```bash
# Force reinstall browsers
playwright install --force

# Check browser availability
playwright install chromium
```

---

## Next Steps

1. **Evaluate baseline**: Run `qwen3vl_base` on task subset
2. **Train with GRPO**: Run online training for 20-50 episodes
3. **Compare results**: Evaluate `qwen3vl_grpo` vs baseline
4. **Scale up**: Train on more tasks, more episodes

See `REPORT.md` for detailed project documentation.
