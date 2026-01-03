# START: HALO-Agent Quick Start

## TensorDock (Recommended)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/HALO-Agent.git
cd HALO-Agent

# 2. Run everything
./launch.sh
```

That's it. `launch.sh` handles:
- System dependencies
- Python environment
- PyTorch + CUDA
- vLLM installation
- Model download (~16GB)
- vLLM server startup
- Training

**First run:** ~15-20 min
**Subsequent:** ~5 min

## Commands

```bash
# Full setup + training
./launch.sh

# Already set up - just train
./launch.sh --skip-setup

# Different task
./launch.sh --skip-setup --task v2.omnizon-1

# More episodes
./launch.sh --skip-setup --task v2.gomail-1 --episodes 20

# Before stopping instance
./teardown.sh
```

## Local Development (Mac/Linux)

```bash
# Setup
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
playwright install chromium

# Smoke test (no GPU needed)
python scripts/smoke_test.py

# Evaluation requires vLLM server on GPU machine
```

## Manual Training Commands

If you prefer running steps manually:

```bash
# Activate environment
source venv/bin/activate

# Start vLLM (in tmux)
tmux new -s vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --gpu-memory-utilization 0.7
# Ctrl+B D to detach

# Dry run
python scripts/train_online_grpo.py --dry-run

# Train
python scripts/train_online_grpo.py --tasks v2.gomail-1 --episodes 10
```

## Troubleshooting

**vLLM not starting:**
```bash
tail -f outputs/logs/vllm.log
```

**Import errors:**
```bash
pip install -e .
python scripts/smoke_test.py
```

**GPU not detected:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```
