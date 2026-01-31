# HALO-Agent: Online RL for Browser Automation

Train a **Vision-Language browser agent** using **Online Reinforcement Learning**. The agent sees screenshots, learns by interacting with the REAL benchmark environment, and updates its policy in real-time.

**Model:** [Qwen3-VL-8B-Instruct](https://github.com/QwenLM/Qwen3-VL) — a VLM with native GUI agent capability.

## Quick Start (TensorDock)

```bash
# Clone and run - that's it!
git clone https://github.com/YOUR_USERNAME/HALO-Agent.git
cd HALO-Agent
./launch.sh
```

`launch.sh` does everything:
- Installs system dependencies
- Sets up Python environment
- Installs PyTorch, vLLM, Qwen dependencies
- Downloads model (~16GB, cached after first run)
- Starts vLLM server
- Runs training

**First run:** ~15-20 min (downloads model)
**Subsequent runs:** ~5 min

### Commands

```bash
# Full setup + training (first time)
./launch.sh

# Skip setup, just train (already set up)
./launch.sh --skip-setup

# Custom task and episodes
./launch.sh --skip-setup --task v2.omnizon-1 --episodes 20

# Before stopping TensorDock instance
./teardown.sh
```

## Approach

We implement two Online RL methods:

**1. Online GRPO** - Sample multiple actions per state, execute them, compute group-relative advantages, update policy. Dense reward shaping provides signal even on partial task progress.

**2. MCTS Exploration** - Monte Carlo Tree Search systematically explores the action space. UCB1 balances exploration/exploitation. AI self-critique provides value estimates. Based on Agent Q (arXiv:2408.07199).

## Agent Modes

| Mode | Description |
|------|-------------|
| `qwen3vl_base` | Qwen3-VL-8B before training (baseline) |
| `qwen3vl_grpo` | Qwen3-VL + Online GRPO LoRA |
| `qwen3vl_mcts` | Qwen3-VL + MCTS-trained LoRA |
| `gpt5_baseline` | GPT-5.2 for comparison (best vision model) |

## Project Structure

```
├── launch.sh                 # Setup + vLLM + training (run this!)
├── teardown.sh               # Cleanup before stopping instance
├── scripts/
│   ├── train_online_grpo.py  # Training entry point
│   ├── eval_subset.py        # Evaluation runner
│   └── smoke_test.py         # Verify setup
├── src/halo/
│   ├── policy/vllm_client.py # VLM policy (screenshot → action)
│   ├── rl/online_grpo.py     # GRPO trainer
│   └── rl/progress.py        # Dense rewards
└── checkpoints/              # Saved LoRA adapters
```

## Why Qwen3-VL?

- **Visual Agent capability** - Built for GUI control (PC/mobile)
- **2D grounding** - Outputs click coordinates directly
- **Official Computer-Use Cookbook** - [Proven for browser tasks](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb)
- **256K context** - Handles long browsing sessions

## Hardware

Everything runs on **TensorDock** (H100 or A100 GPU):
- vLLM server (Qwen3-VL-8B, ~20GB VRAM)
- Browser environment (Playwright headless Chromium)
- Training loop (LoRA weight updates)

Recommended: H100 SXM5 80GB (~$2.25/hr)

## Documentation

- `START.md` - Quick start guide
- `REPORT.md` - Project report
- `CLAUDE.md` - Development instructions
