# HALO-Agent: Online RL for Browser Automation

Train a **Vision-Language browser agent** using **Online Reinforcement Learning**. The agent sees screenshots, learns by interacting with the REAL benchmark environment, and updates its policy in real-time.

**Model:** [Qwen3-VL-8B-Instruct](https://github.com/QwenLM/Qwen3-VL) — a VLM with native GUI agent capability.

## Approach

We implement two Online RL methods:

**1. Online GRPO** - Sample multiple actions per state, execute them, compute group-relative advantages, update policy. Dense reward shaping provides signal even on partial task progress.

**2. MCTS Exploration** - Monte Carlo Tree Search systematically explores the action space. UCB1 balances exploration/exploitation. AI self-critique provides value estimates. Based on Agent Q (arXiv:2408.07199).

## Quick Start

```bash
# Setup on TensorDock H100
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
playwright install chromium

# Add API keys
cp .env.example .env  # Then edit with your keys

# Start vLLM server (in tmux)
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --gpu-memory-utilization 0.7

# Train with Online GRPO
python scripts/train_online_grpo.py --policy_url http://localhost:8000 --domain gomail
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

Recommended: H100 SXM5 80GB (~$2.25/hr) — plenty of headroom for training.
