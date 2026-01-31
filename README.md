# HALO-Agent: Online RL for Browser Automation

Train a **Vision-Language browser agent** using **Online Reinforcement Learning** via the **Tinker API**. The agent sees screenshots, learns by interacting with the [REAL benchmark](https://www.theagi.company/blog/introducing-real-bench) environment, and updates its policy in real-time.

**Model:** [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) — a Mixture-of-Experts VLM (30B total, 3B active) with native GUI agent capability.

**Training:** [Tinker API](https://github.com/thinking-machines-lab/tinker-cookbook) — distributed online RL with hosted inference and gradient updates. No local GPU required for training.

## Quick Start

### Training (Tinker API)

```bash
# 1. Install dependencies (Python 3.11+ required)
pip install -r requirements.txt
pip install agisdk==0.3.5 && playwright install --force

# 2. Set your Tinker API key
export TINKER_API_KEY=your_key_here  # https://auth.thinkingmachines.ai/sign-up

# 3. Train on a single task (smoke test)
python scripts/train_tinker.py --tasks v2.omnizon-1 --num-envs 2

# 4. Train on multiple tasks
python scripts/train_tinker.py \
  --tasks v2.omnizon-1 v2.gomail-1 v2.gocalendar-4 \
  --num-envs 8 \
  --loss-fn importance_sampling \
  --learning-rate 1e-6
```

### Evaluation (local vLLM)

```bash
# Full setup + eval on TensorDock (H100/A100)
./launch.sh

# Skip setup, just run
./launch.sh --skip-setup --task v2.omnizon-1 --episodes 20

# Before stopping instance
./teardown.sh
```

## Architecture

```
Training (Tinker API — no local GPU):
  train_tinker.py → Tinker train loop → BrowserEnv.step()
       ↕ Tinker API ↕
  sample() ←→ Qwen3-VL-30B-A3B (hosted remotely)
  forward_backward() + optim_step() → LoRA updates (hosted remotely)

Evaluation (local vLLM — requires GPU):
  eval_subset.py → HaloAgent → VLLMPolicyClient → vLLM → Qwen3-VL-30B-A3B
```

## REAL Benchmark

REAL Bench is a mini-Internet with 112 tasks across 11 sites:

| Site | Clone Of | Tasks |
|------|----------|-------|
| Omnizon | Amazon | 10 |
| GoMail | Gmail | 21 |
| GoCalendar | Google Calendar | 10 |
| DashDish | DoorDash | 11 |
| OpenDining | OpenTable | 11 |
| Staynb | Airbnb | 11 |
| TopWork | Upwork | 12 |
| NetworkIn | LinkedIn | 10 |
| Udriver | Uber | 14 |
| Zilloft | Zillow | 10 |
| MarriSuite | — | 1 |

**Current SOTA:** Claude-3.7-Sonnet-Thinking at 41.1%, AGI Agent-0 at 45%. All using zero-shot prompting — no one is doing online RL on REAL yet.

## Approach

### 1. Online GRPO via Tinker (Primary)

Group Relative Policy Optimization with dense reward shaping:
- `BrowserGroupBuilder` creates N=8 parallel browser environments per task
- Tinker runs rollouts, computes group-relative advantages, updates LoRA weights
- Dense rewards from `progress.py` provide signal on partial progress (not just success/fail)
- Loss: `importance_sampling` (handles high reward variance in browser tasks)

### 2. MCTS Exploration (Agent Q Style)

Monte Carlo Tree Search for systematic action space exploration:
- UCB1 balances exploration vs exploitation
- AI self-critique ranks trajectory quality
- Best paths used for step-level DPO updates
- Based on Agent Q (LLaMA-3 70B: 18.6% → 81.7% on OpenTable)

## Agent Modes

| Mode | Description |
|------|-------------|
| `qwen3vl_base` | Qwen3-VL-30B-A3B before training (baseline) |
| `qwen3vl_grpo` | Qwen3-VL + Online GRPO LoRA |
| `qwen3vl_mcts` | Qwen3-VL + MCTS-trained LoRA |
| `gpt5_baseline` | GPT-5.2 for comparison |

## Project Structure

```
├── scripts/
│   ├── train_tinker.py         # Tinker RL training entry point
│   ├── train_online_grpo.py    # Local GRPO training (legacy)
│   ├── eval_subset.py          # Evaluation runner
│   └── smoke_test.py           # Verify setup
├── src/halo/
│   ├── constants.py            # DEFAULT_MODEL
│   ├── tinker/                 # Tinker API integration
│   │   ├── browser_env.py      # Tinker Env wrapping REAL benchmark
│   │   ├── group_builder.py    # GRPO group builder
│   │   ├── dataset.py          # Task batching for training loop
│   │   └── action_parser.py    # Parse model output → browser actions
│   ├── rl/
│   │   ├── progress.py         # Dense rewards (site-specific)
│   │   ├── online_grpo.py      # GRPO trainer (local)
│   │   └── mcts.py             # MCTS exploration
│   ├── obs/                    # Observation processing
│   │   ├── obs_summarizer.py   # AXTree extraction
│   │   ├── page_type.py        # Page classification
│   │   └── fingerprint.py      # State hashing, loop detection
│   ├── policy/vllm_client.py   # VLM policy for eval
│   └── sdk/
│       ├── harness.py          # REAL benchmark harness
│       └── agent.py            # Agent interface
├── configs/
│   └── real_v2_task_registry.json  # 121 v2 tasks
├── checkpoints/                # Saved LoRA adapters
├── launch.sh                   # TensorDock setup + eval
└── teardown.sh                 # Cleanup
```

## Dense Rewards

Site-specific milestones provide learning signal even on partial progress:

**Omnizon:** Home (0.1) → Search (0.3) → Product (0.5) → Cart (0.7) → Checkout (0.85) → Confirmation (1.0)

**GoMail:** Inbox (0.1) → Compose/View (0.1) → Action taken (0.55) → Confirmation (1.0)

**GoCalendar:** Calendar (0.2) → Event form (0.4) → Details filled (0.7) → Confirmation (1.0)

**Additional shaping:** +0.1 valid action format, +0.2 new state, -0.5 loop detected, -0.2 action error

## Prerequisites

- **Python 3.11+** (Tinker requires >= 3.11)
- **Tinker API key** — [Sign up](https://auth.thinkingmachines.ai/sign-up)
- **agisdk 0.3.5** + Playwright (for browser environments)
- **GPU** only needed for local eval (vLLM); training uses Tinker's hosted infrastructure

## Documentation

- `CLAUDE.md` — Development instructions and architecture details
- `REPORT.md` — Project report
- `START.md` — Quick start guide
