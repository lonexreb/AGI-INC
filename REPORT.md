# HALO-Agent Project Report

## Executive Summary

HALO-Agent is a browser automation agent for AGI Inc's REAL Benchmark. The project has evolved from a GPT-4o based hierarchical system to an **Online Reinforcement Learning** approach using **Qwen3-VL-8B**, a Vision-Language Model capable of directly processing screenshots for GUI control.

**Current Status**: Deployed on TensorDock H100, running first Online GRPO training.

---

## Project Evolution

### Phase 1: Hierarchical Agent (GPT-4o)

**Goal**: Build a working browser agent using existing LLM APIs.

**What We Built**:
- Worker (gpt-4o-mini) + Manager (gpt-4o) hierarchy
- `src/halo/policy/worker.py` - GPT-4o policy
- `src/halo/agent/orchestrator.py` - Main agent logic
- Verified Action Cache (VAC) and Macro skills

**Limitations Discovered**:
- High API costs (~$0.50/task)
- No learning from experience
- Manager rarely needed (worker handled most cases)

---

### Phase 2: Offline RL Exploration (Qwen2.5-3B)

**Goal**: Train a local model to reduce API costs and enable learning.

**What We Built**:
- `src/halo/policy/qwen_worker.py` - Qwen2.5-3B text policy
- Trajectory collection scripts
- BC (Behavioral Cloning) and DPO training setup

**Limitations Discovered**:
- Required pre-collected trajectories (expensive to gather)
- Text-only model (no vision, relies on AXTree)
- Distribution shift between collected data and live environment

---

### Phase 3: Online RL with VLM (Current)

**Goal**: Real-time learning from screenshots using a Vision-Language Model.

**Why Qwen3-VL-8B**:
- Native GUI agent capability (computer-use)
- Direct 2D coordinate grounding
- Official Computer-Use Cookbook from Qwen team
- 256K context window for long sessions

**What We Built**:

1. **VLM Policy Client** (`src/halo/policy/vllm_client.py`)
   - Connects to vLLM server with OpenAI-compatible API
   - Base64 screenshot encoding
   - `sample_actions(screenshot, goal, n=8)` for GRPO
   - `sample_single(screenshot, goal)` for evaluation
   - Action parsing, validation, and repair

2. **Dense Reward Calculator** (`src/halo/rl/progress.py`)
   - Progress-based rewards per site (Omnizon, GoMail, GoCalendar)
   - Novelty bonus (+0.2) for new states
   - Loop penalty (-0.5) for repeated actions
   - Success bonus (+1.0) for task completion

3. **Online GRPO Trainer** (`src/halo/rl/online_grpo.py`)
   - GRPOConfig dataclass with hyperparameters
   - dr_grpo loss (mean-centering only, handles zero variance)
   - Real-time LoRA weight updates
   - Checkpointing and metrics logging

4. **Training Script** (`scripts/train_online_grpo.py`)
   - CLI entry point with --tasks, --episodes, --dry-run
   - Results saved to `results/grpo_training/`

5. **Deployment Scripts**
   - `launch.sh` - All-in-one setup + vLLM + training
   - `teardown.sh` - Cleanup before stopping instance

---

## Architecture Evolution

```
Phase 1 (GPT-4o)          Phase 2 (Offline RL)       Phase 3 (Online RL) ← CURRENT
─────────────────         ──────────────────         ────────────────────
┌─────────────┐           ┌─────────────┐            ┌─────────────────┐
│   Manager   │           │  BC/DPO     │            │  Qwen3-VL-8B    │
│   (GPT-4o)  │           │  Training   │            │  + Online GRPO  │
├─────────────┤           ├─────────────┤            ├─────────────────┤
│   Worker    │           │ Qwen2.5-3B  │            │  vLLM Server    │
│ (gpt-4o-mini)│          │ (text-only) │            │  (screenshot)   │
├─────────────┤           ├─────────────┤            ├─────────────────┤
│    VAC +    │           │ Trajectory  │            │  Dense Rewards  │
│   Macros    │           │   Replay    │            │  + LoRA Update  │
└─────────────┘           └─────────────┘            └─────────────────┘
     ↓                          ↓                          ↓
 $0.50/task               Needs offline data         Real-time learning
 No learning              No vision                  Screenshot → Action
```

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HALO Agent (Online RL)                   │
├─────────────────────────────────────────────────────────────┤
│  VLM Policy (Qwen3-VL-8B)      │  Dense Reward Calculator   │
│  - Screenshot input (base64)   │  - Progress tracking       │
│  - Action sampling (n=8)       │  - Novelty bonus           │
│  - LoRA fine-tuning            │  - Loop penalty            │
├─────────────────────────────────────────────────────────────┤
│  Online GRPO Trainer                                        │
│  - dr_grpo loss (mean-centering, handles zero variance)     │
│  - Real-time policy updates                                 │
│  - Hot-reload LoRA to vLLM                                  │
├─────────────────────────────────────────────────────────────┤
│  Orchestrator + Recovery Policies                           │
│  - Deterministic fallbacks for loops/errors                 │
│  - Obstruction clearing, form filling                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Supported Modes

| Mode | Description | Model | Status |
|------|-------------|-------|--------|
| `gpt4o_baseline` | GPT-4o for comparison/MCTS critic | gpt-4o | Ready |
| `qwen3vl_base` | Qwen3-VL-8B base (before training) | Qwen3-VL-8B | Ready |
| `qwen3vl_grpo` | Qwen3-VL + Online GRPO LoRA | Qwen3-VL-8B + LoRA | Training |
| `qwen3vl_mcts` | Qwen3-VL + MCTS-trained LoRA | Qwen3-VL-8B + LoRA | Planned |

---

## Technical Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| Task Version | v2 | SDK defaults to v1; must enforce v2 |
| Browser Dimensions | 1280x720 | REAL Benchmark requirement |
| use_html | False | AXTree + screenshot only |
| max_steps | 70 (score-mode) | Benchmark default |
| vLLM Version | >= 0.11.0 | Required for Qwen3-VL support |

---

## File Structure

```
AGI-INC/
├── launch.sh                      # Setup + vLLM + training (run this!)
├── teardown.sh                    # Cleanup before stopping instance
├── src/halo/
│   ├── agent/orchestrator.py      # Main agent logic
│   ├── policy/
│   │   ├── worker.py              # GPT-4o policy (baseline)
│   │   ├── qwen_worker.py         # Qwen text policy
│   │   └── vllm_client.py         # VLM policy (screenshot → action)
│   ├── rl/
│   │   ├── progress.py            # Dense rewards
│   │   └── online_grpo.py         # GRPO trainer
│   ├── sdk/
│   │   ├── agent.py               # HaloAgent class
│   │   └── harness.py             # REAL harness wrapper
│   ├── obs/                       # Observation processing
│   ├── verify/                    # Action verification
│   └── logging/                   # Trajectory logging
├── scripts/
│   ├── smoke_test.py              # Setup verification
│   ├── eval_subset.py             # Evaluation runner
│   ├── train_online_grpo.py       # Training entry point
│   └── list_v2_tasks.py           # Task listing utility
├── configs/
│   └── real_v2_task_registry.json # Task definitions
├── docs/
│   ├── RL_TRACK.md                # RL approach documentation
│   ├── RUNBOOK.md                 # Operational guide
│   └── UITARS_COMPARISON.md       # UI-TARS comparison
└── tests/
    ├── unit/                      # Unit tests
    └── integration/               # Integration tests
```

---

## How to Run

### TensorDock (One Command)
```bash
git clone https://github.com/YOUR_USERNAME/HALO-Agent.git
cd HALO-Agent
./launch.sh
```

`launch.sh` does everything: setup, vLLM server, training.

### Commands
```bash
# Full setup + training (first time)
./launch.sh

# Skip setup, just train (already set up)
./launch.sh --skip-setup

# Custom task and episodes
./launch.sh --skip-setup --task v2.omnizon-1 --episodes 20

# Before stopping instance
./teardown.sh
```

### Local Development
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
playwright install chromium
python scripts/smoke_test.py
```

---

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA H100 80GB HBM3 |
| CUDA | 12.2 |
| Driver | 535.274.02 |
| Provider | TensorDock |
| Cost | ~$2.25/hr |

---

## Progress

### Completed
- [x] Project bootstrap and structure
- [x] GPT-4o hierarchical agent (Phase 1)
- [x] Qwen text-only exploration (Phase 2)
- [x] VLM policy client implementation
- [x] Dense reward calculator
- [x] Online GRPO trainer
- [x] Deployment scripts (launch.sh, teardown.sh)
- [x] TensorDock deployment with H100
- [x] NVIDIA drivers installed and verified

### In Progress
- [ ] First Online GRPO training run
- [ ] Baseline evaluation (qwen3vl_base)
- [ ] Training evaluation (qwen3vl_grpo)

### Pending
- [ ] MCTS integration (Agent Q style)
- [ ] Full benchmark evaluation
- [ ] Performance comparison report

---

## Next Steps

1. **Complete First Training Run**: Monitor `./launch.sh` output
2. **Run Baseline Evaluation**: Establish qwen3vl_base performance
3. **Compare Results**: qwen3vl_base vs qwen3vl_grpo
4. **MCTS Integration**: Implement Agent Q style exploration

---

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
- [Agent Q Paper](https://arxiv.org/abs/2408.07199) - MCTS for web agents
- [Qwen3-VL Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb)
- [AGI SDK Documentation](https://github.com/agi-inc/agisdk)

---

*HALO-Agent v0.3.0 (Online RL)*
