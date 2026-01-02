# HALO-Agent Project Report

## Executive Summary

HALO-Agent is a browser automation agent for AGI Inc's REAL Benchmark. The project has evolved from a GPT-4o based hierarchical system to an **Online Reinforcement Learning** approach using **Qwen3-VL-8B**, a Vision-Language Model capable of directly processing screenshots for GUI control.

---

## Project Evolution

### Phase 1: Hierarchical Agent (GPT-4o)
- **Architecture**: Worker (gpt-4o-mini) + Manager (gpt-4o) hierarchy
- **Features**: Verified Action Cache (VAC), Macro skills, Recovery policies
- **Limitation**: High API costs, no learning from experience

### Phase 2: Offline RL Exploration (Qwen2.5-3B)
- **Approach**: Behavioral Cloning (BC), Direct Preference Optimization (DPO)
- **Model**: Qwen2.5-3B-Instruct (text-only)
- **Limitation**: Required pre-collected trajectories, text-only (no vision)

### Phase 3: Online RL with VLM (Current)
- **Approach**: Online GRPO (Group Relative Policy Optimization)
- **Model**: Qwen3-VL-8B-Instruct (Vision-Language Model)
- **Innovation**: Direct screenshot processing, real-time learning

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
| `qwen3vl_grpo` | Qwen3-VL + Online GRPO LoRA | Qwen3-VL-8B + LoRA | In Development |
| `qwen3vl_mcts` | Qwen3-VL + MCTS-trained LoRA | Qwen3-VL-8B + LoRA | Planned |

---

## Key Components Implemented

### 1. VLM Policy Client (`src/halo/policy/vllm_client.py`)
- Connects to vLLM server with OpenAI-compatible API
- Handles screenshot encoding (base64 PNG)
- `sample_actions(screenshot, goal, n=8)` - For GRPO training
- `sample_single(screenshot, goal)` - For evaluation (greedy)
- Action parsing, validation, and repair

### 2. Dense Reward Calculator (`src/halo/rl/progress.py`)
- Progress-based rewards using site-specific heuristics
- Novelty bonus (+0.2) for visiting new states
- Loop penalty (-0.5) for repeated actions
- Success bonus (+1.0) for task completion
- Supports: Omnizon, GoMail, GoCalendar

### 3. Online GRPO Trainer (`src/halo/rl/online_grpo.py`)
- `GRPOConfig` dataclass with all hyperparameters
- `OnlineGRPOTrainer.train()` - Main training loop
- `compute_advantages()` - dr_grpo style (mean-centering only)
- Checkpointing and metrics logging

### 4. Training Script (`scripts/train_online_grpo.py`)
- CLI entry point for training
- Supports dry-run mode for validation
- Configurable tasks, episodes, temperature
- Results saved to `results/grpo_training/`

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

## File Structure (Current)

```
halo-agent/
├── src/halo/
│   ├── agent/orchestrator.py      # Main agent logic
│   ├── policy/
│   │   ├── worker.py              # GPT-4o policy (baseline)
│   │   ├── qwen_worker.py         # Qwen text policy
│   │   └── vllm_client.py         # VLM policy (NEW)
│   ├── rl/
│   │   ├── progress.py            # Dense rewards
│   │   └── online_grpo.py         # GRPO trainer (NEW)
│   ├── sdk/
│   │   ├── agent.py               # HaloAgent class
│   │   └── harness.py             # REAL harness wrapper
│   ├── obs/                       # Observation processing
│   ├── verify/                    # Action verification
│   └── logging/                   # Trajectory logging
├── scripts/
│   ├── smoke_test.py              # Setup verification
│   ├── eval_subset.py             # Evaluation runner
│   ├── train_online_grpo.py       # Training entry point (NEW)
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

### 1. Setup
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
playwright install --force
```

### 2. Smoke Test
```bash
python scripts/smoke_test.py
```

### 3. Start vLLM Server
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

### 4. Run Evaluation
```bash
python scripts/eval_subset.py --mode qwen3vl_base --subset_size 5
```

### 5. Run Training (Online GRPO)
```bash
# Dry run first
python scripts/train_online_grpo.py --dry-run

# Actual training
python scripts/train_online_grpo.py \
    --tasks v2.omnizon-1 v2.gomail-1 \
    --episodes 20 \
    --num-generations 8
```

---

## Results Summary

### Smoke Test Status
- All HALO modules import successfully
- Supported modes: `gpt4o_baseline`, `qwen3vl_base`, `qwen3vl_grpo`, `qwen3vl_mcts`
- Agent creation works for all modes

### Pending Evaluation
- Full benchmark evaluation pending vLLM server setup
- Online GRPO training ready to run

---

## Next Steps

1. **Run Baseline Evaluation**: Establish qwen3vl_base performance
2. **Online GRPO Training**: Train on subset of tasks
3. **Compare Results**: qwen3vl_base vs qwen3vl_grpo
4. **MCTS Integration**: Implement Agent Q style exploration

---

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) - Group Relative Policy Optimization
- [Agent Q Paper](https://arxiv.org/abs/2408.07199) - MCTS for web agents
- [Qwen3-VL Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb)
- [AGI SDK Documentation](https://github.com/agi-inc/agisdk)

---

*Generated: 2024-12-31*
*HALO-Agent v0.3.0 (Online RL)*
