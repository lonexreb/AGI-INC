# CLAUDE.md - Project Context for AI Assistants

## What We're Building

HALO-Agent is a browser automation agent for AGI Inc's REAL Benchmark. We're training **Qwen3-VL-30B-A3B** (a Mixture-of-Experts Vision-Language Model, 30B total / 3B active params) using **Online Reinforcement Learning** via the **Tinker API** to complete web tasks autonomously.

**Why Qwen3-VL-30B-A3B?**
- Built-in "Visual Agent" capability for GUI control
- 2D grounding (outputs click coordinates directly)
- MoE architecture: 30B total params but only 3B active per forward pass (fast inference)
- Supported by Tinker API for distributed online RL training
- 256K native context for long browsing sessions

## The Goal: Online RL

We are doing **Online RL**, not offline imitation learning. This means:

- The agent interacts with the live environment
- It receives rewards based on its actions
- It updates its policy in real-time
- No pre-collected demonstration datasets required

**Do NOT suggest Behavioral Cloning (BC) or offline approaches as prerequisites.** The whole point is to learn from scratch through environment interaction.

## Training: Tinker API for Online RL

We use the **Tinker API** (by Thinking Machines Lab) for distributed online RL. Tinker handles model hosting, inference, and gradient updates; we provide the browser environment, reward function, and observation encoding.

**Architecture:**
```
train_tinker.py → Tinker train.main(cfg) → do_group_rollout() → BrowserEnv
     ↕ Tinker API ↕
sample() ←→ Qwen3-VL-30B-A3B (hosted by Tinker)
forward_backward() + optim_step() → LoRA weight updates (hosted by Tinker)
```

**Key inversion:** In eval mode, HALO calls the model (via vLLM). In training mode, Tinker calls HALO via `BrowserEnv.step()`.

**Training entry point:**
```bash
# Prerequisites: pip install tinker tinker-cookbook && export TINKER_API_KEY=...
python scripts/train_tinker.py --tasks v2.omnizon-1 --num-envs 8 --loss-fn importance_sampling
```

**Supported loss functions:** `importance_sampling` (default), `ppo`, `cispo`, `dro`

### GRPO via Tinker

Group Relative Policy Optimization where:
1. `BrowserGroupBuilder` creates N=8 `BrowserEnv` instances for the same task
2. Tinker runs parallel rollouts, each env interacts with the model
3. Dense rewards from `progress.py` + `DenseRewardCalculator` provide per-step signal
4. Tinker computes group-relative advantages and updates LoRA weights
5. No local GPU needed — Tinker hosts the model remotely

**Critical config:**
- `--loss-fn importance_sampling` - Works well for browser tasks with high reward variance
- `--num-envs 8` - GRPO group size; need enough samples for meaningful advantage
- `--lora-rank 32` - LoRA adapter rank for weight updates

### MCTS-Guided Exploration (Agent Q Style)

Monte Carlo Tree Search to systematically explore the action space:
1. At each state, sample K candidate actions from policy
2. Use UCB1 to balance exploration vs exploitation
3. AI self-critique ranks action quality (process supervision)
4. Backpropagate values through the tree
5. Use best paths for policy updates

**Key insight:** MCTS finds successful trajectories even when random exploration fails. The tree structure provides credit assignment that sparse rewards cannot.

**From Agent Q paper:**
- Base model at 0% success → MCTS finds paths → 81.7% after training
- Uses step-level DPO on MCTS-generated preferences
- Composite Q-values: `Q = α·Q_mcts + (1-α)·Q_critic`

## Supported Modes

| Mode | Description |
|------|-------------|
| `gpt5_baseline` | GPT-5.2 for comparison / MCTS critic (best vision model) |
| `qwen3vl_base` | Qwen3-VL-30B-A3B base (before training) |
| `qwen3vl_grpo` | Qwen3-VL + Online GRPO LoRA |
| `qwen3vl_mcts` | Qwen3-VL + MCTS-trained LoRA (Agent Q style) |

**LoRA Adapter Paths:**
```
checkpoints/
├── qwen3vl_grpo_lora/   # Online GRPO trained
└── qwen3vl_mcts_lora/   # MCTS (Agent Q) trained
```

**Key Files for Modes:**
- `src/halo/sdk/harness.py` - SUPPORTED_MODES, HaloAgentArgs
- `src/halo/sdk/agent.py` - VALID_MODES, mode handling
- `src/halo/agent/orchestrator.py` - OrchestratorConfig, mode routing

## Technical Constraints

| Constraint | Value | Why |
|------------|-------|-----|
| Model | `Qwen/Qwen3-VL-30B-A3B-Instruct` | MoE VLM (30B total, 3B active) with GUI agent capability |
| SDK Version | `agisdk==0.3.5` | Pinned for REAL benchmark compatibility |
| Task Version | v2 | All tasks use `task_version="v2"` |
| Observation | **Screenshots** + AXTree | VLM sees images directly |
| Browser | 1280x720 headless Chromium | Fixed resolution |
| Action Format | JSON with coordinates or element IDs | Model outputs click (x, y) or element refs |

## Dense Reward Signals

The agent gets rewards for partial progress, not just task completion. These are defined in `src/halo/rl/progress.py`:

**GoMail (Email):**
- Compose window opened: +0.3
- Recipient field filled: +0.2
- Subject filled: +0.2
- Send button visible: +0.2
- Message sent confirmation: +0.5

**GoCalendar (Scheduling):**
- Create event clicked: +0.3
- Title entered: +0.2
- Date/time set: +0.2
- Save clicked: +0.2
- Event created confirmation: +0.5

**Omnizon (Shopping):**
- Search results loaded: +0.2
- Product page reached: +0.3
- Add to cart clicked: +0.3
- Cart page loaded: +0.3
- Checkout initiated: +0.4

**Additional shaping:**
- Valid JSON action: +0.1
- New state reached: +0.2
- Loop detected: -1.0

## File Structure

```
HALO-Agent/
├── src/halo/
│   ├── constants.py          # DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
│   ├── sdk/
│   │   ├── harness.py        # SUPPORTED_MODES, HaloAgentArgs
│   │   └── agent.py          # VALID_MODES, mode handling
│   ├── agent/
│   │   └── orchestrator.py   # OrchestratorConfig, mode routing
│   ├── tinker/               # *** Tinker API integration (online RL) ***
│   │   ├── __init__.py       # Package exports
│   │   ├── browser_env.py    # Tinker Env ABC wrapping REAL benchmark
│   │   ├── group_builder.py  # GRPO group builder (N envs per task)
│   │   ├── dataset.py        # RLDataset/RLDatasetBuilder for task batching
│   │   └── action_parser.py  # Parse/validate model output → browser actions
│   ├── rl/
│   │   ├── online_grpo.py    # Online GRPO trainer (local vLLM, legacy)
│   │   ├── mcts.py           # MCTS exploration
│   │   └── progress.py       # Dense reward functions
│   ├── policy/
│   │   ├── vllm_client.py    # vLLM API client (eval inference)
│   │   └── action_parser.py  # Parse model output to actions
│   └── env/browser_env.py    # REAL benchmark wrapper (eval)
├── scripts/
│   ├── train_tinker.py       # *** Tinker RL training entry point ***
│   ├── serve_policy.py       # Start vLLM server (eval)
│   ├── train_online_grpo.py  # Online GRPO training loop (local vLLM, legacy)
│   ├── train_mcts.py         # MCTS training loop
│   ├── smoke_test.py         # Environment verification
│   └── eval.py               # Evaluation
├── checkpoints/
│   ├── qwen3vl_grpo_lora/    # GRPO-trained adapter
│   └── qwen3vl_mcts_lora/    # MCTS-trained adapter
└── configs/
    └── rl_config.yaml        # Hyperparameters
```

## Common Tasks

**"Help me train with Tinker"**
→ Use `scripts/train_tinker.py`. Configure tasks, loss function, reward weights. Tinker handles the training loop via `tinker_cookbook.rl.train.main()`. The browser environment is in `src/halo/tinker/browser_env.py`.

**"Help me implement online GRPO (local)"**
→ Work on `src/halo/rl/online_grpo.py`. Legacy local-vLLM approach. Core loop: sample actions → execute in env → compute group advantages → policy gradient update.

**"Help me implement MCTS exploration"**
→ Work on `src/halo/rl/mcts.py`. Implement UCB1 selection, tree expansion, AI critique for value estimation, backpropagation.

**"The agent isn't learning"**
→ Check reward variance. Print rewards per group - if all similar, increase `num_generations` or add more reward shaping. Check `frac_reward_zero_std` metric.

**"Rewards are always zero"**
→ Debug `progress.py`. Add print statements to see which conditions are triggering. Check if URL/AXTree parsing is working.

**"Actions are malformed"**
→ This is an RL problem, not a BC problem. Add reward shaping: +0.1 for valid JSON, -0.5 for invalid. The policy will learn valid formats through gradient signal.

## Do NOT

1. **Do NOT suggest BC/SFT as a prerequisite** - We're doing online RL
2. **Do NOT use offline GRPO** - We need live environment interaction
3. **Do NOT train on pre-collected datasets** - The agent collects its own data
4. **Do NOT skip reward shaping** - Sparse rewards won't work, dense signals are essential

## Architecture

### Training: Tinker API (Primary)

Qwen3-VL-30B-A3B is hosted remotely by the Tinker API. We provide the browser environment; Tinker drives inference and gradient updates.

```
Tinker train loop → sample(prompt) → Qwen3-VL-30B-A3B (remote)
                  → BrowserEnv.step(action_tokens) → Playwright browser (local)
                  → forward_backward() + optim_step() → LoRA updates (remote)
```

**Key design decisions:**
- **No Orchestrator recovery policies in training** — recovery masks mistakes, reducing gradient signal. Model learns to handle loops/errors via reward shaping.
- **Fresh observation per step** — each `step()` builds a new ModelInput with current screenshot + goal + last 5 actions. Keeps tokens bounded at ~3-5K per step across 70-step episodes.
- **Async-sync bridge** — `ThreadPoolExecutor` + `run_in_executor()` bridges Tinker's async rollout with Playwright's synchronous API. Max 8 concurrent browsers.

**Prerequisites:**
```bash
pip install tinker tinker-cookbook
export TINKER_API_KEY=your_key_here  # Sign up at https://auth.thinkingmachines.ai/sign-up
```

### Evaluation: vLLM (Local Inference)

For eval modes, vLLM serves the model locally as an OpenAI-compatible API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": f"Goal: {goal}\nWhat action should I take?"}
        ]
    }],
    n=8,
    temperature=0.7
)
```

## Hardware Setup

### Training (Tinker API — no local GPU required)

Tinker hosts the model remotely. You only need a machine to run browsers:

| Component | Runs On | Details |
|-----------|---------|---------|
| Browser (Playwright) | Local / TensorDock | Headless Chromium, REAL benchmark |
| Model + Training | Tinker API (remote) | Inference + LoRA gradient updates |

### Evaluation (vLLM — requires GPU)

| Component | Runs On | Details |
|-----------|---------|---------|
| Browser (Playwright) | TensorDock | Headless Chromium |
| vLLM Server | TensorDock | Policy inference on GPU |

**Recommended TensorDock Config for Eval:**

| GPU | VRAM | Notes |
|-----|------|-------|
| H100 SXM5 | 80GB | Best performance |
| A100 SXM4 | 80GB | Good balance |

## Key Hyperparameters

```yaml
# Tinker Online GRPO (primary)
num_envs: 8               # GRPO group size
loss_fn: importance_sampling  # RL loss function
learning_rate: 1e-6        # Small for stability
lora_rank: 32              # LoRA adapter rank
temperature: 0.7           # Sampling temperature
max_tokens: 300            # Max tokens per model response
max_steps: 70              # Max steps per episode

# Reward shaping
progress_weight: 1.0       # Dense progress reward
novelty_bonus: 0.2         # New state bonus
loop_penalty: -0.5         # Loop detection penalty
action_error_penalty: -0.2 # Invalid action penalty
success_bonus: 1.0         # Task completion bonus

# MCTS
num_simulations: 50       # Tree search depth
ucb_constant: 1.414       # Exploration coefficient
max_children: 5           # Actions per node
```

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Mean episode reward | Increasing trend | Log per episode |
| Success rate | >30% on easy tasks | Eval every 100 steps |
| Action validity | >90% valid JSON | Parse check |
| Reward variance | >0 per group | `frac_reward_zero_std < 0.5` |

## Verified Technical Facts

**Qwen3-VL-30B-A3B:** MoE Vision-Language Model (30B total, 3B active). Native GUI agent capability. Supports 2D grounding (click coordinates). Supported by Tinker API for distributed online RL.

**Tinker API:** Distributed RL training API by Thinking Machines Lab. Primitives: `sample()`, `forward_backward()`, `optim_step()`. Hosts the model remotely — no local GPU needed for training. Supports LoRA fine-tuning.

**GRPO zero-variance problem:** Real. When all rewards in a group are identical, advantages are zero, gradients are zero. Tinker's `importance_sampling` loss handles this better than standard GRPO for high-variance browser tasks.

**Agent Q results:** LLaMA-3 70B went from 18.6% → 81.7% on OpenTable using MCTS + step-level DPO. MCTS with 95.4% at inference time.

**REAL benchmark SOTA:** 41.1% (Claude-3.7-Sonnet-Thinking). Our target is >30% with online RL on Qwen3-VL-30B-A3B.

**vLLM VLM support:** Requires `vllm>=0.11.0`. Used for local evaluation inference only (training uses Tinker API).
