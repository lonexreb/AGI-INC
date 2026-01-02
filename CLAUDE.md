# CLAUDE.md - Project Context for AI Assistants

## What We're Building

HALO-Agent is a browser automation agent for AGI Inc's REAL Benchmark. We're training **Qwen3-VL-8B** (a Vision-Language Model) using **Online Reinforcement Learning** to complete web tasks autonomously.

**Why Qwen3-VL?**
- Built-in "Visual Agent" capability for GUI control
- 2D grounding (outputs click coordinates directly)
- Official [Computer-Use Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb)
- 256K native context for long browsing sessions

## The Goal: Online RL

We are doing **Online RL**, not offline imitation learning. This means:

- The agent interacts with the live environment
- It receives rewards based on its actions
- It updates its policy in real-time
- No pre-collected demonstration datasets required

**Do NOT suggest Behavioral Cloning (BC) or offline approaches as prerequisites.** The whole point is to learn from scratch through environment interaction.

## Two RL Approaches We're Implementing

### 1. Online GRPO with Dense Reward Shaping

Group Relative Policy Optimization where the agent:
1. Observes current browser state
2. Samples N candidate actions (the "group")
3. Executes each action, collects rewards
4. Computes advantages relative to group mean
5. Updates policy to favor higher-reward actions

**Key insight:** Dense rewards from `progress.py` provide learning signal even when tasks aren't fully completed. We don't need sparse "success/fail" - we reward partial progress.

**Critical config:**
- `loss_type="dr_grpo"` - Removes std normalization, prevents zero-gradient when rewards are similar
- `num_generations >= 8` - Need enough samples for meaningful group comparison
- Monitor `frac_reward_zero_std` - If >50%, rewards lack variance, learning stalls

### 2. MCTS-Guided Exploration (Agent Q Style)

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
| `gpt4o_baseline` | GPT-4o for comparison / MCTS critic |
| `qwen3vl_base` | Qwen3-VL-8B base (before training) |
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
| Model | `Qwen/Qwen3-VL-8B-Instruct` | VLM with GUI agent capability |
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
│   ├── sdk/
│   │   ├── harness.py        # SUPPORTED_MODES, HaloAgentArgs
│   │   └── agent.py          # VALID_MODES, mode handling
│   ├── agent/
│   │   └── orchestrator.py   # OrchestratorConfig, mode routing
│   ├── rl/
│   │   ├── online_grpo.py    # Online GRPO trainer
│   │   ├── mcts.py           # MCTS exploration
│   │   └── progress.py       # Dense reward functions
│   ├── policy/
│   │   ├── vllm_client.py    # vLLM API client (VLM inference)
│   │   └── action_parser.py  # Parse model output to actions
│   └── env/browser_env.py    # REAL benchmark wrapper
├── scripts/
│   ├── serve_policy.py       # Start vLLM server
│   ├── train_online_grpo.py  # Online GRPO training loop
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

**"Help me implement online GRPO"**
→ Work on `src/halo/rl/online_grpo.py`. Core loop: sample actions → execute in env → compute group advantages → policy gradient update.

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

## Architecture: vLLM + Vision-Language Model

Qwen3-VL-8B processes **screenshots directly** — no need to convert to text. The model sees the browser and outputs actions.

```
Input: Screenshot (PNG) + Goal text
Output: Action JSON with coordinates or element references
```

Online RL requires fast, repeated inference. vLLM makes this practical:

```
Standard transformers:
  8 action samples × 3 sec each = 24 seconds per step
  = Unusable for RL

vLLM:
  8 action samples batched = 2-3 seconds per step
  = Practical for RL
```

**vLLM serves the VLM as an API:**

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Encode screenshot
with open("screenshot.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": f"Goal: {goal}\nWhat action should I take? Respond with JSON."}
        ]
    }],
    n=8,  # Sample 8 actions
    temperature=0.7
)
```

**For LoRA adapters during training:**
vLLM supports hot-reloading LoRA weights. The training loop:
1. Runs episodes, collects rewards
2. Computes policy gradient
3. Updates LoRA weights
4. Reloads weights in vLLM
5. Repeat

## Hardware Setup: TensorDock (Single Machine)

Everything runs on TensorDock - browser, vLLM, and training. No split setup needed.

| Component | Runs On | Details |
|-----------|---------|---------|
| Browser (Playwright) | TensorDock | Headless Chromium, REAL benchmark |
| vLLM Server | TensorDock | Policy inference on GPU |
| Training Loop | TensorDock | LoRA weight updates on GPU |

**Recommended TensorDock Config:**

| GPU | VRAM | Price | Notes |
|-----|------|-------|-------|
| H100 SXM5 | 80GB | ~$2.25/hr | Best performance, fits 70B models |
| A100 SXM4 | 80GB | ~$1.50/hr | Good balance of cost/performance |
| A100 PCIe | 40GB | ~$1.00/hr | Budget option, fits up to 13B |

For Qwen2.5-3B, even A100-40GB is sufficient. H100 gives faster inference.

**TensorDock Setup:**
```bash
# 1. Spin up instance (Ubuntu 22.04, H100 or A100)
# 2. SSH in and run setup script
# 3. Start vLLM server
# 4. Run training loop - everything is localhost
python scripts/train_online_grpo.py --policy_url http://localhost:8000
```

## Key Hyperparameters

```yaml
# Online GRPO
num_generations: 8        # Actions per state
temperature: 0.7          # Exploration vs exploitation
learning_rate: 1e-6       # Small for stability
loss_type: dr_grpo        # Handles low reward variance
kl_coef: 0.001           # Regularization

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

**Qwen3-VL-8B:** Vision-Language Model with native GUI agent capability. Has official Computer-Use cookbook. Supports 2D grounding (click coordinates). ~20GB VRAM for inference, fits on H100 with room for training.

**GRPO zero-variance problem:** Real. When all rewards in a group are identical, advantages are zero, gradients are zero. Mitigate with `loss_type="dr_grpo"` (mean-centering only, no std normalization).

**Agent Q results:** LLaMA-3 70B went from 18.6% → 81.7% on OpenTable using MCTS + step-level DPO. MCTS with 95.4% at inference time.

**REAL benchmark SOTA:** 41.1% (Claude-3.7-Sonnet-Thinking). Our target is >30% with an 8B VLM.

**vLLM VLM support:** Requires `vllm>=0.11.0`. Supports image inputs via OpenAI-compatible API.
