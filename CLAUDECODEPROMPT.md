# Claude Code Prompt: HALO-Agent Online RL Refactor

Copy and paste this entire prompt into Claude Code to begin the refactor.

---

## Context

I'm building HALO-Agent, a browser automation agent for AGI Inc's REAL Benchmark. The goal is to train **Qwen3-VL-8B** (a Vision-Language Model) using **Online Reinforcement Learning** (not offline/BC/SFT).

**Why Qwen3-VL?** It has built-in Visual Agent capability for GUI control, 2D grounding (outputs click coordinates), and an official [Computer-Use Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb).

The current codebase has accumulated complexity from exploring different approaches. We need to:
1. Audit the codebase and identify files to remove
2. Simplify to focus purely on Online RL with VLM
3. Implement the core Online RL components

## Phase 1: Codebase Audit

First, inspect the repository structure and identify what to keep vs remove.

**Run these commands to understand the current state:**

```bash
# Show directory structure
find . -type f -name "*.py" | head -50

# Check src/halo structure
ls -la src/halo/

# Check scripts
ls -la scripts/

# Look for BC/SFT/offline training code (candidates for removal)
grep -r "behavioral_cloning\|sft\|offline" --include="*.py" -l

# Look for trajectory collection scripts (may not need)
grep -r "collect_traj\|expert_traj" --include="*.py" -l

# Find the core files we need to keep
ls -la src/halo/policy/
ls -la src/halo/agent/
```

**After inspecting, categorize files:**

### KEEP (Core infrastructure)
- `src/halo/agent/orchestrator.py` - Environment interaction (may need simplification)
- `src/halo/policy/` - Policy interfaces (will add vllm_client.py)
- `src/halo/sdk/` - REAL benchmark wrapper
- `configs/` - Configuration files

### LIKELY REMOVE (Offline/BC focused)
- Any `train_bc*.py` scripts
- Any `train_sft*.py` scripts
- Any `collect_*_traj.py` scripts (we don't pre-collect in online RL)
- Any `offline_grpo*.py` (we want online, not offline)
- Overly complex caching logic if it complicates the RL loop

### SIMPLIFY
- `orchestrator.py` if it has too many fallback layers
- Remove manager/worker hierarchy if it complicates things - we just need policy → action → reward

**Tell me what you find before making any changes.**

---

## Phase 2: Implement Online RL Core

After cleanup, implement these files. This is the minimal set needed for Online RL.

### File 1: `src/halo/policy/vllm_client.py`

Purpose: Clean wrapper for vLLM's OpenAI-compatible API. Handles batched action sampling with **image inputs**.

Requirements:
- Connect to vLLM server at configurable URL
- `sample_actions(screenshot: bytes, goal: str, n: int, temperature: float) -> List[str]` - Sample n actions from VLM
- `sample_single(screenshot: bytes, goal: str) -> str` - For evaluation (greedy)
- Handle image encoding (base64)
- Handle connection errors gracefully
- Use chat completion format (Qwen3-VL uses chat)

```python
# Example interface
class VLLMPolicyClient:
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen/Qwen3-VL-8B-Instruct"):
        ...

    def sample_actions(self, screenshot: bytes, goal: str, n: int = 8, temperature: float = 0.7) -> List[str]:
        """Sample n actions from VLM given screenshot and goal."""
        # Encode screenshot to base64
        # Send as image_url in chat completion
        # Return list of action strings
        ...

    def sample_single(self, screenshot: bytes, goal: str, temperature: float = 0.0) -> str:
        """Sample single action (for eval)."""
        ...
```

### File 2: `src/halo/rl/progress.py`

Purpose: Dense reward functions that provide learning signal even without task completion.

Requirements:
- `compute_reward(obs, action, next_obs, task_info, done, success) -> float`
- Task-specific progress detection:
  - GoMail: compose opened, recipient filled, subject filled, send clicked, sent confirmation
  - GoCalendar: create clicked, title entered, date set, save clicked, created confirmation
  - Omnizon: search results, product page, add to cart, cart page, checkout
  - Zilloft: search, filters applied, listing viewed
- General shaping:
  - +0.1 for valid JSON action
  - +0.2 for reaching new state (not seen before in episode)
  - -1.0 for loop detection (repeated action)
- Parse URL and AXTree to detect progress

```python
# Example interface
class DenseRewardCalculator:
    def __init__(self, config: dict = None):
        self.seen_states = set()
        self.action_history = []

    def reset(self):
        """Call at episode start."""
        ...

    def compute_reward(self, obs, action, next_obs, task_info, done, success) -> float:
        """Compute dense reward for a transition."""
        ...
```

### File 3: `src/halo/rl/online_grpo.py`

Purpose: Online GRPO training loop. This is the core RL algorithm.

Requirements:
- Main loop: observe → sample N actions → execute → compute rewards → compute advantages → update policy
- Use `loss_type="dr_grpo"` (mean-centering only, no std normalization) to handle low variance
- Advantages: `A_i = r_i - mean(r)` (dr_grpo removes the /std(r))
- Interface with vLLM for inference, PyTorch for gradient updates
- Support LoRA weight updates and hot-reload to vLLM
- Log: mean_reward, reward_std, frac_reward_zero_std, episode_length, success_rate

```python
# Example interface
class OnlineGRPOTrainer:
    def __init__(
        self,
        policy_client: VLLMPolicyClient,
        env: BrowserEnv,
        reward_calculator: DenseRewardCalculator,
        config: dict
    ):
        ...

    def collect_episode(self) -> List[Transition]:
        """Run one episode, collect transitions with rewards."""
        ...

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """Compute group-relative advantages (dr_grpo style)."""
        ...

    def update_policy(self, transitions: List[Transition]):
        """Compute gradients and update LoRA weights."""
        ...

    def train(self, num_episodes: int):
        """Main training loop."""
        ...
```

### File 4: `src/halo/rl/mcts.py`

Purpose: MCTS exploration for finding successful trajectories (Agent Q style).

Requirements:
- Tree structure: nodes are states, edges are actions
- UCB1 selection: `Q(s,a) + C * sqrt(ln(N) / n(s,a))`
- Expansion: sample K actions from policy at leaf nodes
- Simulation: use AI self-critique to estimate state value (can use same LLM)
- Backpropagation: update Q-values up the tree
- Extract best path after N simulations
- Can generate training data for policy updates

```python
# Example interface
class MCTSNode:
    state_hash: str
    observation: str
    children: Dict[str, 'MCTSNode']  # action -> child node
    visits: int
    value: float

class MCTSExplorer:
    def __init__(
        self,
        policy_client: VLLMPolicyClient,
        env: BrowserEnv,
        config: dict
    ):
        self.ucb_c = config.get("ucb_c", 1.414)
        self.num_simulations = config.get("num_simulations", 50)
        ...

    def search(self, root_observation: str) -> List[Tuple[str, str]]:
        """Run MCTS from root, return best path as [(obs, action), ...]"""
        ...

    def ucb1_select(self, node: MCTSNode) -> str:
        """Select action using UCB1."""
        ...

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expand leaf node with new actions."""
        ...

    def estimate_value(self, observation: str) -> float:
        """Use AI self-critique to estimate state value."""
        ...
```

### File 5: `scripts/train_online_grpo.py`

Purpose: Entry point for Online GRPO training.

Requirements:
- Parse command line args (policy_url, domain, num_episodes, etc.)
- Initialize vLLM client, environment, reward calculator
- Create OnlineGRPOTrainer and run training
- Save checkpoints, log metrics

```python
# Example usage
# python scripts/train_online_grpo.py \
#     --policy_url http://localhost:8000 \
#     --domain gomail \
#     --num_generations 8 \
#     --episodes 100 \
#     --lr 1e-6
```

### File 6: `scripts/train_mcts.py`

Purpose: Entry point for MCTS-based training.

Similar structure to train_online_grpo.py but uses MCTSExplorer.

---

## Implementation Order

1. First, complete Phase 1 audit and tell me what you find
2. Then implement in this order:
   - `vllm_client.py` (needed by everything else)
   - `progress.py` (dense rewards)
   - `online_grpo.py` (main training loop)
   - `train_online_grpo.py` (entry point)
   - `mcts.py` and `train_mcts.py` (second approach)

## Constraints

- **Model:** `Qwen/Qwen3-VL-8B-Instruct` (VLM with GUI agent capability)
- SDK: `agisdk==0.3.5` (pinned)
- Observation: **Screenshots** (VLM sees images directly) + optionally AXTree for rewards
- Browser: 1280x720 headless Chromium
- Action format: JSON with click coordinates (x, y) or element IDs
- **Hardware: TensorDock H100/A100** - Everything runs on same machine (browser, vLLM, training)
- **vLLM:** Requires `vllm>=0.11.0` for Qwen3-VL support

## Do NOT

- Do NOT add BC/SFT as prerequisites
- Do NOT create offline data collection scripts
- Do NOT over-engineer with complex hierarchies
- Do NOT add features we didn't ask for

Keep it simple. We want the minimal code needed to do Online RL on browser tasks.

---

**Start by running the audit commands and reporting what you find in the codebase.**
