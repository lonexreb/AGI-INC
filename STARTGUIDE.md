# STARTGUIDE: Online RL Training with Qwen3-VL on TensorDock

This guide walks you through training a **Vision-Language browser agent** using Online Reinforcement Learning on TensorDock GPUs.

**Model:** Qwen3-VL-8B-Instruct — a VLM that sees screenshots and outputs actions directly.

Everything runs on TensorDock - browser environment, vLLM inference, and training. No local machine needed.

---

## Step 1: Spin Up TensorDock Instance

1. Go to [tensordock.com](https://tensordock.com)
2. Select GPU:
   - **H100 SXM5 80GB** (~$2.25/hr) - Fastest
   - **A100 SXM4 80GB** (~$1.50/hr) - Good balance
   - **A100 PCIe 40GB** (~$1.00/hr) - Budget option
3. OS: **Ubuntu 22.04 LTS**
4. Storage: **200GB** minimum
5. Open ports: **22** (SSH), **8000** (vLLM API)

**Note:** TensorDock has no live availability indicator. If "No available nodes," try different region or GPU tier.

---

## Step 2: Server Setup

SSH into your instance and run:

```bash
# System dependencies
sudo apt update && sudo apt install -y git python3-pip python3-venv wget curl

# Install system libs for Playwright (headless browser)
sudo apt install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2

# Create environment
python3 -m venv venv
source venv/bin/activate

# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# vLLM for fast VLM inference (requires >= 0.11.0 for Qwen3-VL)
pip install "vllm>=0.11.0"

# Qwen VL utilities for image processing
pip install qwen-vl-utils==0.0.14

# Training dependencies
pip install transformers>=4.57.0 accelerate peft bitsandbytes

# Environment dependencies
pip install agisdk==0.3.5 playwright openai python-dotenv numpy httpx tqdm pyyaml

# Install browser
playwright install chromium

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Step 3: Clone Repo and Configure

```bash
git clone <your-repo-url> HALO-Agent
cd HALO-Agent

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-...
AGI_API_KEY=...
EOF

# Verify setup
python scripts/smoke_test.py
```

---

## Step 4: Start vLLM Server

In a terminal (or tmux session):

```bash
source venv/bin/activate

# Start Qwen3-VL-8B with vLLM
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --enable-lora \
    --max-model-len 8192

# Wait for "Uvicorn running on http://0.0.0.0:8000"
```

Verify it's running:
```bash
curl http://localhost:8000/v1/models
```

**Test with an image:**
```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Test with a simple image
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}}
        ]
    }],
    max_tokens=100
)
print(response.choices[0].message.content)
```

---

## Step 5: Understand Dense Rewards

Online RL needs reward signal. We use dense shaping - partial credit for progress:

```
GoMail task "Send email to X":
  +0.3  Compose window opened
  +0.2  Recipient field filled
  +0.2  Subject entered
  +0.2  Send button clicked
  +0.5  "Message sent" confirmation
  +10.0 Task marked success
```

This is defined in `src/halo/rl/progress.py`. Without dense rewards, the agent gets zero signal until task completion (which may never happen early in training).

---

## Step 6: Run Online RL Training

### Option A: Online GRPO

```bash
# In a new terminal (keep vLLM running)
source venv/bin/activate
cd HALO-Agent

python scripts/train_online_grpo.py \
    --policy_url http://localhost:8000 \
    --domain gomail \
    --num_generations 8 \
    --episodes 100 \
    --lr 1e-6 \
    --loss_type dr_grpo
```

How it works:
1. Agent observes browser state (screenshot + goal)
2. Samples 8 candidate actions from policy (via vLLM on localhost)
3. Executes each, collects rewards
4. Computes advantage: `A_i = r_i - mean(r)` (dr_grpo)
5. Policy gradient update
6. Repeat

**Critical:** Use `loss_type=dr_grpo` to handle low reward variance. Standard GRPO divides by std(r), which explodes when rewards are similar.

### Option B: MCTS Exploration (Agent Q Style)

```bash
python scripts/train_mcts.py \
    --policy_url http://localhost:8000 \
    --domain gomail \
    --simulations 50 \
    --ucb_c 1.414 \
    --episodes 100
```

How it works:
1. Build a search tree over browser states
2. At each node, sample K actions from policy (via vLLM)
3. Select action via UCB1: `Q(s,a) + C * sqrt(ln(N) / n(s,a))`
4. Expand tree by executing action, observing new state
5. AI self-critique estimates value of new state
6. Backpropagate values up the tree
7. After N simulations, extract best path
8. Train policy on (state, best_action) pairs

**Key insight:** MCTS finds successful paths through systematic search, even when random sampling fails. Agent Q achieved 95.4% on OpenTable this way.

---

## Step 7: Monitor Training

Key metrics to watch:

| Metric | Good | Bad | Fix |
|--------|------|-----|-----|
| `mean_reward` | Increasing | Flat/decreasing | Check reward shaping |
| `frac_reward_zero_std` | <0.3 | >0.5 | Increase `num_generations` |
| `action_validity` | >0.9 | <0.5 | Add validity reward shaping |
| `episode_length` | Varies | Always max | Agent is stuck, check loop penalty |

---

## Step 8: LoRA Weight Updates

During training, the loop:
1. Runs episodes via vLLM inference (localhost:8000)
2. Collects transitions and rewards
3. Computes policy gradient
4. Saves LoRA adapter checkpoint
5. Hot-reloads weights into vLLM
6. Repeat

vLLM supports dynamic LoRA loading:
```bash
# Training script saves checkpoint, then reloads:
curl -X POST http://localhost:8000/v1/load_lora \
    -d '{"lora_name": "policy", "lora_path": "./outputs/lora_checkpoint"}'
```

---

## Troubleshooting

**"Rewards are always zero"**

The dense reward conditions in `progress.py` aren't triggering. Debug:
```python
# Add to progress.py
print(f"URL: {url}")
print(f"AXTree contains 'compose': {'compose' in axtree}")
```

**"GRPO gradients are zero"**

All rewards in the group are identical. Solutions:
1. Use `loss_type="dr_grpo"` (no std normalization)
2. Increase `num_generations` to 16
3. Add more diverse reward signals

**"Agent repeats same action"**

Loop penalty isn't strong enough. Increase from -1.0 to -2.0 in reward config. Or add state visitation penalty.

**"Actions are malformed JSON"**

Add reward shaping: +0.1 for valid JSON, -0.5 for parse errors. The policy will learn correct format through gradient signal.

**"Playwright fails to launch"**

Missing system dependencies. Run:
```bash
sudo apt install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2
playwright install chromium
```

**"CUDA out of memory"**

Reduce vLLM memory utilization:
```bash
--gpu-memory-utilization 0.5
```
Or use smaller batch for training.

**"TensorDock instance unavailable"**

No live availability indicator - this is common. Try:
1. Different GPU tier (A100 instead of H100)
2. Different region
3. Wait 30 min and retry

---

## Cost Estimate (TensorDock)

| GPU | Hourly | 10hr Training | Notes |
|-----|--------|---------------|-------|
| H100 SXM5 | $2.25 | $22.50 | Fastest, recommended |
| A100 80GB | $1.50 | $15.00 | Good balance |
| A100 40GB | $1.00 | $10.00 | Budget option |

Plus OpenAI API costs if using GPT-4o for AI critic in MCTS (~$0.01-0.05 per episode).

---

## Expected Timeline

| Stage | Episodes | Expected Success Rate |
|-------|----------|----------------------|
| Initial | 0-50 | ~5% (random exploration) |
| Early learning | 50-200 | ~15% (finds some patterns) |
| Mid training | 200-500 | ~25% (consistent progress) |
| Converged | 500+ | ~35% (approaching SOTA) |

Current SOTA is 41%. An 8B VLM hitting 30%+ is a strong result.

---

## Reference

- [Qwen3-VL Computer-Use Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/computer_use.ipynb) - Official guide for GUI control
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL) - Model documentation
