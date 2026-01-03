#!/bin/bash
# ============================================
# HALO-Agent Launch Script for TensorDock
# ============================================
# Does EVERYTHING: setup, vLLM, training
#
# Usage:
#   ./launch.sh                          # Full run (setup + train)
#   ./launch.sh --skip-setup             # Skip setup, just train
#   ./launch.sh --task v2.omnizon-1      # Different task
#   ./launch.sh --episodes 20            # More episodes
#
# First run: ~15-20 min (downloads model)
# Subsequent: ~5 min

set -e

# ============================================
# Configuration
# ============================================
TASK="${TASK:-v2.gomail-1}"
EPISODES="${EPISODES:-10}"
SKIP_SETUP=false
VLLM_PORT=8000
MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup) SKIP_SETUP=true; shift ;;
        --task) TASK="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --help)
            echo "Usage: ./launch.sh [options]"
            echo ""
            echo "Options:"
            echo "  --skip-setup      Skip system setup (use if already set up)"
            echo "  --task NAME       Task to train on (default: v2.gomail-1)"
            echo "  --episodes N      Number of episodes (default: 10)"
            echo ""
            echo "Examples:"
            echo "  ./launch.sh                              # First time setup + train"
            echo "  ./launch.sh --skip-setup                 # Already setup, just train"
            echo "  ./launch.sh --task v2.omnizon-1 --episodes 20"
            echo ""
            echo "Cleanup: ./teardown.sh (run before stopping instance)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_step() {
    echo ""
    echo -e "${CYAN}============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${CYAN}============================================${NC}"
}

print_ok() { echo -e "${GREEN}✓${NC} $1"; }
print_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
print_err() { echo -e "${RED}✗${NC} $1"; }

echo -e "${CYAN}==========================================${NC}"
echo -e "${GREEN}HALO-Agent Launch${NC}"
echo -e "${CYAN}==========================================${NC}"
echo "Task: $TASK | Episodes: $EPISODES"

# ============================================
# PART 1: SYSTEM SETUP
# ============================================
if [ "$SKIP_SETUP" = false ]; then

    print_step "STEP 1/10: System Dependencies"
    sudo apt update
    sudo apt install -y git python3-pip python3-venv python3.11 python3.11-venv tmux curl wget
    # Playwright browser deps
    sudo apt install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
        libxfixes3 libxrandr2 libgbm1 libasound2
    print_ok "System dependencies installed"

    print_step "STEP 2/10: Verify GPU"
    if ! command -v nvidia-smi &> /dev/null; then
        print_err "nvidia-smi not found. GPU drivers not installed."
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv
    print_ok "GPU verified"

    print_step "STEP 3/10: Repository"
    if [ ! -f "CLAUDE.md" ]; then
        if [ -n "$REPO_URL" ]; then
            git clone "$REPO_URL" HALO-Agent
            cd HALO-Agent
        else
            print_warn "Not in HALO-Agent directory"
            read -p "Enter repo URL (or press Enter if already cloned): " REPO_URL
            if [ -n "$REPO_URL" ]; then
                git clone "$REPO_URL" HALO-Agent
                cd HALO-Agent
            fi
        fi
    fi
    print_ok "In HALO-Agent directory"

    print_step "STEP 4/10: Python Environment"
    if [ ! -d "venv" ]; then
        python3.11 -m venv venv || python3 -m venv venv
    fi
    source venv/bin/activate
    print_ok "Virtual environment ready"

    print_step "STEP 5/10: PyTorch + CUDA"
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    print_ok "PyTorch installed"

    print_step "STEP 6/10: vLLM"
    pip install "vllm>=0.11.0"
    print_ok "vLLM installed"

    print_step "STEP 7/10: Qwen Dependencies"
    pip install qwen-vl-utils==0.0.14 "transformers>=4.57.0"
    print_ok "Qwen dependencies installed"

    print_step "STEP 8/10: Project Dependencies"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        pip install agisdk==0.3.5 playwright openai python-dotenv numpy httpx Pillow tqdm pyyaml accelerate peft
    fi
    playwright install chromium
    print_ok "Project dependencies installed"

    print_step "STEP 9/10: Directories & Config"
    mkdir -p checkpoints/qwen3vl_grpo_lora checkpoints/qwen3vl_mcts_lora
    mkdir -p outputs/logs data/trajectories results
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# OpenAI API Key (only needed for gpt4o_baseline mode)
OPENAI_API_KEY=

# vLLM Configuration (optional)
# HALO_VLLM_URL=http://localhost:8000/v1
# HALO_WORKER_MODEL=Qwen/Qwen3-VL-8B-Instruct
EOF
    fi
    print_ok "Setup complete"

else
    echo ""
    print_warn "Skipping setup (--skip-setup)"
    source venv/bin/activate
fi

# ============================================
# PART 2: START VLLM SERVER
# ============================================
print_step "STEP 10/10: vLLM Server"

if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
    print_ok "vLLM already running"
else
    print_warn "Starting vLLM server..."
    echo "  Model: $MODEL"
    echo "  First run downloads ~16GB from HuggingFace"

    nohup vllm serve $MODEL \
        --port $VLLM_PORT \
        --gpu-memory-utilization 0.7 \
        --max-model-len 8192 \
        > outputs/logs/vllm.log 2>&1 &

    VLLM_PID=$!
    echo "  PID: $VLLM_PID"
    echo "  Log: outputs/logs/vllm.log"

    # Wait for vLLM (up to 10 min for first download)
    echo "  Waiting for vLLM to be ready..."
    for i in {1..120}; do
        if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
            print_ok "vLLM server ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            print_err "vLLM failed to start after 10 minutes"
            echo "Check: tail -f outputs/logs/vllm.log"
            exit 1
        fi
        sleep 5
        printf "  %d/120 (%d min)\r" $i $((i * 5 / 60))
    done
fi

# ============================================
# PART 3: RUN TRAINING
# ============================================
print_step "Training: $TASK ($EPISODES episodes)"

# Dry run
echo "Running validation..."
python scripts/train_online_grpo.py --dry-run

# Actual training
echo ""
echo "Starting training..."
python scripts/train_online_grpo.py \
    --tasks "$TASK" \
    --episodes "$EPISODES" \
    --num-generations 4

# ============================================
# DONE
# ============================================
print_step "COMPLETE!"
echo ""
echo "Results:     results/grpo_training/"
echo "Checkpoints: checkpoints/qwen3vl_grpo_lora/"
echo "vLLM logs:   outputs/logs/vllm.log"
echo ""
echo "Run more training:"
echo "  ./launch.sh --skip-setup --task v2.omnizon-1 --episodes 20"
echo ""
echo "Before stopping instance:"
echo "  ./teardown.sh"
echo ""
