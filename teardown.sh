#!/bin/bash
# Teardown Script - Run before stopping TensorDock instance
# Saves checkpoints and stops services gracefully

echo "=========================================="
echo "HALO-Agent Teardown"
echo "=========================================="

# Stop vLLM if running
echo "Stopping vLLM server..."
pkill -f "vllm serve" 2>/dev/null || echo "vLLM not running"

# List checkpoints
echo ""
echo "Checkpoints saved:"
if [ -d "checkpoints" ]; then
    find checkpoints -type f -name "*.pt" -o -name "*.json" -o -name "*.safetensors" | head -20
    echo ""
    du -sh checkpoints/*/
else
    echo "No checkpoints directory found"
fi

# List logs
echo ""
echo "Recent logs:"
if [ -d "outputs/logs" ]; then
    ls -lt outputs/logs/ | head -10
fi

# Git status
echo ""
echo "Git status:"
git status --short 2>/dev/null || echo "Not a git repo"

echo ""
echo "=========================================="
echo "Safe to stop instance"
echo "=========================================="
echo ""
echo "To save your work:"
echo "  git add -A && git commit -m 'training checkpoint' && git push"
echo ""
echo "Or copy checkpoints locally:"
echo "  scp -r user@ip:~/HALO-Agent/checkpoints ./local_checkpoints"
echo ""
