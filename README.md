# HALO-Agent

**Worker+Manager Browser Agent for AGI Inc REAL Bench**

HALO-Agent is a hierarchical browser automation agent designed for AGI Inc's REAL Bench evaluation suite. It implements a worker-manager architecture with verified action caching and macro skill replay for efficient task execution.

## Overview

HALO-Agent combines multiple decision-making strategies in a hierarchical pipeline:

1. **Macro Replay Cache** - Reuse learned skill sequences
2. **Verified Action Cache (VAC)** - Cache verified state-action pairs
3. **Worker Policy** - Fast, lightweight decision-making
4. **Manager Policy** - High-stakes oversight and error recovery

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Install Playwright browsers
playwright install --force

# Copy environment template
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run Smoke Test

```bash
python scripts/smoke_test.py
```

### Run Evaluation

```bash
python scripts/eval_subset.py --tasks 20
```

## Project Structure

```
halo-agent/
├── src/halo/          # Core agent implementation
│   ├── agent/         # Orchestrator and routing logic
│   ├── policy/        # Worker and manager policies
│   ├── cache/         # VAC and macro cache
│   ├── verify/        # Postcondition verification
│   ├── obs/           # Observation processing
│   ├── logging/       # Structured JSONL logging
│   └── sdk/           # AGI SDK wrappers
├── scripts/           # Evaluation and collection scripts
├── data/              # Trajectories and cached data
└── results/           # Evaluation results
```

## Requirements

- Python 3.10+
- AGI SDK 0.3.5 (pinned)
- Playwright
- Anthropic API key

## Important Notes

- **Always use `task_version="v2"`** when running tasks with AGI SDK
- The SDK defaults to v1 if version is omitted (this is critical to avoid)
- See `CLAUDE.md` for detailed project constitution and development guidelines

## Development

See `docs/RUNBOOK.md` for detailed setup and development notes.

## License

Internal project for AGI Inc.
