# HALO-Agent Development Runbook

This document tracks setup steps, configuration notes, and operational procedures for HALO-Agent development.

## Initial Setup

### Requirements

**CRITICAL: Python 3.10+ is required!**

The AGI SDK uses Python `match` statements which require Python 3.10 or later.

```bash
# Check Python version
python3 --version  # Must be 3.10+

# Install Python 3.10+ if needed (macOS with Homebrew)
brew install python@3.11
```

### Environment Setup

```bash
# Create virtual environment (use Python 3.10+)
python3.11 -m venv .venv  # Or python3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install -U pip

# Install AGI SDK
pip install agisdk==0.3.5

# Install Playwright browsers
playwright install --force

# Install dev dependencies
pip install pytest python-dotenv
```

### Environment Variables

Create `.env` from template:
```bash
cp .env.example .env
```

Required variables:
- `OPENAI_API_KEY` - Your OpenAI API key (used by Worker and Manager policies)
- `ANTHROPIC_API_KEY` - Your Anthropic API key (optional, for future use)

---

## Task Discovery

### List Available v2 Tasks

Use `list_v2_tasks.py` to discover all available v2 tasks:

```bash
# List all v2 tasks with summary
python scripts/list_v2_tasks.py

# Export to JSON
python scripts/list_v2_tasks.py --json tasks.json

# Filter by site
python scripts/list_v2_tasks.py --site omnizon gomail

# Get just task IDs (for scripting)
python scripts/list_v2_tasks.py --ids-only
```

**Output example:**
```
============================================================
REAL Bench v2 Task Discovery
============================================================
Source: agisdk package
Total v2 tasks: 121
Sites: 11

Per-site counts:
----------------------------------------
  dashdish: 11 tasks
  gocalendar: 10 tasks
  gomail: 21 tasks
  ...
```

---

## Running Evaluations

### Subset Evaluation

Run evaluation on a subset of tasks:

```bash
# Run baseline_worker on 50 random v2 tasks (default)
python scripts/eval_subset.py --mode baseline_worker

# Run specific tasks
python scripts/eval_subset.py --tasks v2.omnizon-13,v2.gomail-1 --mode hierarchy_vac_macros

# Run with debug output
python scripts/eval_subset.py --mode baseline_worker --debug --max_steps 10

# Run with visible browser
python scripts/eval_subset.py --mode baseline_worker --headless false

# Dry run (see what would execute)
python scripts/eval_subset.py --mode baseline_worker --dry-run
```

**CLI Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--tasks` | (sample 50) | Comma-separated task IDs |
| `--task_type` | None | Filter by site (e.g., omnizon) |
| `--task_version` | v2 | Task version (WARNING: SDK defaults to v1!) |
| `--mode` | baseline_worker | Agent mode(s) to run |
| `--max_steps` | 70 | Max steps per task |
| `--headless` | true | Run headless browser |
| `--debug` | false | Enable debug output |
| `--sample_size` | 50 | Tasks to sample if --tasks not provided |

### Full Matrix Evaluation

Run all modes on all v2 tasks:

```bash
# Run full matrix with all modes
python scripts/eval_full_matrix.py

# Run specific modes
python scripts/eval_full_matrix.py --mode baseline_worker hierarchy_vac_macros

# Run on specific tasks
python scripts/eval_full_matrix.py --tasks v2.omnizon-13,v2.gomail-1

# Debug mode with limited steps
python scripts/eval_full_matrix.py --debug --max_steps 10
```

**Outputs:**
- `results/<run_id>/matrix.csv` - Raw metrics
- `results/<run_id>/matrix.md` - Formatted comparison
- `results/<run_id>/errors.jsonl` - Error logs with tracebacks

---

## Debugging

### Single Task Debug Runner

Debug a single task with step-by-step output:

```bash
# Run with visible browser (default for debugging)
python scripts/run_one_debug.py --task v2.omnizon-13

# Run with specific mode
python scripts/run_one_debug.py --task v2.gomail-1 --mode hierarchy_vac_macros

# Run headless with limited steps
python scripts/run_one_debug.py --task v2.omnizon-13 --headless true --max_steps 10

# Dry run (verify setup without LLM calls)
python scripts/run_one_debug.py --task v2.omnizon-13 --dry-run
```

**Debug output includes per-step:**
- Step index
- Current URL
- Action taken
- Action source (worker/manager/cache)
- Last action error (if any)
- Page type
- Cache hit status
- Manager called status

---

## Running Tests

### Smoke Test
```bash
python scripts/smoke_test.py
```

### Integration Tests
```bash
# Run all integration tests
python -m pytest tests/integration/ -v

# Run task registry tests
python -m pytest tests/integration/test_task_registry.py -v

# Run debug runner smoke tests
python -m pytest tests/integration/test_run_one_debug_smoke.py -v
```

---

## Agent Modes

| Mode | Description |
|------|-------------|
| `baseline_worker` | Worker only (gpt-4o-mini), no manager, no caching |
| `hierarchy_mgr_gate` | Worker + Manager (gpt-4o for errors/loops/high-stakes) |
| `hierarchy_vac` | Worker + Manager + Verified Action Cache |
| `hierarchy_vac_macros` | Full HALO (Worker + Manager + VAC + Macro Skills) |

---

## Critical Constraints

**ALWAYS enforce these constraints:**

1. **Task Version:** Always use `task_version="v2"` or `v2.` prefix
   - SDK defaults to v1 if omitted - this WILL cause failures!

2. **Browser Viewport:** 1280x720
   - Enforced in harness.py

3. **Observations:** AXTree + Screenshot only
   - `use_html=False` (never use HTML observations)
   - `use_axtree=True`
   - `use_screenshot=True`

4. **Max Steps:**
   - Score mode: 70 steps (default)
   - Speed mode: 25 steps

---

## Metrics Definitions

| Metric | Definition |
|--------|------------|
| `valid_tasks` | Tasks that ran without init/agent errors |
| `init_or_agent_errors` | Tasks that crashed during init or execution |
| `successes` | Valid tasks with reward > 0 |
| `failures` | Valid tasks with reward = 0 |
| `success_rate` | successes / valid_tasks |
| `median_steps` | Median steps across ALL valid tasks |
| `median_wall_time` | Median execution time across valid tasks |

---

## File Locations

| Type | Location |
|------|----------|
| Trajectories | `data/trajectories/<mode>/<run_id>.jsonl` |
| Results | `results/<run_id>/` |
| Error logs | `results/<run_id>/errors.jsonl` |
| Cache data | `data/cache/` |

---

## Troubleshooting

### Tasks fail instantly (~0.02s)

**Cause:** Usually invalid task IDs (tasks that don't exist in v2)

**Solution:**
1. Run `python scripts/list_v2_tasks.py` to see valid tasks
2. Use `--tasks` with valid v2 task IDs
3. Check `results/<run_id>/errors.jsonl` for details

### "No module named 'agisdk'"

**Solution:** Activate the virtual environment:
```bash
source .venv/bin/activate
```

### SDK defaults to v1

**Cause:** Task name missing `v2.` prefix

**Solution:** Always use `v2.` prefix (e.g., `v2.omnizon-13` not `omnizon-13`)

---

## Implementation Status

### Completed Modules

| Module | Status | Description |
|--------|--------|-------------|
| `src/halo/obs/` | Complete | Observation summarizer, fingerprinting, page type |
| `src/halo/policy/` | Complete | Worker (gpt-4o-mini), Manager (gpt-4o), Gating |
| `src/halo/cache/` | Complete | VAC + Macro skills |
| `src/halo/verify/` | Complete | Action verifier, Loop detection |
| `src/halo/logging/` | Complete | JSONL trajectory logger |
| `src/halo/agent/` | Complete | Orchestrator with decision hierarchy |
| `src/halo/sdk/` | Complete | AGI SDK integration |
| `scripts/list_v2_tasks.py` | Complete | Task discovery utility |
| `scripts/eval_subset.py` | Complete | Subset evaluation runner |
| `scripts/eval_full_matrix.py` | Complete | Full benchmark matrix |
| `scripts/run_one_debug.py` | Complete | Single-task debug runner |

---

Last updated: 2025-12-23
