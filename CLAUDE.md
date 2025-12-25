# HALO-Agent — Claude Code Instructions (Project Constitution)

## Mission
Bootstrap and implement HALO-Agent, a worker+manager browser agent for AGI Inc REAL Bench (via AGI SDK).
Primary focus: fast iteration + measurable experiments, and a clear RL pipeline (BC → DPO → optional GRPO pilot).

## Ground Truth / Dependencies
- Use AGI SDK pinned to v0.3.5 (publishes REAL task set v2).
- ALWAYS run tasks with task_version="v2" or v2.* task ids.
- The SDK defaults to v1 if version omitted. Do not omit.  (Critical)
- Pull AGI SDK repo locally for reading examples; use pip install for runtime deps.
- Never read or write secrets. Do not print environment variables. Do not read .env files.

## Definition of Done for Monday Demo
1) SDK harness runs end-to-end on a fixed subset (20–30 tasks) with our custom agent.
2) Worker+Manager architecture working.
3) Verified Action Cache + 3 Macro skills working.
4) JSONL trajectory logs generated for every run.
5) A results summary table: success rate, median steps, median wall time.

## Repo Structure to Create
halo-agent/
  README.md
  CLAUDE.md
  .gitignore
  pyproject.toml
  .env.example
  .claude/settings.json
  .claude/agents/ (subagents)
  src/halo/
    agent/            # orchestrator, routing, policy calls
    policy/           # worker, manager, gating logic
    cache/            # VAC + macro cache
    verify/           # postconditions + loop detection
    obs/              # observation summarizer + fingerprints
    logging/          # structured JSONL logger
    sdk/              # AGI SDK wrappers (REAL.harness integration)
  scripts/
    smoke_test.py
    eval_subset.py
    collect_traj.py
  data/
    trajectories/
    cache/
  results/

## Required AGI SDK Setup
- Install: pip install --upgrade agisdk==0.3.5
- Playwright: playwright install --force
- Clone AGI SDK repo locally at ./third_party/agisdk for example reference:
  git clone https://github.com/agi-inc/agisdk.git third_party/agisdk
  git checkout 0.3.5 (or tag matching)

AGI SDK notes:
- It powers REAL Bench; includes examples for custom agents in example/starter.py, example/custom.py, example/hackable.py.
- Use those examples to implement our custom agent interface exactly.

## Execution Loop (Always)
- Make changes.
- Run scripts/eval_subset.py (fast, fixed subset).
- Save run artifacts in results/<run_id>/ and data/trajectories/<run_id>.jsonl.
- Update results summary in results/README.md.

## Agent Runtime Architecture (must implement)
Decision order per step:
1) Macro Replay Cache (skills)
2) Verified Action Cache (state→action)
3) Worker policy (fast)
4) Manager policy (only for errors/loops/high-stakes)

Caching Rules:
- Only cache actions if verifier confirms expected postcondition.
- If cached action fails twice, evict it.

## RL Roadmap (implement scaffolding now, training later)
- Collect trajectories (success/fail).
- Prepare BC dataset and DPO preference pairs.
- Optional: GRPO pilot on a tiny subset.

## Strict Instructions
- Do not attempt full-benchmark runs unless requested; use eval_subset.
- Do not implement heavy RL training unless the system is stable.
- Prefer clarity and reproducibility over cleverness.
