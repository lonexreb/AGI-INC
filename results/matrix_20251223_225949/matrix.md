# HALO-Agent Full Benchmark Matrix

**Run ID:** matrix_20251223_225949
**Date:** 2025-12-24 01:57
**Max Steps:** 70

## Results Matrix

| Mode | Valid/Total | Errors | Success Rate | Δ Score | Median Steps | Usefulness |
|------|-------------|--------|--------------|---------|--------------|------------|
| baseline_worker | 10/10 | 0 | 0.0% | 🟡 +0.0pp | 0.0 | Not useful |
| hierarchy_mgr_gate | 10/10 | 0 | 0.0% | 🟡 +0.0pp | 0.0 | Not useful |
| hierarchy_vac | 10/10 | 0 | 0.0% | 🟡 +0.0pp | 0.0 | Not useful |
| hierarchy_vac_macros | 10/10 | 0 | 0.0% | 🟡 +0.0pp | 0.0 | Not useful |

## Detailed Metrics

| Mode | Successes | Failures | Invalid Action Rate | Manager Call Rate | Cache Hit Rate |
|------|-----------|----------|---------------------|-------------------|----------------|
| baseline_worker | 0 | 10 | 0.0% | 0.0% | 0.0% |
| hierarchy_mgr_gate | 0 | 10 | 0.0% | 0.0% | 0.0% |
| hierarchy_vac | 0 | 10 | 0.0% | 0.0% | 0.0% |
| hierarchy_vac_macros | 0 | 10 | 0.0% | 0.0% | 0.0% |

## Usefulness Labels

- **Useful**: Score improves >= +2 percentage points vs baseline
- **Useful-for-speed**: Score within ±1pp AND median steps decreases >= 10%
- **Not useful**: Does not meet above criteria

## Configuration

- **Task Version:** v2 (CRITICAL: SDK defaults to v1 if omitted)
- **Max Steps:** 70
- **Browser:** 1280x720
- **Observations:** AXTree + Screenshot (no HTML)
