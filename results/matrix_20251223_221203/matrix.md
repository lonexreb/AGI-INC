# HALO-Agent Full Benchmark Matrix

**Run ID:** matrix_20251223_221203
**Date:** 2025-12-23 22:17
**Max Steps:** 15

## Results Matrix

| Mode | Valid/Total | Errors | Success Rate | Δ Score | Median Steps | Usefulness |
|------|-------------|--------|--------------|---------|--------------|------------|
| baseline_worker | 3/3 | 0 | 0.0% | 🟡 +0.0pp | 0.0 | Not useful |
| hierarchy_vac_macros | 3/3 | 0 | 0.0% | 🟡 +0.0pp | 0.0 | Not useful |

## Detailed Metrics

| Mode | Successes | Failures | Invalid Action Rate | Manager Call Rate | Cache Hit Rate |
|------|-----------|----------|---------------------|-------------------|----------------|
| baseline_worker | 0 | 3 | 0.0% | 0.0% | 0.0% |
| hierarchy_vac_macros | 0 | 3 | 0.0% | 0.0% | 0.0% |

## Usefulness Labels

- **Useful**: Score improves >= +2 percentage points vs baseline
- **Useful-for-speed**: Score within ±1pp AND median steps decreases >= 10%
- **Not useful**: Does not meet above criteria

## Configuration

- **Task Version:** v2 (CRITICAL: SDK defaults to v1 if omitted)
- **Max Steps:** 15
- **Browser:** 1280x720
- **Observations:** AXTree + Screenshot (no HTML)
