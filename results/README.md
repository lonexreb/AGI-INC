# HALO-Agent Evaluation Results

**Run ID:** test_v2
**Date:** 2025-12-22 20:28

## Comparison Table

| Mode | Success Rate | Median Steps | Median Time (s) | Errors |
|------|-------------|--------------|-----------------|--------|
| baseline | 0.0% | 0.0 | 0.0 | 0 |

## Mode Descriptions

- **baseline**: Worker policy only (gpt-4o-mini), no manager or caching
- **halo**: Worker + Manager (gpt-4o for high-stakes/errors/loops)
- **halo_cache**: Worker + Manager + Verified Action Cache + Macro Skills

## Configuration

- Task Version: v2
- Max Steps: 25
- Browser: 1280x720
- Observations: AXTree + Screenshot (no HTML)
