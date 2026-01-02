# HALO Agent vs. ByteDance UI-TARS

## Architecture Comparison

| Component | HALO (AGI Inc) | UI-TARS (ByteDance) |
| :--- | :--- | :--- |
| **Vision** | **DOM-Centric:** Uses HTML/AXTree text summaries. | **Vision-Centric:** Uses pure screenshots + VLM (UI-TARS 1.5). |
| **Control** | **Hierarchical:** Manager (Planner) -> Worker (Clicker). | **End-to-End:** VLM predicts actions directly from pixels. |
| **Training** | **Offline RL:** BC/DPO/GRPO on collected logs. | **Supervised VLM:** Pre-trained on massive GUI datasets. |
| **Stack** | Playwright + OpenAI/Qwen. | Electron + Custom VLM. |

## Key Insights for HALO
1.  **Hybrid Fallback:** UI-TARS falls back to Vision when DOM fails. HALO now implements a **Deterministic Fallback** (Hybrid Agent) using semantic AXTree signals to handle loops and stuck states.
2.  **Observability:** UI-TARS has a live event stream. HALO uses structured JSON logs and `state_hash` tracking to detect loops.
3.  **Data Strategy:** HALO's advantage is the **closed-loop RL**. UI-TARS is static. We must aggressively use `train_grpo_unsloth.py` to self-improve on REAL Bench tasks.

## Hybrid Agent Implementation (New!)
We have integrated a UI-TARS-inspired "Hybrid Agent" layer into HALO:

*   **Robust State Hash:** Uses semantic signals (dialog counts, button states, field validation) instead of raw text to detect loops reliably.
*   **Deterministic Fallback:** A "reflex" layer that handles dialogs, form filling, and submission automatically when the LLM gets stuck or loops.
*   **Failure Taxonomy:** Categorizes failures (Loop, Auth Failed, Form Error) to improve training data quality for GRPO.
