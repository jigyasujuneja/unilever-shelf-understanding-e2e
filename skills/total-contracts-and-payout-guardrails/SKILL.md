---
name: total-contracts-and-payout-guardrails
description: >-
  Engineering playbook for mission-critical AI pipelines tied to financial payouts
  or SLAs (PR-16, PR-17, EXP-59): Level-4 Total Contracts, closed decision state
  machines, PT180S SLA timeout quarantine (RETRY_PENDING vs false-negative 0 payouts),
  asymmetric error cost matrices, untrusted OCR prompt-injection isolation, and
  content-addressed SHA-256 VLM response caching.
---

# Level-4 Total Contracts, Financial Payout Guardrails & VLM Caching Playbook

## 1. Level-4 Total Contracts & Closed Decision State Machines (`PR-16`, `EXP-59`)

Never allow an AI inference worker to return raw untyped strings, raise unhandled exceptions into a business workflow, or conflate infrastructure errors with negative business compliance:

1. **Closed 6-State Decision Taxonomy**:
   - Every item evaluation must terminate in a mutually exclusive state:
     - `STAGE2A_FAST_EXIT`: High-margin embedding match accepted at `$0` VLM token cost.
     - `STAGE2B_VERIFIED`: Confirmed by calibrated contact-sheet verification + Bayesian PoE.
     - `STAGE2A_ENERGY_VETO_RETENTION`: Low-confidence Stage 2B flip blocked by the `0.045` visual energy veto or switched by the `PR-14` pack-size verifier.
     - `ABSTAIN_RECAPTURE`: Flagged for merchandiser re-snap due to physical blur/glare/occlusion.
     - `CIRCUIT_BREAKER_FALLBACK`: Served via Stage 2A Rank-1 when upstream VLM endpoints throttle (`HTTP 429` / `503`).
     - `REJECTED_CONTRACT_VIOLATION`: Malformed payload or schema violation isolated before scoring.
2. **Discriminated Failure Taxonomy (System Faults vs. Model Reasoning Errors)**:
   - Track `SYSTEM_CONTRACT_VIOLATION` (timeouts, HTTP 429s, schema errors) separately from `RETRIEVAL_TOP5_MISS`, `VISUAL_PERCEPTION_BLUR_GLARE`, and `REASONING_CALIBRATION_ERROR` so infrastructure hiccups never corrupt model accuracy leaderboards.

## 2. Pay-for-Performance (`P2P`) Payout Quarantine Gate (`PT180S` SLA)

When AI outputs drive retailer financial payouts (`Earning` / `W1` incentives), a false negative caused by a cloud timeout costs $10\times$ more in store disputes (`$2.50` manual audit escalation) than a delayed retry or re-snap (`$0.25`):

1. **Strict `PT180S` SLA Quarantine Contract**:
   - Any store photo exceeding the `180s` processing window or encountering transient upstream exhaustion (`HTTP 429` after retries) must emit:
     - `contract_status = "RETRY_PENDING"`
     - `awp_presence = "PENDING"`
     - `earning_override = None`
   - Route the message to the dead-letter / retry queue (`PR19_pubsub_dlq`).
2. **Zero False Payout Denials**:
   - Inference workers must **never** emit `Earning = "0"` or `awp_presence = "False"` on a timeout, rate-limit, or corrupt upload.

## 3. Untrusted OCR Isolation & Content-Addressed VLM Caching (`PR-16`, `PR-17`)

1. **Indirect Prompt Injection Defense (`<untrusted_pack_text>`)**:
   - Shelf talkers, competitor packaging, and store banners are untrusted external inputs.
   - Strip instruction-override patterns (`ignore previous`, `system:`, `choice:`) via regex and wrap all extracted OCR strings inside `<untrusted_pack_text>...</untrusted_pack_text>` XML tags with an explicit system rule that text inside those tags is data only.
2. **Content-Addressed SHA-256 VLM Response Cache & Checkpointing (`PR-17`)**:
   - Key every VLM evaluation call by `SHA256(crop_bytes || model_id || prompt_template_hash || candidate_sku_ids)`.
   - Persist responses to a content-addressed JSONL disk/GCS cache with per-sample checkpointing so interrupted benchmarks or threshold sweeps (`C_2`, energy veto, PoE weights) resume and re-evaluate in `<50 ms` with `$0` duplicate token spend.
