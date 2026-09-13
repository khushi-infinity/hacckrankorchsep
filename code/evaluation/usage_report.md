# Token Usage and Cost Analysis — Final Full-Dataset Run

## Summary

The final `output.csv` (250 predictions for `dataset/requests.csv`) was produced by the
**deterministic rule-based pipeline** in `code/main.py`. The run used **no external AI model APIs**:
all parsing, forecasting, plan generation, ranking, and OCR (tesseract, run locally) execute on the
local machine.

## Model usage

| Provider | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|
| none (local deterministic engine + local tesseract OCR) | — | 0 | 0 | 0 |
| **Total** | | **0** | **0** | **0** |

## Cost

- Model cost: **USD 0.00**
- Average per request: **USD 0.00** (0 tokens across 250 requests)

## Rationale

The decision space (balance forecasting, plan feasibility, spec ranking rules) is fully specified by
the provided CSVs, so a deterministic engine guarantees reproducibility, zero API cost, and zero
token usage while satisfying the "no hardcoded labels" requirement: every decision is derived from
`financial_profiles.csv`, `financial_events.csv`, `messages.csv`, images, and
`request_payment_options.csv` at run time.
