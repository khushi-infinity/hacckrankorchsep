# Buy or Wait? — Financial Affordability Agent

Deterministic AI financial agent for HackerRank Orchestrate (September 2026). For every request in
`dataset/requests.csv` it reconstructs the user's finances, forecasts 90 days of cashflow, and
recommends the safest way to proceed: pay in full, pay partially, use installments, wait, or not proceed.

## Run

```bash
python3 code/main.py            # -> output.csv (250 predictions, repo root)
python3 code/main.py --samples  # -> output_samples.csv (25 solved samples)
python3 code/evaluate.py        # score output_samples.csv against ground truth
python3 code/evaluate.py output.csv  # self-check any predictions file
```

No API keys, network, or external services are required. Everything is deterministic given the dataset.

## Approach

1. **Financial state reconstruction** (`code/main.py`)
   - Recurring monthly series are detected only when settled history supports them: events are
     grouped by `(event_type, category, direction)`, must have mostly 20–40 day gaps, and forecast
     amounts use the mode of the last three occurrences (ties → most recent value).
   - Pending debits are reserved; pending credits, failed, cancelled, duplicate and `unrealized`
     (non-cash investment) records are ignored. One-time (non-recurring) settled future events
     within the 90-day window still count.
   - Foreign-currency events are converted with the dated `exchange_rates.csv` table.
   - Blank-amount events are resolved through `images.csv` → OCR of the linked PNG
     (tesseract via pytesseract; the pipeline degrades gracefully if OCR is unavailable).
     Money tokens handle US, Indian (1,80,000) and decimal-comma (33,50) formats.
2. **Evidence from messages** — employer messages amend future income only when they describe a
   confirmed change (raises take effect on their stated date); bonuses/commissions are never counted
   until settled. Messages that explicitly cancel an event remove it from the forecast.
3. **90-day safety check** — a daily-balance ledger from `request_date` enforces
   `balance >= minimum_balance_to_keep` after every projected essential expense and every payment.
4. **Decision layer**
   - `amount_safe_to_pay` = max payable today before optional spending changes (headroom minimum over
     90 days, capped at the requested amount).
   - `earliest_date_for_full_payment` = first date a single full payment passes the safety check.
   - Candidate plans: full payment today (with/without spending changes on flexible recurring
     expenses the user agreed to adjust), partial payment (two payments summing to the full amount,
     only when allowed + accepted + completable by the deadline), every supplied installment option
     (respected exactly, filtered by `max_installment_months`), and wait.
   - Ranking follows the problem statement: complete by `desired_completion_date`, no spending
     changes, lowest total cost, earliest start, fewest payments, lowest `payment_option_id`.
5. **Personalization** — payment methods considered, installment limits, protected categories, and
   the flexible expense lists all come from `financial_profiles.csv`, so two users with the same
   balance can receive different recommendations.
6. **Prompt-injection safety** — messages and images are parsed as data only; their text can never
   override the output contract or decision rules.

## Evaluation workflow

`code/evaluate.py` scores a predictions file against `dataset/sample_requests.csv` (25 solved
examples) on the five comparable fields (amount, status, method, plan, earliest date) and prints
per-request diagnostics. `evaluation/validate_output.py` re-checks hard constraints on any run:
row count, schema, `0 <= amount_safe_to_pay <= requested_amount`, chronological payment plans,
installment plans matching a supplied option exactly, and spending changes targeting flexible
recurring events only. `evaluation/usage_report.md` documents the final full-dataset run.

## Results on the solved samples (sanity check, not training data)

- All `affordable_now` and all `not_affordable` samples are reproduced exactly.
- Statuses, methods and plans match the ground-truth structure; remaining amount deltas come from
  conservative forecasting of variable spending categories.

## Limitations

- The OCR step needs `tesseract` + `pytesseract` for the 16 blank-amount events; without it the
  engine treats those events as unsupported (never as zero).
- Variable monthly amounts (e.g. dining) are forecast at the recent mode; the hidden ground truth
  may use a different but similar estimator.
