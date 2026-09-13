#!/usr/bin/env python3
"""Hard-constraint validator for a predictions file (default: repo-root output.csv).

Checks:
  - one row per request, exact schema/order
  - 0 <= amount_safe_to_pay <= requested_amount
  - allowed values for affordability_status / recommended_payment_method
  - affordable_now => earliest == request_date; payment_plan format and chronology
  - installment plans match a supplied payment option exactly
  - partial_payment shape (two payments summing to the requested amount)
  - spending changes target recurring flexible events in categories the user may adjust
"""
import csv, os, sys
from collections import defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "dataset")
COLS = ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
        "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]


def pdate(s):
    s = (s or "").strip()
    if not s:
        return None
    y, m, d = s[:10].split("-")
    return date(int(y), int(m), int(d))


def main(path):
    requests = {r["request_id"]: r for r in csv.DictReader(open(os.path.join(DATA, "requests.csv")))}
    options = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, "request_payment_options.csv"))):
        options[r["request_id"]].append(r)
    events = {e["event_id"]: e for e in csv.DictReader(open(os.path.join(DATA, "financial_events.csv")))}
    profiles = {u["user_id"]: u for u in csv.DictReader(open(os.path.join(DATA, "financial_profiles.csv")))}

    rows = list(csv.DictReader(open(path)))
    errs, warns = [], []
    if [f for f in csv.reader(open(path)).__next__()] != COLS:
        errs.append("header mismatch")
    if len(rows) != len(requests):
        errs.append(f"row count {len(rows)} != requests {len(requests)}")

    for r in rows:
        rid = r["request_id"]
        req = requests.get(rid)
        if not req:
            errs.append(f"{rid}: unknown request"); continue
        try:
            a = float(r["amount_safe_to_pay"])
            want = float(req["requested_amount"])
            if not (0 <= a <= want + 1e-9):
                errs.append(f"{rid}: amount {a} out of [0,{want}]")
        except ValueError:
            errs.append(f"{rid}: non-numeric amount")
        if r["affordability_status"] not in {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}:
            errs.append(f"{rid}: bad status")
        if r["recommended_payment_method"] not in {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}:
            errs.append(f"{rid}: bad method")
        rd = pdate(req["request_date"])
        if r["affordability_status"] == "affordable_now" and r["earliest_date_for_full_payment"] != req["request_date"]:
            errs.append(f"{rid}: affordable_now earliest must equal request_date")
        plan = []
        if r["payment_plan"] != "none":
            prev = None
            for tok in r["payment_plan"].split("|"):
                d_s, a_s = tok.split(":")
                d = pdate(d_s)
                if prev and d < prev:
                    errs.append(f"{rid}: plan not chronological")
                prev = d
                plan.append((d, float(a_s)))
        # installment option match
        if r["recommended_payment_method"] == "installments":
            opts = options.get(rid, [])
            def key(p):
                return [(d.isoformat(), round(a, 2)) for d, a in p]
            if key(plan) not in [key([(pdate(o["first_payment_date"]) + timedelta(days=int(float(o["payment_frequency_days"] or 0) * i)), float(o["payment_amount"])) for i in range(int(float(o["number_of_payments"])))]) for o in opts if o["payment_method"] == "installments"]:
                errs.append(f"{rid}: plan does not match any supplied installment option")
        if r["recommended_payment_method"] == "partial_payment":
            if len(plan) != 2 or abs(sum(a for _, a in plan) - float(req["requested_amount"])) > 0.01:
                errs.append(f"{rid}: partial plan must have 2 payments summing to requested amount")
            if plan and plan[0][0] != rd:
                errs.append(f"{rid}: partial first payment must be on request_date")
        # spending changes
        if r["spending_changes_needed"] != "none":
            prof = profiles[req["user_id"]]
            can_red = set((prof["expense_categories_user_is_willing_to_reduce"] or "").split("|"))
            can_stop = set((prof["expense_categories_user_is_willing_to_stop"] or "").split("|"))
            seen = set()
            for tok in r["spending_changes_needed"].split("|"):
                if tok.startswith("stop:"):
                    eid, kind, val = tok[5:], "stop", None
                elif tok.startswith("reduce_to:"):
                    _, amt = tok[10:].rsplit(":", 1)
                    eid, kind, val = tok[10:].rsplit(":", 1)[0], "reduce", amt
                else:
                    errs.append(f"{rid}: bad change token {tok}"); continue
                e = events.get(eid)
                if not e or e["flexibility"] not in {"reducible", "stoppable", "reducible_or_stoppable"}:
                    errs.append(f"{rid}: {eid} is not flexible")
                if e and kind == "stop" and e["category"] not in can_stop:
                    errs.append(f"{rid}: user will not stop {e['category']}")
                if e and kind == "reduce" and e["category"] not in can_red:
                    errs.append(f"{rid}: user will not reduce {e['category']}")
                if eid in seen:
                    errs.append(f"{rid}: duplicate change target {eid}")
                seen.add(eid)
    print(f"rows={len(rows)} errors={len(errs)}")
    for e in errs[:40]:
        print("ERR", e)
    for w in warns[:10]:
        print("WARN", w)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output.csv"))
