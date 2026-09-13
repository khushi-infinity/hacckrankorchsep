#!/usr/bin/env python3
"""Score output.csv against the 25 solved sample requests."""
import csv, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def close(a, b, tol=0.01):
    try:
        return abs(float(a) - float(b)) <= tol * max(1, abs(float(b)))
    except Exception:
        return str(a).strip() == str(b).strip()

def main(out_path):
    samples = list(csv.DictReader(open(os.path.join(ROOT, "dataset", "sample_requests.csv"))))
    try:
        out = {r["request_id"]: r for r in csv.DictReader(open(out_path))}
    except Exception as e:
        print("Cannot read output:", e); sys.exit(1)

    score = Counter()
    exact = 0
    detail = []
    for s in samples:
        rid = s["request_id"]
        o = out.get(rid)
        if not o:
            detail.append((rid, "MISSING")); continue
        checks = []
        checks.append(("amt", close(o["amount_safe_to_pay"], s["amount_safe_to_pay"])))
        checks.append(("status", o["affordability_status"] == s["affordability_status"]))
        checks.append(("method", o["recommended_payment_method"] == s["recommended_payment_method"]))
        checks.append(("plan", o["payment_plan"] == s["payment_plan"]))
        checks.append(("earliest", (o["earliest_date_for_full_payment"] or "").strip() == (s["earliest_date_for_full_payment"] or "").strip()))
        ok_n = sum(1 for _, ok in checks if ok)
        if ok_n == len(checks):
            exact += 1
        score.update(["ok" for _, ok in checks if ok] + ["miss" for _, ok in checks if not ok])
        detail.append((rid, f"{ok_n}/{len(checks)}", [(n, ok) for n, ok in checks if not ok]))
    total_fields = sum(len(samples) * 5 for _ in [1])
    print(f"Exact row matches: {exact}/{len(samples)}")
    print(f"Field hits: {score['ok']}/{score['ok']+score['miss']}")
    for d in detail:
        print(d)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output.csv"))
