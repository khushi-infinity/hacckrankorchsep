#!/usr/bin/env python3
"""
Buy or Wait? — AI-powered financial affordability agent (HackerRank Orchestrate Sept 2026).

Deterministic pipeline per request:
  1. Reconstruct the user's financial state from financial_profiles.csv + financial_events.csv
     (recurring series detection, pending reserves, duplicate/cancelled/failed filtering,
     foreign-currency conversion via exchange_rates.csv, income only when settled/scheduled).
  2. Enrich with evidence from messages.csv (amendments/cancellations) and images (OCR of
     blank-amount events when pytesseract is available).
  3. Forecast the daily balance for 90 days enforcing minimum_balance_to_keep.
  4. Generate candidate plans (full / partial / installments / wait / spending changes),
     verify each against the forecast, rank by the spec's tie-break rules, emit output.csv.

No hardcoded answers; every decision is derived from the data. See README.md.
"""
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")

STATUSES_IGNORED = {"cancelled", "failed"}


def pdate(s):
    s = (s or "").strip()
    if not s:
        return None
    y, m, d = s[:10].split("-")
    return date(int(y), int(m), int(d))


def num(s):
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return 0.0


def fmt_amt(x):
    x = round(x + 0.0, 2)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s


def load_csv(name):
    with open(os.path.join(DATA, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- data loading
class Store:
    def __init__(self):
        self.profiles = {r["user_id"]: r for r in load_csv("financial_profiles.csv")}
        self.requests = load_csv("requests.csv")
        self.events = load_csv("financial_events.csv")
        self.options = defaultdict(list)
        for r in load_csv("request_payment_options.csv"):
            self.options[r["request_id"]].append(r)
        self.messages = load_csv("messages.csv")
        self.images = load_csv("images.csv")
        self.rates = {}
        for r in load_csv("exchange_rates.csv"):
            self.rates[(r["rate_date"], r["from_currency"], r["to_currency"])] = num(r["rate"])
        self.msg_by_user = defaultdict(list)
        self.msg_by_request = defaultdict(list)
        self.msg_by_event = defaultdict(list)
        for m in self.messages:
            self.msg_by_user[m["user_id"]].append(m)
            if m["request_id"]:
                self.msg_by_request[m["request_id"]].append(m)
            if m["related_event_id"]:
                self.msg_by_event[m["related_event_id"]].append(m)
        self.img_by_event = defaultdict(list)
        self.img_by_request = defaultdict(list)
        for im in self.images:
            if im["related_event_id"]:
                self.img_by_event[im["related_event_id"]].append(im)
            if im["request_id"]:
                self.img_by_request[im["request_id"]].append(im)

    def to_home(self, amount, cur, home, on_date):
        """Convert amount in `cur` to `home` currency using the dated rate table."""
        if cur == home or not cur:
            return amount
        d = on_date.isoformat() if isinstance(on_date, date) else str(on_date)[:10]
        for rd in (d, _prev_month_end(d)):
            r = self.rates.get((rd, cur, home))
            if r:
                return amount * r
        # try any available date for the pair (deterministic: latest date <= d, else earliest)
        cands = sorted(k[0] for k in self.rates if k[1] == cur and k[2] == home)
        if cands:
            le = [c for c in cands if c <= d]
            rd = le[-1] if le else cands[0]
            return amount * self.rates[(rd, cur, home)]
        return amount  # last resort


def _prev_month_end(dstr):
    y, m, _ = dstr.split("-")
    y, m = int(y), int(m)
    m -= 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}-28"


# ------------------------------------------------------------ message evidence
MONEY_RE = re.compile(r"(?:IDR|ZAR|INR|EUR|USD|Rps|Rp)?\s?([0-9][0-9,\.]{2,20})", re.I)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
CANCEL_WORDS = ["cancel", "batal", "dibatalkan", "refunded", "dikembalikan", "gagal", "failed", "ditolak"]
RAISE_WORDS = ["naik", "raise", "increas", "meningkat", "bertambah", "new salary", "gaji bulanan",
               "berubah", "changed", "berlaku", "effective", "adjust"]
BONUS_WORDS = ["bonus", "commission", "komisi", "reward", "incentive", "lottery", "undian"]


def parse_messages(store, user, request, events_by_id):
    """Return dict of amendments: cancelled_event_ids, income_change (amount, from_date)."""
    out = {"cancelled": set(), "income": []}
    msgs = list(store.msg_by_user.get(user, [])) + list(store.msg_by_request.get(request["request_id"], []))
    seen = set()
    for m in msgs:
        if m["message_id"] in seen:
            continue
        seen.add(m["message_id"])
        txt = m["message_text"] or ""
        low = txt.lower()
        if m["related_event_id"]:
            if any(w in low for w in CANCEL_WORDS):
                out["cancelled"].add(m["related_event_id"])
        if m["source_type"] == "employer" and any(w in low for w in RAISE_WORDS) and not any(w in low for w in BONUS_WORDS):
            amounts = [parse_money_token(a) for a in re.findall(r"\d[\d,\.]*", txt)]
            amounts = [a for a in (amounts or []) if a and a > 1000]
            dates = DATE_RE.findall(txt)
            if amounts:
                eff = pdate(dates[-1]) if dates else None
                out["income"].append({"amount": max(amounts), "eff": eff, "mid": m["message_id"]})
    return out


# ------------------------------------------------------------- OCR for images
def ocr_image(path):
    try:
        import pytesseract
        from PIL import Image
        txt = pytesseract.image_to_string(Image.open(path))
        return txt
    except Exception:
        return ""


def parse_money_token(tok):
    """Parse a numeric token handling US/Indian thousands and decimal commas."""
    t = tok.strip().rstrip('.').lstrip('.')
    if not t or not any(c.isdigit() for c in t):
        return None
    if ',' in t and '.' in t:
        return num(t.replace(',', ''))
    if ',' in t:
        parts = t.split(',')
        if len(parts) == 2 and len(parts[-1]) == 2:
            return num(t.replace(',', '.'))  # decimal comma e.g. 33,50
        return num(t.replace(',', ''))       # 4,365,000 / 1,80,000 / 21,599
    return num(t)


def ocr_amount_from_text(txt):
    if not txt:
        return None
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    def nums_in(line, lo=10, hi=1e10):
        out = []
        for m in re.finditer(r"\d[\d,\.]*", line):
            v = parse_money_token(m.group(0))
            if v is not None and lo <= v <= hi:
                out.append(v)
        return out
    # 1) salary docs: net pay / transferred amount
    for l in lines:
        low = l.lower()
        if ('transferred' in low or 'net pay' in low) and nums_in(l):
            return max(nums_in(l))
    # 2) amount due lines (first value on the line)
    for l in lines:
        low = l.lower()
        if 'amount due' in low:
            m = re.search(r'=\s*([\d,\.]+)', low)
            if m:
                v = parse_money_token(m.group(1))
                if v:
                    return v
            vs = nums_in(l)
            if vs:
                return vs[0]
    # 3) total lines (last wins)
    for l in reversed(lines):
        low = l.lower()
        if re.search(r'\b(total paid|grand total|total|net amount|balance due)\b', low) and 'qty' not in low:
            vs = nums_in(l)
            if vs:
                return max(vs)
    # 4) last line with a plausible number
    for l in reversed(lines):
        vs = nums_in(l)
        if vs:
            return max(vs)
    return None


# -------------------------------------------------- recurring series detection
def detect_recurring(events, request_date):
    """Group settled history into monthly series. Returns list of dicts."""
    by_key = defaultdict(list)
    for e in events:
        if e["status"] != "settled" or e["direction"] not in ("debit", "credit"):
            continue
        d = pdate(e["event_date"]) or pdate(e["settlement_date"])
        if not d or d >= request_date:
            continue
        by_key[(e["event_type"], e["category"], e["direction"])].append((d, e))
    series = []
    for key, items in by_key.items():
        items.sort(key=lambda t: t[0])
        if len(items) < 3:
            continue
        gaps = [(items[i + 1][0] - items[i][0]).days for i in range(len(items) - 1)]
        good = [g for g in gaps if 20 <= g <= 40]
        if len(good) < max(1, len(gaps) - 1):
            continue  # not reliably monthly
        recent = items[-3:]
        amounts = [num(e["amount"]) for _, e in recent]
        # mode of recent amounts, tie -> most recent value
        cnt = defaultdict(int)
        for a in amounts:
            cnt[round(a, 2)] += 1
        best = sorted(cnt.items(), key=lambda kv: (kv[1], amounts[-1] == kv[0]), reverse=True)
        amt = best[0][0]
        day = recent[-1][0].day
        flex = recent[-1][1]["flexibility"]
        min_amt = num(recent[-1][1]["minimum_allowed_amount"]) if recent[-1][1]["minimum_allowed_amount"].strip() else 0.0
        ev_ids = [e["event_id"] for _, e in items]
        series.append({
            "type": key[0], "category": key[1], "direction": key[2], "amount": amt,
            "day": day, "flex": flex, "min_amount": min_amt, "ids": ev_ids,
            "last_id": ev_ids[-1], "n": len(items),
        })
    return series


def project_series(series, start, days):
    """Yield (date, signed_amount, series) for projected occurrences in (start, start+days]."""
    out = []
    d = start
    end = start + timedelta(days=days)
    for s in series:
        # occurrences: same day-of-month each month
        y, m = start.year, start.month
        for k in range(0, 4):
            mm = m + k
            yy = y + (mm - 1) // 12
            mm = (mm - 1) % 12 + 1
            try:
                dd = min(s["day"], 28)
                occ = date(yy, mm, dd)
            except ValueError:
                continue
            if start < occ <= end:
                sign = -1 if s["direction"] == "debit" else 1
                out.append((occ, sign * s["amount"], s))
    return out


# ------------------------------------------------------------------- forecasting
class Forecast:
    """Daily net cashflow model for one user relative to request_date."""

    def __init__(self, store, user, request_date, amendments, days=90):
        self.store = store
        self.user = user
        self.start = request_date
        self.days = days
        prof = store.profiles[user]
        self.home = prof["home_currency"]
        self.bal0 = num(prof["current_available_balance"])
        self.minbal = num(prof["minimum_balance_to_keep"])
        self.events_by_id = {e["event_id"]: e for e in store.events if e["user_id"] == user}
        user_events = [e for e in store.events if e["user_id"] == user]
        self.series = detect_recurring(user_events, request_date)
        self.cancelled = set(amendments.get("cancelled", set()))
        self.onetime = []  # (date, signed_home_amount, label)
        self._build_one_time(user_events)
        self._apply_income_amendments(amendments)
        self.base_daily = self._daily_map(self.recurring_flows(), self.onetime)

    # -- helpers ------------------------------------------------------------
    def _build_one_time(self, user_events):
        seen_dup = set()
        for e in user_events:
            if e["event_id"] in self.cancelled:
                continue
            st = e["status"]
            if st in STATUSES_IGNORED or st == "unrealized" or e["direction"] == "non_cash":
                continue
            d = pdate(e["settlement_date"]) or pdate(e["event_date"])
            if not d or not (self.start < d <= self.start + timedelta(days=self.days)):
                continue
            if e["direction"] == "credit" and st == "pending":
                continue  # pending credits never counted
            if st == "pending" and e["direction"] == "debit":
                pass  # reserve: future debit
            amt, cur = num(e["amount"]), e["currency"]
            if not amt and st == "settled":
                continue  # historical blank amount irrelevant for future
            if not amt:
                # blank amount -> look for image evidence
                amt = self._amount_from_image(e)
                if not amt:
                    continue
            key = (e["event_type"], e["category"], e["direction"], round(amt, 2), d.isoformat())
            if key in seen_dup:
                continue  # duplicate representation
            seen_dup.add(key)
            home_amt = self.store.to_home(amt, cur, self.home, d)
            sign = -1 if e["direction"] == "debit" else 1
            self.onetime.append((d, sign * home_amt, e))

    def _amount_from_image(self, ev):
        imgs = self.store.img_by_event.get(ev["event_id"], [])
        for im in imgs:
            p = os.path.join(DATA, "media", "images", im["image_id"] + ".png")
            if os.path.exists(p):
                v = ocr_amount_from_text(ocr_image(p))
                if v:
                    return self.store.to_home(v, ev["currency"] or self.home, self.home, self.start)
        return None

    def _apply_income_amendments(self, amendments):
        for inc in amendments.get("income", []):
            amt = inc["amount"]
            eff = inc["eff"] or self.start
            if eff <= self.start + timedelta(days=self.days):
                self.onetime.append((max(eff, self.start + timedelta(days=1)), amt, {"event_id": "msg:" + inc["mid"]}))

    def recurring_flows(self, overrides=None):
        """overrides: {series_key: ('stop',) | ('reduce', new_amount)}"""
        overrides = overrides or {}
        flows = []
        for i, s in enumerate(self.series):
            key = s["last_id"]
            ov = overrides.get(key)
            if ov and ov[0] == "stop":
                continue
            amt = s["amount"]
            if ov and ov[0] == "reduce":
                amt = max(ov[1], 0.0)
            sign = -1 if s["direction"] == "debit" else 1
            flows.extend((d, sign * amt, s) for d, _, _ in project_series([s], self.start, self.days))
        return flows

    def _daily_map(self, rec_flows, onetime):
        daily = defaultdict(float)
        for d, amt, _ in rec_flows:
            daily[d] += amt
        for d, amt, _ in onetime:
            daily[d] += amt
        return daily

    def net_by(self, t):
        """Cumulative net cashflow from start (exclusive) to t (inclusive), no request payment."""
        tot = 0.0
        for d, v in self.base_daily.items():
            if self.start < d <= t:
                tot += v
        return tot

    def min_over_window(self, pay_date=None, pay_amt=0.0, extra=()):
        """Minimum balance over (start, start+days] given an optional payment. extra: [(date, amt)]"""
        daily = dict(self.base_daily)
        for d, amt in extra:
            daily[d] = daily.get(d, 0.0) + amt
        lo = self.bal0 - (pay_amt if pay_date == self.start else 0.0)
        if lo < self.minbal:
            return lo
        bal = self.bal0 - (pay_amt if pay_date == self.start else 0.0)
        end = self.start + timedelta(days=self.days)
        d = self.start
        while d <= end:
            d += timedelta(days=1)
            if d > end:
                break
            bal += daily.get(d, 0.0)
            if d == pay_date:
                bal -= pay_amt
            if bal < self.minbal:
                return bal
        return bal

    def max_safe_today(self, cap):
        """Max payable today keeping min balance over 90d, without spending changes."""
        # headroom = min over t of (bal0 + net(t) - minbal)
        head = None
        bal = self.bal0
        end = self.start + timedelta(days=self.days)
        d = self.start
        daily = self.base_daily
        while d < end:
            d += timedelta(days=1)
            bal += daily.get(d, 0.0)
            head = bal - self.minbal if head is None else min(head, bal - self.minbal)
        head = min(head if head is not None else self.bal0 - self.minbal, self.bal0 - self.minbal)
        return max(0.0, min(cap, head))

    def earliest_full_date(self, amount, horizon=None):
        """First date paying `amount` in one go keeps balance >= min through window end."""
        horizon = horizon or self.days
        end_all = self.start + timedelta(days=self.days)
        # today first
        if self.plan_safe([(self.start, amount)]):
            return self.start
        d = self.start
        bal = self.bal0
        daily = self.base_daily
        while d < end_all:
            d += timedelta(days=1)
            bal += daily.get(d, 0.0)
            if d > self.start + timedelta(days=horizon):
                break
            if bal - amount >= self.minbal:
                # verify rest of window with payment at d
                bal2 = bal - amount
                d2 = d
                while d2 < end_all:
                    d2 += timedelta(days=1)
                    bal2 += daily.get(d2, 0.0)
                    if bal2 < self.minbal:
                        break
                if bal2 >= self.minbal:
                    return d
        return None

    def plan_safe(self, payments):
        """payments: [(date, amt)] all on/before window end; check min balance each day."""
        daily = dict(self.base_daily)
        pays = defaultdict(float)
        for d, a in payments:
            pays[d] += a
        bal = self.bal0
        end = self.start + timedelta(days=self.days)
        d = self.start
        while d < end:
            d += timedelta(days=1)
            if d > end:
                break
            bal += daily.get(d, 0.0) - pays.get(d, 0.0)
            if bal < self.minbal - 1e-9:
                return False
        return True


# ------------------------------------------------------------- spending changes
def find_spending_changes(f, requested):
    """Greedy search for stop/reduce changes on flexible recurring debits so full is safe today."""
    gap = requested - f.max_safe_today(requested)
    if gap <= 1e-9:
        return []
    prof = f.store.profiles[f.user]
    can_reduce = set((prof["expense_categories_user_is_willing_to_reduce"] or "").split("|")) - {""}
    can_stop = set((prof["expense_categories_user_is_willing_to_stop"] or "").split("|")) - {""}
    cands = []
    for s in f.series:
        if s["direction"] != "debit":
            continue
        if s["flex"] in ("stoppable", "reducible_or_stoppable") and s["category"] in can_stop and s["amount"] >= 1:
            cands.append({"kind": "stop", "id": s["last_id"], "series": s,
                          "value": s["amount"], "min_amount": 0.0})
        if s["flex"] in ("reducible", "reducible_or_stoppable") and s["category"] in can_reduce:
            floor = s["min_amount"] or 0.0
            save = s["amount"] - floor
            if save >= 1:
                cands.append({"kind": "reduce", "id": s["last_id"], "series": s,
                              "value": floor, "min_amount": floor})
    cands.sort(key=lambda c: -c["series"]["amount"])
    chosen = []
    used = set()
    for c in cands:
        if len(chosen) >= 3:
            break
        if c["id"] in used:
            continue
        ov = {c["id"]: (c["kind"], c["value"])}
        trial = dict(ov)
        for ch in chosen:
            trial[ch["id"]] = (ch["kind"], ch["value"])
        f2 = _forecast_with_overrides(f, trial)
        if f2.max_safe_today(requested) >= requested - 1e-6:
            chosen.append(c)
            used.add(c["id"])
            gap = requested - f2.max_safe_today(requested)
            if gap <= 1e-9:
                break
    return chosen if gap <= 1e-9 else (chosen if _forecast_with_overrides(
        f, {c["id"]: (c["kind"], c["value"]) for c in chosen}).max_safe_today(requested) >= requested - 1e-6 else chosen)


def _forecast_with_overrides(f, overrides):
    import copy
    g = copy.copy(f)
    rec = f.recurring_flows(overrides)
    g.base_daily = f._daily_map(rec, f.onetime)
    return g


# ------------------------------------------------------------------- decisions
def evaluate(store, request):
    user = request["user_id"]
    rd = pdate(request["request_date"])
    desired = pdate(request["desired_completion_date"])
    requested = num(request["requested_amount"])
    prof = store.profiles[user]
    methods = [m.strip() for m in (prof["payment_methods_user_will_consider"] or "").split("|") if m.strip()]
    max_inst = int(num(prof["max_installment_months"]))
    amendments = parse_messages(store, user, request, {})
    f = Forecast(store, user, rd, amendments)

    cap = min(requested, f.bal0)
    astp = round(f.max_safe_today(cap), 2)
    earliest = f.earliest_full_date(requested)

    currency = prof["home_currency"]
    cur_fmt = lambda x: f"{currency} {fmt_amt(x)}"

    candidates = []  # each: dict(status, method, plan=[(date, amt)], changes=[], opt_id, start, total)

    # 1) full payment today
    if "full_payment" in methods and earliest == rd and requested <= astp + 1e-6:
        candidates.append({"status": "affordable_now", "method": "full_payment",
                           "plan": [(rd, requested)], "changes": [], "opt_id": "zzz",
                           "start": rd, "total": requested, "by_desired": True})

    # 2) full payment today with spending changes
    if "full_payment" in methods and astp < requested - 1e-6 and requested <= f.bal0:
        changes = find_spending_changes(f, requested)
        if changes:
            ov = {c["id"]: (c["kind"], c["value"]) for c in changes}
            f2 = _forecast_with_overrides(f, ov)
            if f2.max_safe_today(cap) >= requested - 1e-6:
                candidates.append({"status": "affordable_with_plan", "method": "full_payment",
                                   "plan": [(rd, requested)], "changes": changes,
                                   "opt_id": "zzz", "start": rd, "total": requested,
                                   "by_desired": True})

    # 3) partial payment
    allows_partial = (request["allows_partial_payment"] or "").strip().lower() == "true"
    if ("partial_payment" in methods and allows_partial and 0 < astp < requested - 1e-6
            and earliest and earliest <= desired):
        rem = round(requested - astp, 2)
        if f.plan_safe([(rd, astp), (earliest, rem)]):
            candidates.append({"status": "affordable_with_plan", "method": "partial_payment",
                               "plan": [(rd, astp), (earliest, rem)], "changes": [],
                               "opt_id": "zzz", "start": rd, "total": requested,
                               "by_desired": True})

    # 4) installments from supplied options
    for opt in store.options.get(request["request_id"], []):
        if opt["payment_method"] != "installments":
            continue
        n = int(num(opt["number_of_payments"]))
        if max_inst and n > max_inst:
            continue
        amt = num(opt["payment_amount"])
        first = pdate(opt["first_payment_date"])
        freq = int(num(opt["payment_frequency_days"] or 0)) or 30
        if not first or first < rd:
            continue
        pays = [(first + timedelta(days=freq * i), amt) for i in range(n)]
        if pays[-1][0] > f.start + timedelta(days=f.days):
            continue  # schedule exceeds the 90-day forecast window: cannot verify safety
        last = pays[-1][0]
        if f.plan_safe(pays):
            candidates.append({"status": "affordable_with_plan", "method": "installments",
                               "plan": pays, "changes": [], "opt_id": opt["payment_option_id"],
                               "start": first, "total": n * amt + num(opt["financing_fee"]),
                               "by_desired": last <= desired})

    # 5) wait for a later full payment (pay on the desired completion date when safe)
    if "full_payment" in methods and earliest and earliest > rd and desired > rd:
        wait_date = desired if f.plan_safe([(desired, requested)]) else earliest
        if f.plan_safe([(wait_date, requested)]):
            candidates.append({"status": "affordable_later", "method": "wait",
                               "plan": [(wait_date, requested)], "changes": [],
                               "opt_id": "zzz", "start": wait_date, "total": requested,
                               "by_desired": wait_date <= desired})

    # rank: completes by desired, no changes, min total, earlier start, fewer payments, option id
    def rank(c):
        n_pay = len(c["plan"]) if c["plan"] else 99
        return (0 if c["by_desired"] else 1,
                0 if not c["changes"] else 1,
                round(c["total"], 2),
                c["start"].isoformat(),
                n_pay,
                c["opt_id"])

    candidates.sort(key=rank)

    # output assembly ------------------------------------------------------
    if candidates:
        c = candidates[0]
        status = c["status"]
        method = c["method"]
        plan = "|".join(f"{d.isoformat()}:{fmt_amt(a)}" for d, a in c["plan"]) if c["plan"] else "none"
        changes = "|".join(
            f"stop:{ch['id']}" if ch["kind"] == "stop" else f"reduce_to:{ch['id']}:{fmt_amt(ch['value'])}"
            for ch in c["changes"]) or "none"
        earliest_out = earliest.isoformat() if (earliest and status != "affordable_later") else ""
        if status == "affordable_now":
            earliest_out = rd.isoformat()
        if status == "affordable_later" and c["plan"]:
            earliest_out = c["plan"][0][0].isoformat()
        expl = _explain(request, f, c, status, currency)
    else:
        status, method, plan, changes = "not_affordable", "not_recommended", "none", "none"
        earliest_out = ""
        expl = (f"Do not proceed with {cur_fmt(requested)} by {desired.isoformat()}. "
                f"Although {cur_fmt(astp)} is available today, the full amount cannot be "
                f"completed safely within 90 days while protecting the {cur_fmt(f.minbal)} minimum.")
        if astp <= 0:
            expl = (f"Do not make this payment by {desired.isoformat()}. None of the available options "
                    f"keeps the {cur_fmt(f.minbal)} minimum protected.")

    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": fmt_amt(astp),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan,
        "earliest_date_for_full_payment": earliest_out,
        "spending_changes_needed": changes,
        "decision_explanation": expl,
    }


def _explain(request, f, c, status, currency):
    rd = f.start
    cur = lambda x: f"{currency} {fmt_amt(x)}"
    if status == "affordable_now":
        return (f"Pay {cur(c['total'])} today. This leaves at least {cur(f.minbal)} available "
                f"over the next 90 days.")
    if c["method"] == "full_payment" and c["changes"]:
        ch = c["changes"]
        parts = []
        for x in ch:
            if x["kind"] == "stop":
                parts.append(f"stop {x['series']['category']} ({x['id']})")
            else:
                parts.append(f"reduce {x['series']['category']} to {cur(x['value'])} ({x['id']})")
        return (f"{'; '.join(parts).capitalize()}, then pay {cur(c['total'])} today. "
                f"This keeps the {cur(f.minbal)} minimum protected.")
    if c["method"] == "partial_payment":
        a1, a2 = c["plan"][0][1], c["plan"][1][1]
        d2 = c["plan"][1][0].isoformat()
        return (f"Pay {cur(a1)} today and the remaining {cur(a2)} on {d2}. This completes the "
                f"full request and keeps the {cur(f.minbal)} minimum protected.")
    if c["method"] == "installments":
        n = len(c["plan"])
        amt = c["plan"][0][1]
        d1 = c["plan"][0][0].isoformat()
        return (f"Use {n} installments of {cur(amt)}, starting {d1}. This leaves at least "
                f"{cur(f.minbal)} available.")
    if c["method"] == "wait":
        d = c["plan"][0][0].isoformat()
        return (f"Pay {cur(c['total'])} in full on {d}. Paying earlier would take the balance "
                f"below the {cur(f.minbal)} minimum.")
    return f"Recommendation: {status}."


# ----------------------------------------------------------------------- main
def main():
    store = Store()
    reqs = store.requests
    if len(sys.argv) > 1 and sys.argv[1] == "--samples":
        reqs = load_csv("sample_requests.csv")
    out_rows = []
    for req in reqs:
        try:
            out_rows.append(evaluate(store, req))
        except Exception as ex:  # never fail a whole run
            out_rows.append({
                "request_id": req["request_id"], "amount_safe_to_pay": "0",
                "affordability_status": "not_affordable", "recommended_payment_method": "not_recommended",
                "payment_plan": "none", "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": f"Evaluation error: {ex}",
            })
    cols = ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
            "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]
    out_path = os.path.join(ROOT, "output.csv")
    if len(sys.argv) > 1 and sys.argv[1] == "--samples":
        out_path = os.path.join(ROOT, "output_samples.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
