#!/usr/bin/env python3
"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output): five quantiles of each
trophy's earn rate at the case target.

Model: decompose each trophy's rate as earners / players and forecast the
two components separately on log-time, x = ln(1 + hours since release),
fitting over change points: each component's slope blends its
full-history least-squares slope with the slope of the latest segment
after a level jump, so the flat launch-night stretch does not attenuate
the growth seen after the first refresh. Earners per trophy by least squares on
ln(earners + 0.5), each slope shrunk toward its tier's pooled slope,
weighted by earner count and damped with horizon. The two forecasts are
combined into a rate, blended with a shrunk logit-rate trend and 20%
persistence toward the last observed rate, with the median move capped.
When history is flat or stale (no refresh seen yet) a generative
refresh model applies: the first documented refresh took players
638 -> 3,868 (x6.06) while a typical trophy's earners grew x1.34, so
a rate moves to r x 1.34 / 6.06 on the refreshed page (the platinum's
earners stay put); the spread is that refresh's own noise -- cohort
dispersion, new-earner Poisson noise and jump-size uncertainty.
Otherwise quantile spreads are asymmetric on the logit scale,
combining a sqrt-horizon random walk, a per-trophy residual term
and dilution terms.

Standard library only. Deterministic. No network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
EPS = 0.01          # logit clamp for rates in percent
MAX_RATE = 99.99
PLATINUM_RATE = 99.99  # a trophy at (essentially) 100% stays there
# The documented first Exophase refresh (case notes / kit): players
# 638 -> 3,868 (x6.06) while a typical trophy's earners grew x1.34.
G_REFRESH = 6.06    # tracked-players jump at one refresh
G_EARN = 1.34       # typical per-trophy earner growth at one refresh
SIG_COHORT = 0.38   # logit sd of per-trophy growth around the typical
SIG_REFRESH = 0.20  # logit sd from the jump sizes themselves


def parse_time(s):
    """Parse an ISO-8601 UTC timestamp (date or datetime) to epoch seconds."""
    if s is None:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def logit_pct(r):
    """logit of a rate given in percent, clamped to [0.01, 99.99]."""
    p = min(max(r, EPS), MAX_RATE) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    """Inverse logit, returning a percent in [0, 100]."""
    if l >= 0:
        p = 1.0 / (1.0 + math.exp(-l))
    else:
        e = math.exp(l)
        p = e / (1.0 + e)
    return min(max(p * 100.0, 0.0), 100.0)


def lsq_slope_intercept(xs, ys):
    """Ordinary least squares line y ~ a + b*x. Returns (a, b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def change_points(xs_series):
    """Indices where a series jumps: |diff| > 0.35 in log space.

    Returns a list of indices i (1..n-1) where a new segment starts.
    """
    pts = []
    for i in range(1, len(xs_series)):
        if abs(xs_series[i] - xs_series[i - 1]) > 0.35:
            pts.append(i)
    return pts


def segment_start(xs, ys, min_points=3):
    """Start index of the latest segment after a change point.

    Change points are jumps > 0.35 in the series (log space). If the
    latest segment has too few points, fall back to the last quarter of
    the series; if the series is too short, start at 0.
    """
    n = len(xs)
    if n < 2:
        return 0
    cps = change_points(ys)
    if cps and n - cps[-1] >= min_points:
        return cps[-1]
    k = max(n // 4, 1)
    if n - (n - k) >= min_points or n <= min_points:
        return n - k if n > min_points else 0
    return 0


def blended_slope(xs, ys, start):
    """Blend of the full-history LSQ slope and the recent-segment slope."""
    _, slope_full = lsq_slope_intercept(xs, ys)
    if len(xs) - start >= 3:
        _, slope_recent = lsq_slope_intercept(xs[start:], ys[start:])
    elif len(xs) >= 2:
        k = max(len(xs) // 4, 1)
        _, slope_recent = lsq_slope_intercept(xs[-k:], ys[-k:])
    else:
        slope_recent = slope_full
    return 0.5 * slope_full + 0.5 * slope_recent


def forecast_case(case):
    trophies = sorted(case.get("trophies", []), key=lambda t: t["id"])
    history = case.get("history", []) or []
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = horizon_min / 1440.0
    horizon_hours = horizon_min / 60.0

    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = parse_time(case.get("at")) or 0.0
    at_ts = parse_time(case.get("at"))
    target_ts = parse_time(case.get("target"))
    if target_ts is None and at_ts is not None:
        target_ts = at_ts + horizon_min * 60.0

    def x_of(ts):
        if ts is None:
            return 0.0
        hours = max((ts - release) / 3600.0, 0.0)
        return math.log1p(hours)

    x_target = x_of(target_ts)

    # ---- Gather history series -------------------------------------------
    # xs: log-time of each read; players; rates per trophy index.
    xs = []
    players = []
    rates_hist = []  # list per read: list of rates (None where missing)
    for h in history:
        ts = parse_time(h.get("at")) or parse_time(h.get("slot"))
        if ts is None:
            continue
        xs.append(x_of(ts))
        p = h.get("players_tracked")
        players.append(float(p) if p else 0.0)
        rates_hist.append(list(h.get("rates") or []))

    n_reads = len(xs)
    n_tr = len(trophies)

    # Last observed rates (from the last read that has them).
    last_rates = [None] * n_tr
    for rates in reversed(rates_hist):
        for i in range(n_tr):
            if last_rates[i] is None and i < len(rates) and rates[i] is not None:
                last_rates[i] = float(rates[i])
        if all(v is not None for v in last_rates):
            break
    last_rates = [0.0 if v is None else v for v in last_rates]
    if n_reads == 0:
        # No history at all: fall back to a tier prior from the observed
        # launch-night centre of each tier instead of forecasting 0.01%
        # for every trophy.
        tier_prior = {"platinum": 0.63, "gold": 0.94,
                      "silver": 1.10, "bronze": 1.10}
        last_rates = [tier_prior.get(t.get("tier", "bronze"), 1.10)
                      for t in trophies]

    # ---- Flat / stale detection ------------------------------------------
    flat = False
    stale = False
    if n_reads >= 2:
        p_min, p_max = min(players), max(players)
        r_changed = False
        for i in range(n_tr):
            col = [r[i] for r in rates_hist
                   if i < len(r) and r[i] is not None]
            if col and (max(col) - min(col)) > 1e-9:
                r_changed = True
                break
        flat = (p_max - p_min) <= 1e-9 and not r_changed
        if not flat and n_reads >= 3:
            # Stale: the last change happened long ago relative to the
            # history span (no refresh for most of the observed window).
            last_change = 0
            for k in range(1, n_reads):
                changed = abs(players[k] - players[k - 1]) > 1e-9
                if not changed:
                    for i in range(n_tr):
                        a = rates_hist[k - 1][i] if i < len(rates_hist[k - 1]) else None
                        b = rates_hist[k][i] if i < len(rates_hist[k]) else None
                        if a is not None and b is not None and abs(a - b) > 1e-9:
                            changed = True
                            break
                if changed:
                    last_change = k
            span = max(xs[-1] - xs[0], 1e-9)
            if (xs[-1] - xs[last_change]) > 0.6 * span and n_reads >= 4:
                stale = True

    # Expected refreshes before the target (flat/stale branch only).
    n_refresh = min(max(horizon_hours / 6.0, 1.0), 3.0) if (flat or stale) else 0.0
    dilution_factor = 0.22 ** n_refresh if (flat or stale) else 1.0

    # ---- Players forecast -------------------------------------------------
    p_last = players[-1] if players else 0.0
    p_hat = p_last
    if n_reads >= 2 and p_last > 0:
        ln_p = [math.log(max(p, 1.0)) for p in players]
        # Blend the full-history slope with the recent (post-change-point)
        # slope so a level jump does not dominate the growth estimate.
        start = segment_start(xs, ln_p)
        slope_p = blended_slope(xs, ln_p, start)
        slope_p = min(max(slope_p, -0.5), 3.0)
        dx = max(x_target - xs[-1], 0.0)
        growth = math.exp(slope_p * dx)
        growth = min(max(growth, 1.0), 40.0)
        p_hat = p_last * growth

    # ---- Per-trophy earner + logit-trend fits ----------------------------
    tiers = [t.get("tier", "bronze") for t in trophies]

    # Tier-pooled earner slopes and logit slopes (earner-weighted).
    earner_slopes = [0.0] * n_tr
    logit_slopes = [0.0] * n_tr
    residuals = [0.0] * n_tr
    for i in range(n_tr):
        pts_x, pts_e, pts_l = [], [], []
        for k in range(n_reads):
            if i >= len(rates_hist[k]) or rates_hist[k][i] is None:
                continue
            r = float(rates_hist[k][i])
            e = r / 100.0 * players[k]
            pts_x.append(xs[k])
            pts_e.append(math.log(e + 0.5))
            pts_l.append(logit_pct(r))
        if len(pts_x) >= 2:
            # Fit over change points: blend the full-history slope with
            # the recent-segment slope so the flat launch-night stretch
            # does not attenuate the growth seen after the first refresh.
            seg = segment_start(pts_x, pts_e)
            b_e = blended_slope(pts_x, pts_e, seg)
            a_l, b_l = lsq_slope_intercept(pts_x, pts_l)
            if len(pts_x) - seg >= 3:
                a_lr, b_lr = lsq_slope_intercept(
                    pts_x[seg:], pts_l[seg:])
                b_l = 0.5 * b_l + 0.5 * b_lr
                a_l = a_lr
                res = [y - (a_lr + b_lr * x)
                       for x, y in zip(pts_x[seg:], pts_l[seg:])]
            else:
                res = [y - (a_l + b_l * x) for x, y in zip(pts_x, pts_l)]
            earner_slopes[i] = b_e
            logit_slopes[i] = b_l
            if len(res) >= 3:
                mean_res = sum(res) / len(res)
                var = sum((v - mean_res) ** 2 for v in res) / max(len(res) - 2, 1)
                residuals[i] = math.sqrt(max(var, 0.0))
        elif len(pts_x) == 1:
            earner_slopes[i] = 0.0
            logit_slopes[i] = 0.0

    # Tier-pooled slopes.
    tier_earner = {}
    tier_logit = {}
    for tier in set(tiers):
        idxs = [i for i in range(n_tr) if tiers[i] == tier]
        w_sum = 0.0
        e_acc = 0.0
        l_acc = 0.0
        for i in idxs:
            w = max(last_rates[i] / 100.0 * max(p_last, 1.0), 0.0) + 1.0
            w_sum += w
            e_acc += w * earner_slopes[i]
            l_acc += w * logit_slopes[i]
        if w_sum > 0:
            tier_earner[tier] = e_acc / w_sum
            tier_logit[tier] = l_acc / w_sum
        else:
            tier_earner[tier] = 0.0
            tier_logit[tier] = 0.0

    # ---- Forecast each trophy --------------------------------------------
    dx = max(x_target - xs[-1], 0.0) if xs else 0.0
    # Horizon damping for earner growth: far-future growth flattens.
    damp = 1.0 / (1.0 + 0.4 * dx)
    sqrt_d = math.sqrt(max(horizon_days, 0.0))

    rows = []
    for i, tr in enumerate(trophies):
        r_last = last_rates[i]
        tier = tiers[i]

        # A trophy everyone tracked already holds stays (essentially) 100%.
        if r_last >= PLATINUM_RATE:
            rows.append({"id": tr["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue

        # Earners forecast: slope shrunk toward the tier pool, weighted by
        # earner count; anchored at the last observed earner count.
        e_last = r_last / 100.0 * max(p_last, 1.0)
        w_own = e_last / (e_last + 20.0)
        slope_e = (w_own * earner_slopes[i]
                   + (1.0 - w_own) * tier_earner.get(tier, 0.0))
        slope_e *= damp
        slope_e = min(max(slope_e, -2.0), 4.0)
        e_hat = math.exp(math.log(e_last + 0.5) + slope_e * dx) - 0.5
        e_hat = max(e_hat, 0.0)
        if p_hat > 0:
            e_hat = min(e_hat, p_hat)
        r_decomp = 100.0 * e_hat / p_hat if p_hat > 0 else r_last

        # Logit-rate trend: slope shrunk halfway toward the tier pool.
        slope_l = 0.5 * logit_slopes[i] + 0.5 * tier_logit.get(tier, 0.0)
        slope_l = min(max(slope_l, -3.0), 3.0)
        r_trend = inv_logit_pct(logit_pct(r_last) + slope_l * dx)

        # Combine decomposition and trend, then 20% persistence to last.
        m_logit = 0.5 * logit_pct(r_decomp) + 0.5 * logit_pct(r_trend)
        m_logit = 0.8 * m_logit + 0.2 * logit_pct(r_last)
        # Cap how far the median may move from the last observed rate.
        last_logit = logit_pct(r_last)
        m_logit = min(max(m_logit, last_logit - 3.0), last_logit + 3.0)

        # Flat / stale branch: an unseen Exophase refresh is the event
        # being forecast. Generative model of one refresh, from the one
        # documented in the case notes (players 638 -> 3,868 while total
        # earners grew 943 -> 4,276): the tracked-player count jumps by
        # G_REFRESH, a typical trophy's earners grow by G_EARN, the
        # everyone-has-it trophy stays ~100% (handled above) and the
        # platinum's earners stay put (nobody new can hold it yet), so a
        # rate moves to r * g / G on the refreshed page. The spread is
        # the refresh's own noise: cohort-preference dispersion, the
        # Poisson noise of the new earners and uncertainty in the jump
        # sizes -- calibrated, roughly symmetric on the logit scale,
        # instead of a wide one-sided dilution band.
        if flat or stale:
            strength = 1.0 if flat else 0.5
            g_i = 1.0 if tier == "platinum" else G_EARN
            r_med = r_last * g_i / G_REFRESH
            gen_logit = logit_pct(r_med) if r_med > 0 else last_logit
            m_logit = strength * gen_logit + (1.0 - strength) * m_logit
            m_logit = min(m_logit, last_logit)  # never shade upward here

        median = inv_logit_pct(m_logit)

        # ---- Spreads (asymmetric, on the logit scale) --------------------
        resid = min(max(residuals[i], 0.05), 1.0)
        rw = 0.5 * sqrt_d
        if flat or stale:
            e_last_i = r_last / 100.0 * max(p_last, 1.0)
            lam_i = max((G_EARN - 1.0) * e_last_i, 0.0) \
                if tier != "platinum" else 0.0
            e_new_i = max(G_EARN * e_last_i, 1.0) \
                if tier != "platinum" else max(e_last_i, 1.0)
            sig_gen = math.sqrt(SIG_COHORT * SIG_COHORT
                                + lam_i / (e_new_i * e_new_i)
                                + SIG_REFRESH * SIG_REFRESH)
            if stale and not flat:
                sig_gen *= 1.3
            sig_lo = sig_gen
            sig_hi = sig_gen
        else:
            dil_lo = 0.25 * sqrt_d
            dil_hi = 0.15 * sqrt_d
            sig_lo = math.sqrt(rw * rw + resid * resid + dil_lo * dil_lo)
            sig_hi = math.sqrt(rw * rw + resid * resid + dil_hi * dil_hi)

        q = []
        for tau, z in zip((0.05, 0.25, 0.5, 0.75, 0.95), Z):
            sig = sig_lo if z < 0 else sig_hi
            q.append(round(inv_logit_pct(m_logit + z * sig), 4))
        # Enforce 0..100 and non-decreasing.
        q = [min(max(v, 0.0), 100.0) for v in q]
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": tr["id"], "q": q})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        out_name = "%s.out" % case.get("id", os.path.splitext(name)[0])
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
