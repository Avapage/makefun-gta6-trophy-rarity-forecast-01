#!/usr/bin/env python3
"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Every *.in case in $MAKEFUN_INPUT_DIR (default /input) is answered with
a <case id>.out file in $MAKEFUN_OUTPUT_DIR (default /output) holding
five quantiles (5/25/50/75/95) of each trophy's earn rate at the
case's target moment.

How the forecast is built.  A rate is earners over tracked players, so
the two parts are projected separately along log-time
x = ln(1 + hours since release).  Each part's growth slope averages a
whole-history least-squares fit with a fit over only the segment that
follows the most recent level jump; that keeps the dead-flat
launch-night stretch from diluting the growth that begins with the
first refresh of the tracked page.  Earner slopes (fitted on
ln(earners + 0.5)) are pulled toward the pooled slope of the trophy's
tier in proportion to the earner count behind them, and are damped as
the horizon grows.  The decomposition forecast is then averaged, on
the logit scale, with a shrunk logit-rate trend and a 20 percent
anchor on the last observed rate, and the median may move at most
3 logits from that rate.  When the history shows no refresh yet --
completely flat, or unchanged for most of its span -- a generative
one-refresh model takes over instead: the one documented refresh
multiplied tracked players by 6.06 while a typical trophy's earners
multiplied by 1.34 (the platinum's earners cannot grow, and a trophy
already at ~100 percent stays there), so a rate lands near r * g / G
and the spread is that refresh's own noise: cohort dispersion, the
Poisson noise of the new earners, and uncertainty in the jump sizes.
A case with no reads at all falls back to each tier's launch-night
centre.  Pure standard library, fully deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# The five forecast quantiles and their standard-normal scores.
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

LOGIT_FLOOR_PCT = 0.01   # rates are clamped inside (0, 100) on the logit scale
LOGIT_CEIL_PCT = 99.99
SATURATED_PCT = 99.99    # a trophy already at ~100% is forecast to stay there

# The documented first refresh of the tracked page: tracked players
# went 638 -> 3,868 (a factor of 6.06) while a typical trophy's
# earners grew by a factor of 1.34 over the same step.
PLAYERS_JUMP = 6.06
EARNERS_JUMP = 1.34
COHORT_SD = 0.38         # per-trophy growth scatter around the typical factor
JUMP_SD = 0.20           # uncertainty in the two jump factors themselves

# Launch-night centre of each tier, used only when a case has no reads.
EMPTY_HISTORY_CENTRE = {"platinum": 0.63, "gold": 0.94,
                        "silver": 1.10, "bronze": 1.10}


def stamp_seconds(text):
    """Epoch seconds for an ISO-8601 date/datetime string, else None."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def to_logit(pct):
    """Log-odds of a percentage rate, clamped inside (0, 100) first."""
    share = min(max(pct, LOGIT_FLOOR_PCT), LOGIT_CEIL_PCT) / 100.0
    return math.log(share / (1.0 - share))


def from_logit(value):
    """Inverse of to_logit: a percentage inside [0, 100]."""
    if value >= 0:
        share = 1.0 / (1.0 + math.exp(-value))
    else:
        e = math.exp(value)
        share = e / (1.0 + e)
    return min(max(share * 100.0, 0.0), 100.0)


def fit_line(xs, ys):
    """Least-squares fit y ~ intercept + slope * x, returned as a pair."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        sxx += dx ** 2
        sxy += dx * (y - mean_y)
    if sxx <= 1e-12:
        return mean_y, 0.0
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def jump_indices(values):
    """Positions i >= 1 where the series steps by over 0.35 (log space)."""
    return [i for i in range(1, len(values))
            if abs(values[i] - values[i - 1]) > 0.35]


def tail_start(xs, ys, need=3):
    """Index where the current, post-jump segment of a series begins.

    The latest jump's segment wins when it holds at least `need`
    points.  Otherwise the last quarter of the series is used when
    that is long enough, and very short series start at 0.
    """
    n = len(xs)
    if n < 2:
        return 0
    jumps = jump_indices(ys)
    if jumps and n - jumps[-1] >= need:
        return jumps[-1]
    quarter = max(n // 4, 1)
    if quarter >= need or n <= need:
        return n - quarter if n > need else 0
    return 0


def mixed_slope(xs, ys, start):
    """Average of the whole-history slope and the latest segment's."""
    _, whole = fit_line(xs, ys)
    if len(xs) - start >= 3:
        _, recent = fit_line(xs[start:], ys[start:])
    elif len(xs) >= 2:
        k = max(len(xs) // 4, 1)
        _, recent = fit_line(xs[-k:], ys[-k:])
    else:
        recent = whole
    return 0.5 * whole + 0.5 * recent


def prepare(case):
    """Extract everything the forecast needs from a raw case dict."""
    trophies = sorted(case.get("trophies", []), key=lambda t: t["id"])
    horizon_min = float(case.get("horizon_minutes") or 0)

    release = stamp_seconds((case.get("game") or {}).get("release_date"))
    if release is None:
        release = stamp_seconds(case.get("at")) or 0.0
    at_ts = stamp_seconds(case.get("at"))
    target_ts = stamp_seconds(case.get("target"))
    if target_ts is None and at_ts is not None:
        target_ts = at_ts + horizon_min * 60.0

    def log_time(ts):
        if ts is None:
            return 0.0
        return math.log1p(max((ts - release) / 3600.0, 0.0))

    xs, players, grid = [], [], []
    for read in case.get("history", []) or []:
        ts = stamp_seconds(read.get("at")) or stamp_seconds(read.get("slot"))
        if ts is None:
            continue
        xs.append(log_time(ts))
        tracked = read.get("players_tracked")
        players.append(float(tracked) if tracked else 0.0)
        grid.append(list(read.get("rates") or []))

    # Most recent observed rate per trophy, scanning reads backwards.
    last = [None] * len(trophies)
    for row in reversed(grid):
        for i in range(len(trophies)):
            if last[i] is None and i < len(row) and row[i] is not None:
                last[i] = float(row[i])
        if all(v is not None for v in last):
            break
    last = [0.0 if v is None else v for v in last]
    if not xs:
        # No reads at all: each trophy starts from its tier's centre.
        last = [EMPTY_HISTORY_CENTRE.get(t.get("tier", "bronze"), 1.10)
                for t in trophies]

    return {
        "trophies": trophies,
        "tiers": [t.get("tier", "bronze") for t in trophies],
        "horizon_days": horizon_min / 1440.0,
        "x_target": log_time(target_ts),
        "xs": xs,
        "players": players,
        "grid": grid,
        "last_rates": last,
    }


def stagnation(xs, players, grid, n_tr):
    """(flat, stale) flags for the history.

    Flat: neither the player count nor any rate ever moved.  Stale:
    something did move, but the last change sits in the first 40
    percent of the observed span, so a refresh is still unseen.
    """
    n = len(xs)
    if n < 2:
        return False, False
    rates_moved = False
    for i in range(n_tr):
        seen = [row[i] for row in grid
                if i < len(row) and row[i] is not None]
        if seen and max(seen) - min(seen) > 1e-9:
            rates_moved = True
            break
    flat = (max(players) - min(players)) <= 1e-9 and not rates_moved
    if flat or n < 3:
        return flat, False
    last_change = 0
    for k in range(1, n):
        changed = abs(players[k] - players[k - 1]) > 1e-9
        if not changed:
            prev, cur = grid[k - 1], grid[k]
            for i in range(n_tr):
                a = prev[i] if i < len(prev) else None
                b = cur[i] if i < len(cur) else None
                if a is not None and b is not None and abs(a - b) > 1e-9:
                    changed = True
                    break
        if changed:
            last_change = k
    span = max(xs[-1] - xs[0], 1e-9)
    stale = (xs[-1] - xs[last_change]) > 0.6 * span and n >= 4
    return flat, stale


def project_players(xs, players, x_target):
    """Project the tracked-player count to the target log-time."""
    if not players:
        return 0.0
    last = players[-1]
    if len(xs) < 2 or last <= 0:
        return last
    log_players = [math.log(max(p, 1.0)) for p in players]
    start = tail_start(xs, log_players)
    slope = min(max(mixed_slope(xs, log_players, start), -0.5), 3.0)
    gap = max(x_target - xs[-1], 0.0)
    growth = min(max(math.exp(slope * gap), 1.0), 40.0)
    return last * growth


def trophy_fits(xs, players, grid, n_tr):
    """Per-trophy (earner slope, logit slope, residual sd) from the reads."""
    earner_slopes = [0.0] * n_tr
    logit_slopes = [0.0] * n_tr
    resid_sd = [0.0] * n_tr
    for i in range(n_tr):
        sx, se, sl = [], [], []
        for k in range(len(xs)):
            row = grid[k]
            if i >= len(row) or row[i] is None:
                continue
            rate = float(row[i])
            earners = rate / 100.0 * players[k]
            sx.append(xs[k])
            se.append(math.log(earners + 0.5))
            sl.append(to_logit(rate))
        if len(sx) < 2:
            continue
        seg = tail_start(sx, se)
        earner_slopes[i] = mixed_slope(sx, se, seg)
        head, slope_all = fit_line(sx, sl)
        if len(sx) - seg >= 3:
            head_t, slope_tail = fit_line(sx[seg:], sl[seg:])
            logit_slopes[i] = 0.5 * slope_all + 0.5 * slope_tail
            resid = [y - (head_t + slope_tail * x)
                     for x, y in zip(sx[seg:], sl[seg:])]
        else:
            logit_slopes[i] = slope_all
            resid = [y - (head + slope_all * x) for x, y in zip(sx, sl)]
        if len(resid) >= 3:
            centre = sum(resid) / len(resid)
            var = sum((v - centre) ** 2 for v in resid) / max(len(resid) - 2, 1)
            resid_sd[i] = math.sqrt(max(var, 0.0))
    return earner_slopes, logit_slopes, resid_sd


def pool_by_tier(tiers, last_rates, p_last, earner_slopes, logit_slopes):
    """Earner-count-weighted average slopes within each trophy tier."""
    pooled_e = {}
    pooled_l = {}
    for tier in set(tiers):
        w_total = 0.0
        e_total = 0.0
        l_total = 0.0
        for i, t in enumerate(tiers):
            if t != tier:
                continue
            w = max(last_rates[i] / 100.0 * max(p_last, 1.0), 0.0) + 1.0
            w_total += w
            e_total += w * earner_slopes[i]
            l_total += w * logit_slopes[i]
        pooled_e[tier] = e_total / w_total if w_total > 0 else 0.0
        pooled_l[tier] = l_total / w_total if w_total > 0 else 0.0
    return pooled_e, pooled_l


def forecast_case(case):
    prep = prepare(case)
    trophies = prep["trophies"]
    tiers = prep["tiers"]
    xs = prep["xs"]
    players = prep["players"]
    grid = prep["grid"]
    last_rates = prep["last_rates"]
    n_tr = len(trophies)

    flat, stale = stagnation(xs, players, grid, n_tr)
    p_last = players[-1] if players else 0.0
    p_hat = project_players(xs, players, prep["x_target"])
    earner_slopes, logit_slopes, resid_sd = trophy_fits(
        xs, players, grid, n_tr)
    pooled_e, pooled_l = pool_by_tier(
        tiers, last_rates, p_last, earner_slopes, logit_slopes)

    gap = max(prep["x_target"] - xs[-1], 0.0) if xs else 0.0
    damp = 1.0 / (1.0 + 0.4 * gap)   # far-future earner growth flattens
    root_days = math.sqrt(max(prep["horizon_days"], 0.0))

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = last_rates[i]
        tier = tiers[i]

        # A trophy everyone tracked already holds stays at ~100%.
        if rate_last >= SATURATED_PCT:
            rows.append({"id": trophy["id"],
                         "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue

        # Decomposition: grow the earner count with the shrunk slope,
        # then divide by the projected player count.
        earners_last = rate_last / 100.0 * max(p_last, 1.0)
        own = earners_last / (earners_last + 20.0)
        slope_e = own * earner_slopes[i] + (1.0 - own) * pooled_e.get(tier, 0.0)
        slope_e = min(max(slope_e * damp, -2.0), 4.0)
        earners_hat = math.exp(math.log(earners_last + 0.5)
                               + slope_e * gap) - 0.5
        earners_hat = max(earners_hat, 0.0)
        if p_hat > 0:
            earners_hat = min(earners_hat, p_hat)
        rate_decomp = 100.0 * earners_hat / p_hat if p_hat > 0 else rate_last

        # Trend: the logit rate itself, slope shrunk halfway to the pool.
        slope_l = 0.5 * logit_slopes[i] + 0.5 * pooled_l.get(tier, 0.0)
        slope_l = min(max(slope_l, -3.0), 3.0)
        rate_trend = from_logit(to_logit(rate_last) + slope_l * gap)

        # Combine the two, anchor 20 percent on the last rate, and cap
        # how far the median may travel from it.
        last_logit = to_logit(rate_last)
        centre = 0.5 * to_logit(rate_decomp) + 0.5 * to_logit(rate_trend)
        centre = 0.8 * centre + 0.2 * last_logit
        centre = min(max(centre, last_logit - 3.0), last_logit + 3.0)

        # Flat/stale histories are really asking about the first unseen
        # refresh, so the generative refresh model supplies the centre:
        # players jump by PLAYERS_JUMP, earners by EARNERS_JUMP (the
        # platinum's earners stay put), and the result never shades the
        # median upward.
        if flat or stale:
            weight = 1.0 if flat else 0.5
            gain = 1.0 if tier == "platinum" else EARNERS_JUMP
            refreshed = rate_last * gain / PLAYERS_JUMP
            gen = to_logit(refreshed) if refreshed > 0 else last_logit
            centre = weight * gen + (1.0 - weight) * centre
            centre = min(centre, last_logit)

        # Spread around the centre, asymmetric on the logit scale.
        resid = min(max(resid_sd[i], 0.05), 1.0)
        walk = 0.5 * root_days
        if flat or stale:
            if tier != "platinum":
                lam = max((EARNERS_JUMP - 1.0) * earners_last, 0.0)
                grown = max(EARNERS_JUMP * earners_last, 1.0)
            else:
                lam = 0.0
                grown = max(earners_last, 1.0)
            spread = math.sqrt(COHORT_SD * COHORT_SD
                               + lam / (grown * grown)
                               + JUMP_SD * JUMP_SD)
            if stale and not flat:
                spread *= 1.3
            spread_lo = spread
            spread_hi = spread
        else:
            dil_lo = 0.25 * root_days
            dil_hi = 0.15 * root_days
            spread_lo = math.sqrt(walk * walk + resid * resid
                                  + dil_lo * dil_lo)
            spread_hi = math.sqrt(walk * walk + resid * resid
                                  + dil_hi * dil_hi)

        quantiles = []
        for z in Z_SCORES:
            spread = spread_lo if z < 0 else spread_hi
            quantiles.append(round(from_logit(centre + z * spread), 4))
        quantiles = [min(max(v, 0.0), 100.0) for v in quantiles]
        for j in range(1, 5):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def run():
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
        result = forecast_case(case)
        out_name = "%s.out" % case.get("id", os.path.splitext(name)[0])
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(run())
