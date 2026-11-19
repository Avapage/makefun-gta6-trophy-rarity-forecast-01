# GTA 6 Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Avapage/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Avapage/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01

A deterministic Python 3.13 (stdlib-only) forecaster for GTA 6 trophy rarity.
It reads `makefun/gta6-trophies-case/v1` cases and forecasts the 5 quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) of each trophy's earn rate (percent of
players, 0–100) at the requested horizon.

Model: decompose each trophy's rate into earners / players, forecast both on
log-hours since release across change points, with tier-pooled slope
shrinkage, horizon damping, a persistence blend, and a refresh/dilution prior
for flat or stale histories. See `main.py` for the full method.

## Usage

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Each `<id>.out` is exactly `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}` —
55 rows, ids 1–55, non-decreasing quantiles.
