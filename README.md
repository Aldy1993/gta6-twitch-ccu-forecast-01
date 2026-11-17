# GTA VI launch-day Twitch CCU forecaster

Entry for make.fun competition `gta6-twitch-ccu-forecast-01` — forecast
Twitch concurrent viewers in the "Grand Theft Auto VI" category for hours
2–24 UTC of release day, 19 November 2026, as five quantiles each,
given hour 1's print.

## How it works

`main.py` (Python 3.13, standard library only, offline, deterministic)
reads the case file and, for each hour, fits a size-weighted regression
of `log(viewers_h / viewers_1)` on `log(viewers_1)` across the case's
own `comparables` — past big launches' hourly Twitch curves. Launch-day
shape is common across games, but how much the day grows depends
strongly on how big hour 1 already is (correlation about −0.9 late in
the day), so the regression personalises the curve to the launch. The
median is shrunk 15% toward the pooled median; the spread is the
regression's residual scale, halved.

## Validation

Practice answers were reconstructed from the same public SullyGnome
source the kit cites (all four hour-1 prints match the cases exactly):

| program | practice pinball loss | typical miss | 90% coverage |
|---|---|---|---|
| house baseline (starter) | 0.387 | 82.7% | 0.783 |
| this program | **0.170** | 29.9% | 0.989 |

Leave-one-out over the 37 comparables (big launches): 0.271 vs the
baseline's 0.295. Runs in ~0.1 s per case, byte-identical across runs.
