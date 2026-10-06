"""
Simulation-based power for the pre-registered design (OSF open question 2).

Effect sizes are unknown before data, so power is reported ACROSS a range of
plausible effects rather than for one assumed value. The analyses here are
participant-level approximations of the registered mixed models (a mean or a
paired difference per person, then a t-test). They discard within-person
information the mixed model uses, so they are conservative.

H1  Weight of Advice, 80 % vs 60 %: two-sample test on each participant's mean
    WoA over 30 trials; 35 per cell.
H3  Reliance on the trial AFTER an AI error, propagating vs absorbed: paired
    difference per participant. The error counts come from the real v2
    schedule (80 %: 3 + 3 errors; 60 %: 6 + 6), so the 80 % cell estimates
    each person's difference from only 3 + 3 trials.

Assumptions (stated, varied where they matter): trial-level WoA noise SD 0.30
or 0.40 (WoA spans roughly 0-1); a person's own H3 effect varies around the
mean with SD 0.10; alpha 0.05 two-sided; Holm across the primaries would
lower alpha for all but the smallest p, so power at alpha 0.025 is also shown.

Run:  python src/power_analysis.py
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

N_PER_CELL, N_TRIALS, REPS = 35, 30, 4000
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "power_analysis.json")
rng = np.random.default_rng(2026)


def power_h1(d, alpha):
    """Analytic power, two-sample t-test on participant means, effect d."""
    df = 2 * N_PER_CELL - 2
    nc = d * np.sqrt(N_PER_CELL / 2)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc))


def power_h3(delta, sd_trial, alpha, sd_slope=0.10):
    """Monte Carlo: paired t on each participant's after-error WoA difference.

    Propagating errors are hypothesised to REDUCE the next trial's WoA by
    `delta` relative to absorbed errors; each person's own effect varies
    around `delta` with SD `sd_slope`.
    """
    hits = 0
    for _ in range(REPS):
        parts = []
        for k in (3, 6):                                  # errors per placement: 80 % cell, 60 % cell
            slope = delta + rng.normal(0, sd_slope, N_PER_CELL)
            after_abs = rng.normal(0, sd_trial, (N_PER_CELL, k)).mean(1)
            after_prop = (rng.normal(0, sd_trial, (N_PER_CELL, k)) - slope[:, None]).mean(1)
            parts.append(after_abs - after_prop)
        hits += stats.ttest_1samp(np.concatenate(parts), 0).pvalue < alpha
    return hits / REPS


def main():
    res = {"h1": {}, "h3": {}}
    print("H1  WoA 80 %% vs 60 %%, %d per cell (participant-mean t-test)" % N_PER_CELL)
    print("    d      power a=.05   a=.025")
    for d in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        p5, p25 = power_h1(d, 0.05), power_h1(d, 0.025)
        res["h1"][str(d)] = {"alpha_05": p5, "alpha_025": p25}
        print("    %.1f    %.2f         %.2f" % (d, p5, p25))
    print("\nH3  after-error WoA, propagating vs absorbed (paired, n = %d)" % (2 * N_PER_CELL))
    print("    effect (WoA)  noise SD   power a=.05   a=.025")
    for sd in (0.30, 0.40):
        for delta in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
            p5, p25 = power_h3(delta, sd, 0.05), power_h3(delta, sd, 0.025)
            res["h3"]["%.2f/sd%.2f" % (delta, sd)] = {"alpha_05": p5, "alpha_025": p25}
            print("    %.2f          %.2f       %.2f          %.2f" % (delta, sd, p5, p25))
    json.dump(res, open(OUT, "w"), indent=2)
    print("\nwritten ->", OUT)


if __name__ == "__main__":
    main()
