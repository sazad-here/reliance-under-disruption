"""
Fit the simulator's parameters from real open data.

This is the step that turns the twin from "a supply chain I made up" into
"a supply chain calibrated against 99,441 real orders". Until this runs, every
number in TwinParams is a guess, and so is any accuracy PatchTST reports.

    Olist  ->  lead-time delay distribution (delay_pmf)
    DataCo ->  disruption base rate
    M5     ->  demand mean, variance, weekly seasonality

Only the STATISTICS cross over. No real row is ever shown to a participant --
the trials are generated, because only a generative process gives us a
ground-truth correct answer for every period (plan section 04).

Run:  python src/fit_params.py
Out:  data/processed/fitted_params.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "processed", "fitted_params.json")

PERIOD_DAYS = 7  # one simulator period = one week


# ----------------------------------------------------------------- Olist
MAX_LEAD_DAYS = 60  # drop beyond this: abandoned or mis-recorded, not variability


def fit_lead_time():
    """Lead-time distribution from REALISED purchase-to-delivery times.

    DEPARTS FROM PLAN SECTION 04, deliberately and with the supervisor's
    agreement. The plan says to fit the delay from "actual minus estimated"
    timestamps. Checked against the data, that measure is unusable here:

        91.9% of Olist orders arrive EARLY
        median gap = -11.9 days

    Olist pads its delivery promises by about two weeks, so actual-minus-
    estimated measures FORECAST CONSERVATISM, not delivery variability. Fitting
    it that way gives delay_pmf = [0.919, 0.046, 0.019, 0.016] -- a near-
    deterministic lead time, which would flatten the disruptions and make the
    task trivially easy.

    We therefore fit the dispersion of the realised lead time itself, which is
    what the simulator actually needs. State this in the method section.

    Preprocessing applied:
      - delivered orders only
      - drop rows with a missing delivery timestamp (8 rows)
      - drop lead times <= 0 or > 60 days (0.32%, up to 209 days)
    """
    path = os.path.join(RAW, "olist", "olist_orders_dataset.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(
        path,
        parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date",
                     "order_purchase_timestamp"],
    )
    n_all = len(df)
    df = df[df.order_status == "delivered"]
    n_delivered = len(df)
    df = df.dropna(subset=["order_delivered_customer_date",
                           "order_estimated_delivery_date"])
    n_dated = len(df)

    lead_days = (
        df.order_delivered_customer_date - df.order_purchase_timestamp
    ).dt.total_seconds() / 86400.0
    gap_days = (
        df.order_delivered_customer_date - df.order_estimated_delivery_date
    ).dt.total_seconds() / 86400.0

    keep = (lead_days > 0) & (lead_days <= MAX_LEAD_DAYS)
    lead_clean = lead_days[keep]
    n_clean = int(keep.sum())

    weeks = lead_clean / PERIOD_DAYS
    total_periods = np.maximum(1, np.round(weeks)).astype(int)
    nominal = int(total_periods.min())
    extra = (total_periods - nominal).clip(0, 3)
    pmf = [float((extra == k).mean()) for k in range(4)]
    pmf = [p / sum(pmf) for p in pmf]

    # the comparison measure, reported so the choice above is auditable
    rejected = gap_days[keep].clip(lower=0)
    rej_extra = np.ceil(rejected / PERIOD_DAYS).clip(upper=3)
    rej_pmf = [round(float((rej_extra == k).mean()), 4) for k in range(4)]

    return {
        "measure": "realised purchase-to-delivery lead time (NOT actual-minus-estimated)",
        "preprocessing": {
            "rows_total": n_all,
            "rows_delivered": n_delivered,
            "rows_with_dates": n_dated,
            "rows_after_outlier_cut": n_clean,
            "pct_removed": round(100 * (1 - n_clean / n_dated), 2),
            "max_lead_days": MAX_LEAD_DAYS,
        },
        "lead_time_nominal": nominal,
        "delay_pmf": [round(p, 4) for p in pmf],
        "mean_lead_periods": round(float(total_periods.mean()), 3),
        "sd_lead_periods": round(float(total_periods.std()), 3),
        "delay_variance": round(float(np.var(extra)), 4),
        "lead_days_mean": round(float(lead_clean.mean()), 2),
        "lead_days_median": round(float(lead_clean.median()), 2),
        "_rejected_measure": {
            "note": "what plan section 04 asked for; kept for the audit trail",
            "pct_arriving_early": round(float((gap_days < 0).mean() * 100), 1),
            "median_gap_days": round(float(gap_days.median()), 2),
            "delay_pmf_would_be": rej_pmf,
        },
    }


# ---------------------------------------------------------------- DataCo
def fit_disruption_rate():
    path = os.path.join(RAW, "dataco", "DataCoSupplyChainDataset.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(
        path, encoding="latin-1",
        usecols=["Days for shipping (real)", "Days for shipment (scheduled)",
                 "Late_delivery_risk"],
    )
    real = df["Days for shipping (real)"]
    sched = df["Days for shipment (scheduled)"]
    slip = (real - sched).clip(lower=0)

    return {
        "n_rows": int(len(df)),
        "late_delivery_rate": round(float(df["Late_delivery_risk"].mean()), 4),
        "mean_slip_days": round(float(slip.mean()), 2),
        "severe_slip_rate": round(float((slip >= 2).mean()), 4),
        # a "disruption" in our sense is a SHOCK, not routine lateness: use the
        # severe-slip rate, scaled to per-period shock frequency.
        "disruption_rate_per_period": round(float((slip >= 3).mean()), 4),
    }


# -------------------------------------------------------------------- M5
def fit_demand():
    d = os.path.join(RAW, "m5")
    path = os.path.join(d, "sales_train_validation.csv")
    if not os.path.exists(path):
        alt = os.path.join(d, "sales_train_evaluation.csv")
        path = alt if os.path.exists(alt) else None
    if not path:
        return None

    df = pd.read_csv(path)
    day_cols = [c for c in df.columns if c.startswith("d_")]

    # Aggregate to a store-department series -- a single SKU is too intermittent
    # to be the demand process for one warehouse.
    grouped = df.groupby(["store_id", "dept_id"])[day_cols].sum()
    series = grouped.values.astype(np.float64)

    # daily -> weekly, dropping any ragged tail
    n_weeks = series.shape[1] // PERIOD_DAYS
    weekly = series[:, : n_weeks * PERIOD_DAYS].reshape(
        series.shape[0], n_weeks, PERIOD_DAYS
    ).sum(axis=2)

    # scale each series to a comparable warehouse size, then pool
    scaled = weekly / weekly.mean(axis=1, keepdims=True)
    pooled_cv = float(scaled.std(axis=1).mean())

    # weekly seasonality from day-of-week shares of the daily series
    daily_pooled = series.sum(axis=0)
    n_days = (len(daily_pooled) // PERIOD_DAYS) * PERIOD_DAYS
    dow = daily_pooled[:n_days].reshape(-1, PERIOD_DAYS).mean(axis=0)
    seasonality = (dow / dow.mean()).round(3).tolist()

    target_mean = 140.0  # warehouse scale; CV and seasonality are what we take
    return {
        "n_series": int(series.shape[0]),
        "n_weeks": int(n_weeks),
        "coefficient_of_variation": round(pooled_cv, 4),
        "demand_mean": target_mean,
        "demand_sd": round(target_mean * pooled_cv, 2),
        "seasonality": seasonality,
    }


def main():
    print("=" * 70)
    print("FITTING SIMULATOR PARAMETERS FROM REAL DATA")
    print("=" * 70)

    out = {}

    print("\n--- Olist: lead-time delay ---")
    olist = fit_lead_time()
    if olist:
        out["olist"] = olist
        pp = olist["preprocessing"]
        print("  rows: %d total -> %d delivered -> %d dated -> %d after outlier cut (%.2f%% removed)"
              % (pp["rows_total"], pp["rows_delivered"], pp["rows_with_dates"],
                 pp["rows_after_outlier_cut"], pp["pct_removed"]))
        print("  measure: %s" % olist["measure"])
        print("  lead time: mean %.2f d, median %.2f d -> nominal %d period(s)"
              % (olist["lead_days_mean"], olist["lead_days_median"], olist["lead_time_nominal"]))
        print("  FITTED delay_pmf:       %s   (variance %.3f)"
              % (olist["delay_pmf"], olist["delay_variance"]))
        print("  my placeholder was:     [0.7, 0.2, 0.08, 0.02]   (variance 0.524)")
        rj = olist["_rejected_measure"]
        print("  [rejected] plan s04 measure: %.1f%% arrive early, median gap %+.1f d"
              % (rj["pct_arriving_early"], rj["median_gap_days"]))
        print("             would have given: %s" % rj["delay_pmf_would_be"])
    else:
        print("  NOT FOUND -- keeping placeholder")

    print("\n--- DataCo: disruption rate ---")
    dataco = fit_disruption_rate()
    if dataco:
        out["dataco"] = dataco
        print("  rows analysed:          %d" % dataco["n_rows"])
        print("  late_delivery_risk:     %.1f%%" % (100 * dataco["late_delivery_rate"]))
        print("  mean schedule slip:     %.2f days" % dataco["mean_slip_days"])
        print("  FITTED disruption rate: %.1f%% of periods"
              % (100 * dataco["disruption_rate_per_period"]))
        print("  (my placeholder was:    6.0%%)")
    else:
        print("  NOT FOUND -- keeping placeholder")

    print("\n--- M5: demand process ---")
    m5 = fit_demand()
    if m5:
        out["m5"] = m5
        print("  series analysed:        %d store-departments over %d weeks"
              % (m5["n_series"], m5["n_weeks"]))
        print("  coefficient of variation: %.3f" % m5["coefficient_of_variation"])
        print("  FITTED demand_sd:       %.1f (mean %.0f)" % (m5["demand_sd"], m5["demand_mean"]))
        print("  (my placeholder was:    35.0 (mean 140))")
        print("  FITTED seasonality:     %s" % m5["seasonality"])
    else:
        print("  NOT FOUND -- accept the competition rules, then:")
        print("    python -m kaggle competitions download -c m5-forecasting-accuracy -p data/raw/m5")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n  written -> %s" % os.path.relpath(OUT))

    if olist and dataco and m5:
        print("\n  All three fitted. Next: regenerate the corpus and re-run calibration.")
    else:
        missing = [n for n, v in (("Olist", olist), ("DataCo", dataco), ("M5", m5)) if not v]
        print("\n  Still missing: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
