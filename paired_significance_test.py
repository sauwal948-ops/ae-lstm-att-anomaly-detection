#!/usr/bin/env python3
"""
Paired significance test for the AE-LSTM-ATT vs LSTM-AE comparison, per
reviewer comment: "the reported improvement ... is very small ...
justify whether this difference is statistically significant".

Reads results_raw.json (produced by multiseed_eval.py or
multiseed_eval_subtle.py -- same schema either way:
    {model: {anomaly_type: {metric: [value_per_seed, ...]}}}
seeds are paired by POSITION (index 0 of AE-LSTM-ATT's list used the same
seed as index 0 of LSTM-AE's list -- true as long as the JSON came from a
script that loops `for seed in SEEDS: ...` in a fixed order, which both of
your scripts do).

For each anomaly type and metric, runs:
  - Paired t-test (parametric; reported for reference, but under-powered
    and its normality assumption is shaky with only n=5)
  - Wilcoxon signed-rank test (nonparametric; the primary test to report,
    since it doesn't assume normally-distributed differences)
  - Cohen's d for paired samples (effect size, since p-values alone are
    uninformative at n=5)

Usage:
  python paired_significance_test.py results_raw.json
"""
import sys, json
import numpy as np
from scipy import stats

def paired_cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")

def main(path):
    with open(path) as f:
        results = json.load(f)

    models = list(results.keys())
    if "AE-LSTM-ATT" not in models or "LSTM-AE" not in models:
        print(f"Expected 'AE-LSTM-ATT' and 'LSTM-AE' keys, found: {models}")
        sys.exit(1)

    anomaly_types = list(results["AE-LSTM-ATT"].keys())
    metrics = ["auc", "f1"]

    print(f"{'Anomaly':<20}{'Metric':<8}{'ATT mean':<10}{'Base mean':<10}"
          f"{'diff':<9}{'paired-t p':<12}{'Wilcoxon p':<12}{'Cohen d':<9}")
    print("-" * 90)

    rows = []
    for at in anomaly_types:
        for metric in metrics:
            a = results["AE-LSTM-ATT"][at][metric]
            b = results["LSTM-AE"][at][metric]
            n = min(len(a), len(b))
            if n < 2:
                print(f"{at:<20}{metric:<8}  -- fewer than 2 paired seeds, skipping")
                continue
            a, b = np.array(a[:n]), np.array(b[:n])

            t_stat, t_p = stats.ttest_rel(a, b)
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
            except ValueError:
                # all differences identical (or all zero) -- Wilcoxon undefined
                w_p = float("nan")
            d = paired_cohens_d(a, b)
            diff = a.mean() - b.mean()

            print(f"{at:<20}{metric:<8}{a.mean():<10.4f}{b.mean():<10.4f}"
                  f"{diff:<+9.4f}{t_p:<12.4f}{w_p:<12.4f}{d:<9.3f}")
            rows.append(dict(anomaly=at, metric=metric, att_mean=a.mean(), base_mean=b.mean(),
                              diff=diff, paired_t_p=t_p, wilcoxon_p=w_p, cohens_d=d,
                              att_values=a.tolist(), base_values=b.tolist()))

    with open("significance_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print("\nSaved full results to significance_results.json")
    print("\nNote: n=5 paired samples is low power -- treat the Wilcoxon p-value as the")
    print("primary evidence and the paired-t p-value as a parametric cross-check, and")
    print("report Cohen's d alongside p regardless of significance, since a small p at")
    print("n=5 already implies a large effect size by construction.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python paired_significance_test.py results_raw.json")
        sys.exit(1)
    main(sys.argv[1])
