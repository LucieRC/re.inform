#!/usr/bin/env python3
"""Margins of error for the population shares of the 6 chosen stance clusters.

Why this exists
- fr_stance_k6.json gives each cluster's weighted share but no uncertainty. The public
  methodology article (website2/backend/data/articles/methodologie_profils.md) reports one.

Two margins, both at the usual 95% level, in percentage points
- survey_margin_pts: sampling error alone, the groups being taken as fixed. Accounts for the
  survey weights (linearised variance of a weighted proportion).
- bootstrap_margin_pts: redo the whole grouping on resampled respondents, so it also includes
  the fact that the borders between groups move. Half-width of the 2.5%-97.5% interval of the
  resampled shares. Each resample's clusters are matched to the published ones by closest centre.

Deterministic (seed 42). Reads the same inputs as stance_personas.py and first checks that it
recovers the published shares. Writes output/fr_stance_k6_margins.csv, and
output/fr_stance_missing_axes.csv (share of respondents with no usable answer on each axis, whose
score is then set to the median).

Run from EVS/: python bin/stance_margins.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import french_personas as fp  # noqa: E402
import stance_personas as sp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/evs_france_cleaned_cleaned_data.csv")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    seed = args.random_state
    k = 6

    df_raw = fp.load_france_csv(Path(args.csv))
    axes = sp.build_stance_axes(df_raw)
    w = axes["gwght"].to_numpy()
    cols = list(sp.AXES)
    X = StandardScaler().fit_transform(axes[cols].apply(lambda s: s.fillna(s.median())).values)

    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X, sample_weight=w)
    labels = km.labels_
    shares = np.array([w[labels == c].sum() / w.sum() for c in range(k)])

    published = json.loads((outdir / "fr_stance_k6.json").read_text(encoding="utf-8"))
    for c in published["clusters"]:
        if abs(shares[c["cluster"]] - c["population_share_w"]) > 1e-9:
            raise SystemExit(f"Cannot recover published share for {c['stance_id']}")

    n_eff = w.sum() ** 2 / (w ** 2).sum()

    rng = np.random.default_rng(seed)
    n = len(w)
    boot = np.empty((args.n_boot, k))
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        m = KMeans(n_clusters=k, n_init=10, random_state=seed + b).fit(X[idx], sample_weight=w[idx])
        cost = ((km.cluster_centers_[:, None, :] - m.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        ref, new = linear_sum_assignment(cost)
        for r, nw in zip(ref, new):
            boot[b, r] = w[idx][m.labels_ == nw].sum() / w[idx].sum()

    rows = []
    for c in published["clusters"]:
        cl = c["cluster"]
        y = (labels == cl).astype(float)
        se = np.sqrt((w ** 2 * (y - shares[cl]) ** 2).sum()) / w.sum()
        lo, hi = np.percentile(boot[:, cl], [2.5, 97.5])
        rows.append({
            "stance_id": c["stance_id"],
            "population_share_w_pct": round(100 * shares[cl], 2),
            "n_respondents": int(y.sum()),
            "survey_margin_pts": round(100 * 1.96 * se, 2),
            "bootstrap_low_pct": round(100 * lo, 2),
            "bootstrap_high_pct": round(100 * hi, 2),
            "bootstrap_margin_pts": round(100 * (hi - lo) / 2, 2),
            "effective_sample_size": round(n_eff, 1),
            "n_boot": args.n_boot,
            "random_state": seed,
        })
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "fr_stance_k6_margins.csv", index=False)
    print(res.to_string(index=False))

    missing = pd.DataFrame({
        "axis": cols,
        "missing_share_pct": [round(100 * float(axes[c].isna().mean()), 1) for c in cols],
    })
    missing.to_csv(outdir / "fr_stance_missing_axes.csv", index=False)
    print(missing.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
