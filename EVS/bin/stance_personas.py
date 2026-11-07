#!/usr/bin/env python3
"""Cluster the French EVS sample on persuasion-relevant "stance" axes.

Why this exists
- `french_personas.py` clusters on general life values (family, leisure, religion...). Those
  axes barely separate people on what matters for re.inform: trust in institutions, priority
  given to the environment, appetite for authority, openness to immigration, etc.
  (see docs/design/PROJECT_REVIEW.md §3).
- This script builds a small set of such axes from the same EVS 2017-2018 France extract,
  clusters on them for k in a small range (default 5..9), and writes, for each k, the clusters
  with their weighted population share, axis scores, a light demographic profile, party
  leanings, and a crosswalk to the existing k=10 value clusters (P01..P10).
- It is used to ground the persona options proposed in docs/proposals/personas.html.

Inputs (read only)
- data/evs_france_cleaned_cleaned_data.csv
- data/ZA7505_v5-0-0.dta (only for party value labels)

Outputs (EVS/output/)
- fr_stance_k{K}.csv / .json for each K
- fr_stance_scores.csv: k-selection diagnostics (silhouette, stability, smallest share)
- fr_stance_k{K}_vs_values_k10.csv: weighted crosswalk to the k=10 value clusters

Run from EVS/: python bin/stance_personas.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import french_personas as fp  # noqa: E402  (reuse cleaning + k=10 value axes)


AXES: Dict[str, str] = {
    "env_priority": "Puts the environment before growth / trusts the green movement",
    "inst_trust": "Trusts institutions (press, parliament, government, courts, EU, UN, elections)",
    "authority": "Wants authority and order (strong leader, obedience, death penalty, surveillance)",
    "openness": "Open to immigration and to the wider world",
    "right_wing": "Places self on the right (left-right scale)",
    "engagement": "Interested and active in politics (petitions, boycotts, demonstrations)",
    "hardship": "Struggling (low income, low life satisfaction, little control over life)",
    "tradition": "Religious and morally traditional (homosexuality, abortion, euthanasia)",
}


def _n(df: pd.DataFrame, v: str) -> pd.Series:
    return fp._nanify_neg_missing(df[v]).astype(float)


def _zmean(cols: List[pd.Series]) -> pd.Series:
    """Mean of item-level z-scores (items keep their own scale differences out)."""
    zs = [(c - c.mean()) / c.std() for c in cols]
    return pd.concat(zs, axis=1).mean(axis=1, skipna=True)


def build_stance_axes(df: pd.DataFrame) -> pd.DataFrame:
    rev4 = lambda v: 5 - _n(df, v)  # 1..4 scale where 1 = "a great deal" / "very good"
    rev10 = lambda v: 11 - _n(df, v)

    out = pd.DataFrame(index=df.index)

    out["env_priority"] = _zmean([
        _n(df, "B008").map({1: 1.0, 3: 0.5, 2: 0.0}),
        rev4("E069_14"),
    ])

    trust_items = [rev4(v) for v in ["E069_04", "E069_07", "E069_08", "E069_11", "E069_12",
                                      "E069_17", "E069_18", "E069_20"]]
    trust_items += [_n(df, "E111_01")]
    trust_items += [rev4(v) for v in ["E265_01", "E265_05", "E265_06"]]  # fair elections/coverage
    trust_items += [_n(df, "E265_07")]  # rich people buy elections: 1 very often -> low trust
    out["inst_trust"] = _zmean(trust_items)

    out["authority"] = _zmean([
        _n(df, "E018").map({1: 1.0, 2: 0.5, 3: 0.0}),
        rev4("E114"),
        rev4("E116"),
        _n(df, "A042"),
        _n(df, "F144_02"),
        rev4("H009"), rev4("H010"), rev4("H011"),
    ])

    out["openness"] = _zmean([
        _n(df, "G052"),
        _n(df, "C002_01"),          # 5 = strongly disagree that nationals get priority
        1 - _n(df, "A124_06"),      # did not reject immigrants as neighbours
        rev4("G063"),               # feels close to the world
    ])

    out["right_wing"] = _zmean([_n(df, "E033")])

    act = {1: 1.0, 2: 0.5, 3: 0.0}
    out["engagement"] = _zmean([
        rev4("E023"),
        _n(df, "E025").map(act), _n(df, "E026").map(act), _n(df, "E027").map(act),
    ])

    out["hardship"] = _zmean([rev10("X047E_EVS5"), rev10("A170"), rev10("A173")])

    out["tradition"] = _zmean([
        rev10("F118"), rev10("F120"), rev10("F122"),
        _n(df, "D081"),             # 5 = strongly disagree that same-sex couples parent as well
        _n(df, "F063"),             # importance of God 1..10
    ])

    out["gwght"] = df["gwght"].astype(float)
    return out


def demographics(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["age"] = _n(df, "X003")
    d["age_65plus"] = (d["age"] >= 65).astype(float).where(d["age"].notna())
    d["age_under35"] = (d["age"] < 35).astype(float).where(d["age"].notna())
    d["female"] = (_n(df, "X001") == 2).astype(float).where(_n(df, "X001").notna())
    edu = _n(df, "X025R")
    d["edu_upper"] = (edu == 3).astype(float).where(edu.notna())
    d["edu_lower"] = (edu == 1).astype(float).where(edu.notna())
    d["town_under5k"] = (df["size_5c"] == 1).astype(float)
    d["city_500k_plus"] = (df["size_5c"] == 5).astype(float)
    d["env_first_raw"] = (_n(df, "B008") == 1).astype(float).where(_n(df, "B008").notna())
    d["no_trust_gov_raw"] = (_n(df, "E069_11") == 4).astype(float).where(_n(df, "E069_11").notna())
    return d


def party_labels(dta: Path) -> Dict[int, str]:
    try:
        import pyreadstat
        _, meta = pyreadstat.read_dta(dta, metadataonly=True)
        labs = meta.value_labels.get(meta.variable_to_label.get("E181_EVS5"), {})
        short = {}
        for k, v in labs.items():
            v = str(v).replace("FR: ", "")
            short[int(k)] = v.split(" (")[0]
        return short
    except Exception:
        return {}


def wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna()
    return float(np.average(x[m], weights=w[m])) if m.any() else float("nan")


def k10_value_labels(df_raw: pd.DataFrame, outdir: Path, seed: int) -> pd.Series:
    """Reproduce the k=10 value clusters of french_personas.py and return persona ids."""
    feats = fp.build_value_axes(df_raw)
    cols = [c for c in feats.columns if c != "gwght"]
    X = feats[cols].astype(float).apply(lambda s: s.fillna(s.median()))
    X = StandardScaler().fit_transform(X.values)
    labels = KMeans(n_clusters=10, n_init=50, random_state=seed).fit_predict(X)
    ref = json.loads((outdir / "fr_personas_k10.json").read_text(encoding="utf-8"))
    cl2pid = {int(p["cluster"]): p["persona_id"] for p in ref["personas"]}
    return pd.Series([cl2pid[int(c)] for c in labels], index=df_raw.index, name="value_persona")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/evs_france_cleaned_cleaned_data.csv")
    ap.add_argument("--dta", default="data/ZA7505_v5-0-0.dta")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=9)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed = args.random_state

    df_raw = fp.load_france_csv(Path(args.csv))
    axes = build_stance_axes(df_raw)
    demo = demographics(df_raw)
    w = axes["gwght"]
    axis_cols = list(AXES)

    Xdf = axes[axis_cols].apply(lambda s: s.fillna(s.median()))
    X = StandardScaler().fit_transform(Xdf.values)
    Z = pd.DataFrame(X, columns=axis_cols, index=axes.index)
    # weighted centring so cluster scores read as "vs the French average"
    Z = Z - pd.Series({c: wmean(Z[c], w) for c in axis_cols})

    value_persona = k10_value_labels(df_raw, outdir, seed)
    parties = party_labels(Path(args.dta))
    party = fp._nanify_neg_missing(df_raw["E181_EVS5"])

    score_rows = []
    for k in range(args.k_min, args.k_max + 1):
        km = KMeans(n_clusters=k, n_init=50, random_state=seed)
        labels = km.fit_predict(X, sample_weight=w.values)
        sil = float(silhouette_score(X, labels))
        runs = [KMeans(n_clusters=k, n_init=10, random_state=rs).fit_predict(X, sample_weight=w.values)
                for rs in range(seed, seed + 8)]
        ari = float(np.mean([adjusted_rand_score(runs[i], runs[j])
                             for i in range(len(runs)) for j in range(i + 1, len(runs))]))

        lab = pd.Series(labels, index=axes.index)
        shares = lab.groupby(lab).apply(lambda s: w[s.index].sum() / w.sum())
        score_rows.append({"k": k, "silhouette": sil, "stability_ari": ari,
                           "min_cluster_share_w": float(shares.min())})

        rows = []
        for cl in shares.sort_values(ascending=False).index:
            idx = lab.index[lab == cl]
            ww = w[idx]
            z = {f"z_{c}": wmean(Z.loc[idx, c], ww) for c in axis_cols}
            top = sorted(axis_cols, key=lambda c: -abs(z[f"z_{c}"]))[:3]
            dem = {c: wmean(demo.loc[idx, c], ww) for c in demo.columns}
            p = party[idx]
            pw = ww[p.notna()].groupby(p.dropna()).sum()
            pw = pw / ww.sum()
            top_parties = [{"party": parties.get(int(pid), str(int(pid))), "share": float(s)}
                           for pid, s in pw.sort_values(ascending=False).head(4).items()]
            vp = ww.groupby(value_persona[idx]).sum() / ww.sum()
            rows.append({
                "cluster": int(cl),
                "population_share_w": float(shares[cl]),
                "n_respondents": int(len(idx)),
                "top_axes": ";".join(f"{c}{'+' if z[f'z_{c}'] > 0 else '-'}" for c in top),
                **z,
                **dem,
                "top_parties": json.dumps(top_parties, ensure_ascii=False),
                "value_personas_k10": json.dumps({k_: round(float(v), 3) for k_, v in
                                                  vp.sort_values(ascending=False).items()}),
            })
        res = pd.DataFrame(rows)
        res.insert(0, "stance_id", [f"S{i + 1:02d}" for i in range(len(res))])
        res.to_csv(outdir / f"fr_stance_k{k}.csv", index=False)

        cl2sid = dict(zip(res["cluster"], res["stance_id"]))
        sid = lab.map(cl2sid)
        cross = pd.crosstab(value_persona, sid, values=w, aggfunc="sum", normalize="index").round(3)
        cross.index.name = "value_persona_k10 (row = 100%)"
        cross.to_csv(outdir / f"fr_stance_k{k}_vs_values_k10.csv")

        payload = {
            "k": k,
            "random_state": seed,
            "source_csv": args.csv,
            "n_respondents": int(len(axes)),
            "weights": "gwght",
            "axes": AXES,
            "clusters": [{**r, "top_parties": json.loads(r["top_parties"]),
                          "value_personas_k10": json.loads(r["value_personas_k10"])}
                         for r in res.to_dict(orient="records")],
        }
        (outdir / f"fr_stance_k{k}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote fr_stance_k{k}.csv/.json  silhouette={sil:.3f}  stability={ari:.3f}  "
              f"min_share={shares.min():.3f}")

    pd.DataFrame(score_rows).to_csv(outdir / "fr_stance_scores.csv", index=False)
    print("Wrote fr_stance_scores.csv")

    # --- Two-question grid (no clustering): climate priority x trust in institutions.
    # Meant for a UI where the user places the person they talk to from two observable traits.
    b008 = fp._nanify_neg_missing(df_raw["B008"])
    climate = pd.Series("unsure", index=df_raw.index)
    climate[b008 == 1] = "env_first"
    climate[b008 == 2] = "growth_first"
    t = axes["inst_trust"]
    ts = t.dropna().sort_values()
    t_med = float(np.interp(0.5, np.cumsum(w[ts.index]) / w[ts.index].sum(), ts.values))
    trust = pd.Series(np.where(t.fillna(t.median()) >= t_med, "trusts", "distrusts"), index=df_raw.index)
    cell = climate + "|" + trust
    k6 = json.loads((outdir / "fr_stance_k6.json").read_text(encoding="utf-8"))
    lab6 = KMeans(n_clusters=6, n_init=50, random_state=seed).fit_predict(X, sample_weight=w.values)
    sid6 = pd.Series(lab6, index=axes.index).map({c["cluster"]: c["stance_id"] for c in k6["clusters"]})
    grid_rows = []
    for c in sorted(cell.unique()):
        idx = cell.index[cell == c]
        ww = w[idx]
        p = party[idx]
        pw = ww[p.notna()].groupby(p.dropna()).sum() / ww.sum()
        grid_rows.append({
            "cell": c,
            "population_share_w": float(ww.sum() / w.sum()),
            "n_respondents": int(len(idx)),
            **{f"z_{a}": wmean(Z.loc[idx, a], ww) for a in axis_cols},
            **{d_: wmean(demo.loc[idx, d_], ww) for d_ in demo.columns},
            "top_parties": [{"party": parties.get(int(pid), str(int(pid))), "share": float(s)}
                            for pid, s in pw.sort_values(ascending=False).head(4).items()],
            "stance_k6": {k_: round(float(v), 3) for k_, v in
                          (ww.groupby(sid6[idx]).sum() / ww.sum()).sort_values(ascending=False).items()},
            "value_personas_k10": {k_: round(float(v), 3) for k_, v in
                                   (ww.groupby(value_persona[idx]).sum() / ww.sum())
                                   .sort_values(ascending=False).items()},
        })
    grid = sorted(grid_rows, key=lambda r: -r["population_share_w"])
    (outdir / "fr_stance_grid.json").write_text(json.dumps({
        "definition": {
            "climate": "B008: 1 -> env_first, 2 -> growth_first, other answer / don't know / no answer -> unsure",
            "trust": "inst_trust axis split at its weighted median",
            "inst_trust_weighted_median": t_med,
        },
        "cells": grid,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{**r, "top_parties": json.dumps(r["top_parties"], ensure_ascii=False),
                   "stance_k6": json.dumps(r["stance_k6"]),
                   "value_personas_k10": json.dumps(r["value_personas_k10"])} for r in grid]
                 ).to_csv(outdir / "fr_stance_grid.csv", index=False)
    print("Wrote fr_stance_grid.csv/.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
