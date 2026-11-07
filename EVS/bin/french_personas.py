#!/usr/bin/env python3
"""Build French EVS "personality" clusters.

Goal
- Produce 10–20 personas representing the French EVS5 sample.
- Each persona has:
  1) a "system of values" summary (interpretable axes)
  2) a recommended persuasion format (calm discussion, example/social proof, authority/rules, etc.)

Inputs
- EVS/data/evs_france_cleaned_cleaned_data.csv (already filtered to cntry=250)
- EVS/data/ZA7505_v5-0-0.dta (for variable labels)

Outputs
- EVS/output/fr_personas_k{K}.csv
- EVS/output/fr_personas_k{K}.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


NEG_MISSING = {-5, -4, -3, -2, -1}


@dataclass(frozen=True)
class AxisSpec:
    name: str
    features: List[str]


def _nanify_neg_missing(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
    return s.mask(s.isin(list(NEG_MISSING)))


def _reverse_1_to_n(s: pd.Series, n: int) -> pd.Series:
    # assumes s in [1..n]
    return (n + 1) - s


def _binary_yes_no_to_01(s: pd.Series) -> pd.Series:
    # EVS typically: 1 yes, 2 no
    return s.map({1: 1.0, 2: 0.0})


def load_france_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop numeric columns that are entirely missing-coded.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    all_missing = [c for c in numeric_cols if df[c].isin(list(NEG_MISSING)).all()]
    if all_missing:
        df = df.drop(columns=all_missing)

    return df


def load_variable_labels_from_dta(dta_path: Path, variables: Iterable[str]) -> Dict[str, str]:
    variables = list(dict.fromkeys(variables))
    _, meta = pyreadstat.read_dta(dta_path, usecols=variables, row_limit=1)

    out: Dict[str, str] = {}
    for v in variables:
        if v in meta.column_names:
            out[v] = meta.column_labels[meta.column_names.index(v)]
    return out


def build_value_axes(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable value axes.

    Notes on coding assumptions:
    - Many EVS/WVS items are standard across waves and are coded consistently.
    - Negative codes (-5..-1) are treated as missing.
    - We orient axes so that "higher" generally means "more" of the named concept.
    """

    df = df_raw.copy()

    # --- Life priorities (1 very important .. 4 not at all) => reverse to 1..4 importance
    importance_vars = [
        "A001",  # family
        "A002",  # friends
        "A003",  # leisure
        "A004",  # politics
        "A005",  # work
        "A006",  # religion
    ]
    for v in importance_vars:
        if v in df.columns:
            s = _nanify_neg_missing(df[v])
            df[f"imp_{v}"] = _reverse_1_to_n(s, 4)

    # Happiness (A008): standard 1 very happy .. 4 not at all happy => reverse
    if "A008" in df.columns:
        s = _nanify_neg_missing(df["A008"])
        df["happiness"] = _reverse_1_to_n(s, 4)

    # Subjective health (A009): standard 1 very good .. 5 very poor => reverse
    if "A009" in df.columns:
        s = _nanify_neg_missing(df["A009"])
        df["health"] = _reverse_1_to_n(s, 5)

    # Life satisfaction (A170): 1 dissatisfied .. 10 satisfied
    if "A170" in df.columns:
        df["life_satisfaction"] = _nanify_neg_missing(df["A170"])

    # Freedom of choice/control (A173): 1 none .. 10 a great deal
    if "A173" in df.columns:
        df["freedom_control"] = _nanify_neg_missing(df["A173"])

    # Interpersonal trust (A165): 1 trust, 2 can't be too careful
    if "A165" in df.columns:
        s = _nanify_neg_missing(df["A165"])
        df["interpersonal_trust"] = s.map({1: 1.0, 2: 0.0})

    # Child qualities (0/1 mentioned): construct "conformity/order" vs autonomy index.
    # Conventionally: good manners, hard work, responsibility are more conformity;
    # independence, imagination, tolerance/respect are more autonomy/universalism.
    child = {
        "manners": "A027",
        "independence": "A029",
        "hard_work": "A030",
        "responsibility": "A032",
        "imagination": "A034",
        "tolerance": "A035",
    }
    for k, v in child.items():
        if v in df.columns:
            df[f"child_{k}"] = _nanify_neg_missing(df[v])

    conformity_vars = ["child_manners", "child_hard_work", "child_responsibility"]
    autonomy_vars = ["child_independence", "child_imagination", "child_tolerance"]
    if all(c in df.columns for c in conformity_vars + autonomy_vars):
        df["conformity_order"] = df[conformity_vars].mean(axis=1) - df[autonomy_vars].mean(axis=1)
        df["universalism_tolerance"] = df["child_tolerance"]

    # Economic ideology (10-point scales):
    # E035: 1 incomes more equal .. 10 larger differences => reverse to pro-equality
    if "E035" in df.columns:
        s = _nanify_neg_missing(df["E035"])
        df["pro_equality"] = _reverse_1_to_n(s, 10)

    # E036: 1 private ownership .. 10 state ownership
    if "E036" in df.columns:
        df["pro_state_ownership"] = _nanify_neg_missing(df["E036"])

    # E037: 1 people responsibility .. 10 government responsibility
    if "E037" in df.columns:
        df["pro_government_responsibility"] = _nanify_neg_missing(df["E037"])

    # Democracy characteristics (1 not essential .. 10 essential)
    for v, name in [
        ("E224", "dem_tax_rich"),
        ("E225", "dem_religious_law"),
        ("E226", "dem_free_elections"),
        ("E227", "dem_state_aid_unemp"),
        ("E228", "dem_army_takeover"),
        ("E229", "dem_civil_rights"),
    ]:
        if v in df.columns:
            df[name] = _nanify_neg_missing(df[v])

    liberal_components = [c for c in ["dem_free_elections", "dem_civil_rights", "dem_tax_rich"] if c in df.columns]
    if liberal_components:
        df["liberal_democracy"] = df[liberal_components].mean(axis=1)

    auth_components = [c for c in ["dem_religious_law", "dem_army_takeover"] if c in df.columns]
    if auth_components:
        df["authoritarian_democracy"] = df[auth_components].mean(axis=1)

    if "dem_state_aid_unemp" in df.columns:
        df["welfare_democracy"] = df["dem_state_aid_unemp"]

    # Religiosity composite
    if "F034" in df.columns:
        s = _nanify_neg_missing(df["F034"])
        df["religiosity_identity"] = (4 - s)  # 1 religious .. 3 atheist

    if "F063" in df.columns:
        df["religiosity_god_importance"] = _nanify_neg_missing(df["F063"])

    belief_vars = {
        "belief_god": "F050",
        "belief_life_after_death": "F051",
        "belief_hell": "F053",
        "belief_heaven": "F054",
    }
    for name, v in belief_vars.items():
        if v in df.columns:
            df[name] = _binary_yes_no_to_01(_nanify_neg_missing(df[v]))

    relig_components = [
        c
        for c in [
            "religiosity_identity",
            "religiosity_god_importance",
            "belief_god",
            "belief_life_after_death",
            "belief_hell",
            "belief_heaven",
        ]
        if c in df.columns
    ]
    if relig_components:
        df["religiosity"] = df[relig_components].mean(axis=1)

    # Consolidated feature set used for clustering.
    cluster_features = [
        "imp_A001",
        "imp_A002",
        "imp_A003",
        "imp_A004",
        "imp_A005",
        "imp_A006",
        "happiness",
        "health",
        "life_satisfaction",
        "freedom_control",
        "interpersonal_trust",
        "conformity_order",
        "universalism_tolerance",
        "pro_equality",
        "pro_state_ownership",
        "pro_government_responsibility",
        "liberal_democracy",
        "authoritarian_democracy",
        "welfare_democracy",
        "religiosity",
    ]
    cluster_features = [c for c in cluster_features if c in df.columns]

    # Return only features + weight
    out = df[[c for c in cluster_features if c in df.columns]].copy()
    if "gwght" in df.columns:
        out["gwght"] = df["gwght"].astype(float)
    else:
        out["gwght"] = 1.0

    return out


def _weighted_cluster_shares(labels: np.ndarray, weights: np.ndarray) -> Dict[int, float]:
    totals = {}
    total_w = float(np.sum(weights))
    for k in np.unique(labels):
        totals[int(k)] = float(np.sum(weights[labels == k]) / total_w)
    return totals


def choose_best_k(
    X: np.ndarray,
    weights: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    min_cluster_share: float,
) -> Tuple[int, pd.DataFrame]:
    rows = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=25, random_state=random_state)
        labels = model.fit_predict(X)

        sil = float(silhouette_score(X, labels, metric="euclidean"))
        shares = _weighted_cluster_shares(labels, weights)
        min_share = min(shares.values()) if shares else 0.0

        # stability: agreement across different initializations
        label_sets = []
        for rs in range(random_state, random_state + 8):
            m = KMeans(n_clusters=k, n_init=10, random_state=rs)
            label_sets.append(m.fit_predict(X))
        aris = []
        for i in range(len(label_sets)):
            for j in range(i + 1, len(label_sets)):
                aris.append(adjusted_rand_score(label_sets[i], label_sets[j]))
        stability = float(np.mean(aris)) if aris else float("nan")

        rows.append(
            {
                "k": k,
                "silhouette": sil,
                "stability_ari": stability,
                "min_cluster_share_w": float(min_share),
            }
        )

    scores = pd.DataFrame(rows).sort_values(["silhouette", "stability_ari"], ascending=False)

    # Prefer solutions with no tiny clusters.
    viable = scores[scores["min_cluster_share_w"] >= min_cluster_share]
    if not viable.empty:
        best = int(viable.iloc[0]["k"])
    else:
        # fallback: take best silhouette anyway
        best = int(scores.iloc[0]["k"])

    return best, scores


def persuasion_format_for_cluster(z: pd.Series) -> str:
    """Rule-based persuasion profile from cluster z-scores.

    Returns a small controlled vocabulary.
    """

    # Authority/rules: high conformity + low interpersonal trust OR high authoritarian-democracy
    if z.get("conformity_order", 0) > 0.6 and (z.get("interpersonal_trust", 0) < -0.2 or z.get("authoritarian_democracy", 0) > 0.6):
        return "authority_rules"

    # Moral/order/tradition framing
    if z.get("religiosity", 0) > 0.6 and z.get("conformity_order", 0) > 0.3:
        return "moral_tradition_order"

    # Moral/care framing: pro-equality + liberal democracy + tolerance
    if z.get("pro_equality", 0) > 0.4 and z.get("liberal_democracy", 0) > 0.4 and z.get("universalism_tolerance", 0) > 0.2:
        return "moral_care_fairness"

    # Calm discussion/evidence: high liberal-democracy + high freedom/control + moderate trust
    if z.get("liberal_democracy", 0) > 0.4 and z.get("freedom_control", 0) > 0.2 and z.get("interpersonal_trust", 0) > -0.2:
        return "calm_discussion_evidence"

    # Example/social proof: family/friends important + moderate trust
    if z.get("imp_A001", 0) > 0.3 and z.get("imp_A002", 0) > 0.1 and z.get("interpersonal_trust", 0) > -0.1:
        return "example_social_proof"

    # Constraint/enforcement: low trust + low liberal democracy
    if z.get("interpersonal_trust", 0) < -0.6 and z.get("liberal_democracy", 0) < -0.3:
        return "constraint_enforcement"

    return "calm_discussion"


def summarize_value_system(z: pd.Series) -> str:
    parts: List[str] = []

    if z.get("religiosity", 0) > 0.6:
        parts.append("religious")
    elif z.get("religiosity", 0) < -0.6:
        parts.append("secular")

    if z.get("liberal_democracy", 0) > 0.6:
        parts.append("liberal-democratic")
    elif z.get("authoritarian_democracy", 0) > 0.6:
        parts.append("authoritarian-leaning")

    if z.get("pro_equality", 0) > 0.6 and z.get("pro_government_responsibility", 0) > 0.4:
        parts.append("egalitarian")
    elif z.get("pro_equality", 0) < -0.6 and z.get("pro_state_ownership", 0) < -0.4:
        parts.append("market-leaning")

    if z.get("conformity_order", 0) > 0.6:
        parts.append("order/conformity")
    elif z.get("conformity_order", 0) < -0.6:
        parts.append("autonomy")

    if z.get("universalism_tolerance", 0) > 0.6:
        parts.append("tolerant")

    if not parts:
        return "mixed"
    return ", ".join(parts)


def build_personas(
    features_df: pd.DataFrame,
    k_min: int,
    k_max: int,
    random_state: int,
    min_cluster_share: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    feature_cols = [c for c in features_df.columns if c != "gwght"]

    # simple missing imputation (median) on axes
    X_df = features_df[feature_cols].copy()
    for c in feature_cols:
        X_df[c] = X_df[c].astype(float)
        X_df[c] = X_df[c].fillna(X_df[c].median())

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    weights = features_df["gwght"].astype(float).values

    best_k, k_scores = choose_best_k(
        X,
        weights,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
        min_cluster_share=min_cluster_share,
    )

    model = KMeans(n_clusters=best_k, n_init=50, random_state=random_state)
    labels = model.fit_predict(X)

    # Cluster-level means in original feature scale
    df_labeled = features_df.copy()
    df_labeled["cluster"] = labels

    overall_means = df_labeled[feature_cols].mean(numeric_only=True)
    overall_stds = df_labeled[feature_cols].std(numeric_only=True).replace(0, np.nan)

    rows = []
    shares = _weighted_cluster_shares(labels, weights)

    for cl in sorted(df_labeled["cluster"].unique()):
        sub = df_labeled[df_labeled["cluster"] == cl]
        means = sub[feature_cols].mean(numeric_only=True)
        z = (means - overall_means) / overall_stds
        z = z.fillna(0.0)

        # identify top defining axes
        top = z.abs().sort_values(ascending=False).head(4)
        top_axes = list(top.index)

        rows.append(
            {
                "cluster": int(cl),
                "population_share_w": float(shares.get(int(cl), 0.0)),
                "value_system": summarize_value_system(z),
                "persuasion_format": persuasion_format_for_cluster(z),
                "top_axes": ";".join(top_axes),
                **{f"z_{c}": float(z.get(c, 0.0)) for c in feature_cols},
            }
        )

    personas = pd.DataFrame(rows).sort_values("population_share_w", ascending=False).reset_index(drop=True)

    # nice names: Persona 1..K in share order
    personas.insert(0, "persona_id", [f"P{i+1:02d}" for i in range(len(personas))])

    # Simple axis display names
    axis_names = {
        "imp_A001": "Family importance",
        "imp_A002": "Friends importance",
        "imp_A003": "Leisure importance",
        "imp_A004": "Politics importance",
        "imp_A005": "Work importance",
        "imp_A006": "Religion importance",
        "happiness": "Happiness",
        "health": "Health (subjective)",
        "life_satisfaction": "Life satisfaction",
        "freedom_control": "Freedom/control",
        "interpersonal_trust": "Interpersonal trust",
        "conformity_order": "Conformity/order",
        "universalism_tolerance": "Tolerance (child quality)",
        "pro_equality": "Pro-equality",
        "pro_state_ownership": "Pro-state ownership",
        "pro_government_responsibility": "Gov responsibility",
        "liberal_democracy": "Liberal democracy",
        "authoritarian_democracy": "Authoritarian democracy",
        "welfare_democracy": "Welfare (unemployment aid)",
        "religiosity": "Religiosity",
    }

    return personas, k_scores, axis_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(Path("data/evs_france_cleaned_cleaned_data.csv")))
    parser.add_argument("--dta", default=str(Path("data/ZA7505_v5-0-0.dta")))
    parser.add_argument("--outdir", default=str(Path("output")))
    parser.add_argument("--k-min", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--min-share", type=float, default=0.03)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    csv_path = Path(args.csv)
    dta_path = Path(args.dta)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = load_france_csv(csv_path)
    features = build_value_axes(df_raw)

    personas, k_scores, axis_names = build_personas(
        features,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        min_cluster_share=args.min_share,
    )

    k = len(personas)
    personas_path = outdir / f"fr_personas_k{k}.csv"
    scores_path = outdir / f"fr_personas_k{k}_scores.csv"
    json_path = outdir / f"fr_personas_k{k}.json"

    personas.to_csv(personas_path, index=False)
    k_scores.to_csv(scores_path, index=False)

    payload = {
        "k": k,
        "source_csv": str(csv_path),
        "source_dta": str(dta_path),
        "n_respondents": int(len(features)),
        "axes": axis_names,
        "personas": personas.to_dict(orient="records"),
        "k_selection_scores": k_scores.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also emit variable labels (useful for interpretation) for the axes' originating vars.
    # Not all derived axes have direct labels, but the raw vars do.
    raw_vars = [
        "A001","A002","A003","A004","A005","A006","A008","A009",
        "A027","A029","A030","A032","A034","A035",
        "A165","A170","A173",
        "E035","E036","E037",
        "E224","E225","E226","E227","E228","E229",
        "F034","F050","F051","F053","F054","F063",
    ]
    try:
        labels = load_variable_labels_from_dta(dta_path, raw_vars)
        (outdir / f"fr_personas_k{k}_var_labels.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        # Keep the main pipeline working even if metadata extraction fails.
        pass

    print("Wrote", personas_path)
    print("Wrote", scores_path)
    print("Wrote", json_path)

    # Quick stdout summary
    show_cols = ["persona_id", "population_share_w", "value_system", "persuasion_format", "top_axes"]
    print("\nTop personas (by weighted share):")
    print(personas[show_cols].head(12).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
