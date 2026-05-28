"""
analyze_anomalies.py

Scores companies for anomalous import behaviour using an Isolation Forest,
then explains each company's score with DIFFI (Depth-based Isolation Forest
Feature Importance).

DIFFI assigns each company a per-feature importance vector derived from the
average inverse-depth at which each feature was used to isolate that company
across all trees in the forest. High importance on a feature means that
feature was decisive in isolating (flagging) that company.

Output: a ranked table of all companies with their composite anomaly score and
a plain-English description of the primary drivers.

Usage:
    python -m src.analyze_anomalies data/processed/company_features.csv
    python -m src.analyze_anomalies data/processed/company_features.csv --output results/rankings.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Human-readable labels for each feature
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "value_per_kg_mean":  "average declared value per kilogram",
    "value_per_kg_cv":    "inconsistency of value-per-kg across shipments",
    "weight_value_corr":  "correlation between shipment weight and declared value",
    "value_cv":           "volatility of declared shipment values",
    "value_skew":         "skewness of declared shipment values",
    "route_diversity":    "variety of shipping routes (relative to volume)",
    "origin_diversity":   "variety of origin countries (relative to volume)",
    "hs_diversity":       "variety of commodity codes (relative to volume)",
    "hs_hhi":             "concentration of commodity codes",
    "shipper_hhi":        "reliance on a single overseas shipper",
    "weekend_ratio":      "proportion of shipments arriving on weekends",
    "interval_cv":        "irregularity of shipment timing",
}

# Direction guidance: True = high values are suspicious, False = low values are suspicious
_HIGH_IS_SUSPICIOUS = {
    "value_per_kg_mean":  True,
    "value_per_kg_cv":    True,
    "weight_value_corr":  False,
    "value_cv":           True,
    "value_skew":         True,
    "route_diversity":    False,
    "origin_diversity":   False,
    "hs_diversity":       False,
    "hs_hhi":             True,
    "shipper_hhi":        True,
    "weekend_ratio":      True,
    "interval_cv":        True,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        raise ValueError(f"No data found in {path}")
    return df


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 300,
    random_state: int = 42,
) -> tuple[IsolationForest, np.ndarray]:
    """
    Fit Isolation Forest and return (model, anomaly_scores).
    Scores are in [0, 1] — higher means more anomalous.
    sklearn's score_samples returns negative values (lower = more anomalous),
    so we invert and normalise here.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)
    raw_scores = model.score_samples(X)          # negative; lower = more anomalous
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    return model, anomaly_scores


# ---------------------------------------------------------------------------
# DIFFI — per-sample feature importance
# ---------------------------------------------------------------------------

def compute_diffi_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Return a (n_samples, n_features) matrix where entry [i, j] is the
    normalised importance of feature j in isolating sample i.

    Algorithm (Carletti et al., 2023):
      For each tree, walk the decision path for each sample. At each split
      node at depth d, accumulate 1/(d+1) for the feature used at that node.
      Average across trees, then normalise each sample's vector to sum to 1.
    """
    n_samples, n_features = X.shape
    importance = np.zeros((n_samples, n_features))

    for tree_estimator in model.estimators_:
        t = tree_estimator.tree_
        children_left  = t.children_left
        children_right = t.children_right
        split_feature  = t.feature
        threshold      = t.threshold

        for i in range(n_samples):
            node  = 0
            depth = 0
            # Walk until leaf (feature == -2 at leaf nodes)
            while split_feature[node] != -2:
                feat = split_feature[node]
                importance[i, feat] += 1.0 / (depth + 1)
                if X[i, feat] <= threshold[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
                depth += 1

    importance /= len(model.estimators_)

    # Normalise each row to [0, 1] sum so importances are comparable across companies
    row_sums = importance.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    importance /= row_sums

    return importance


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------

def _direction(feature: str, company_value: float, global_median: float) -> str:
    high_suspicious = _HIGH_IS_SUSPICIOUS.get(feature, True)
    is_high = company_value > global_median
    if (high_suspicious and is_high) or (not high_suspicious and not is_high):
        return "unusually high"
    return "unusually low"


def build_description(
    feature_importances: np.ndarray,
    feature_names: list[str],
    company_values: pd.Series,
    global_medians: pd.Series,
    top_n: int = 3,
) -> str:
    ranked = sorted(
        zip(feature_importances, feature_names),
        reverse=True,
    )[:top_n]

    parts = []
    for importance, feat in ranked:
        if importance < 0.05:
            continue
        label    = FEATURE_LABELS.get(feat, feat)
        c_val    = company_values.get(feat, np.nan)
        g_med    = global_medians.get(feat, np.nan)
        if np.isnan(c_val) or np.isnan(g_med):
            parts.append(f"{label} (importance {importance:.2f})")
            continue
        direction = _direction(feat, c_val, g_med)
        parts.append(
            f"{label} is {direction} "
            f"(company: {c_val:.3g}, median: {g_med:.3g}, importance: {importance:.2f})"
        )

    return "; ".join(parts) if parts else "no dominant driver identified"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_companies(
    features_df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    diffi_matrix: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    global_medians = features_df[feature_names].median()

    rows = []
    for idx, company in enumerate(features_df.index):
        description = build_description(
            feature_importances=diffi_matrix[idx],
            feature_names=feature_names,
            company_values=features_df.loc[company, feature_names],
            global_medians=global_medians,
        )
        rows.append({
            "rank":          idx + 1,         # filled after sort
            "company":       company,
            "anomaly_score": round(float(anomaly_scores[idx]), 4),
            "n_shipments":   int(features_df.loc[company, "n_shipments"])
                             if "n_shipments" in features_df.columns else None,
            "top_driver":    feature_names[int(diffi_matrix[idx].argmax())],
            "description":   description,
        })

    ranking = (
        pd.DataFrame(rows)
        .sort_values("anomaly_score", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_report(ranking: pd.DataFrame, top_n: int = 20) -> None:
    print(f"\n{'='*80}")
    print(f"  COMPANY ANOMALY RANKINGS  —  top {top_n} of {len(ranking)}")
    print(f"{'='*80}\n")
    for _, row in ranking.head(top_n).iterrows():
        n = f"({row['n_shipments']} shipments)" if row["n_shipments"] else ""
        print(f"#{row['rank']:>3}  {row['company']}  {n}")
        print(f"      Score : {row['anomaly_score']:.4f}")
        print(f"      Why   : {row['description']}")
        print()


def save_ranking(ranking: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(path, index=False)
    print(f"Full ranking saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score and rank companies by anomalousness using DIFFI Isolation Forest."
    )
    parser.add_argument("input", help="Path to company_features.csv (output of clean_data.py)")
    parser.add_argument("--output", default="data/processed/company_rankings.csv")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected fraction of anomalies (default 0.05)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees in Isolation Forest (default 300)")
    parser.add_argument("--top", type=int, default=20,
                        help="How many companies to print in the console report")
    args = parser.parse_args()

    print(f"Loading features from {args.input} …")
    features_df = load_features(args.input)
    print(f"  {len(features_df)} companies, {features_df.shape[1]} features")

    feature_names = [c for c in features_df.columns if c != "n_shipments"]
    X_raw = features_df[feature_names].values

    # RobustScaler handles the heavy-tailed distributions typical in trade data
    scaler = RobustScaler()
    X = scaler.fit_transform(X_raw)

    print(f"Fitting Isolation Forest ({args.n_estimators} trees, contamination={args.contamination}) …")
    model, anomaly_scores = run_isolation_forest(
        X,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
    )

    print("Computing DIFFI feature importances …")
    diffi_matrix = compute_diffi_scores(model, X)

    print("Building company rankings …")
    ranking = rank_companies(features_df, anomaly_scores, diffi_matrix, feature_names)

    print_report(ranking, top_n=args.top)
    save_ranking(ranking, args.output)


if __name__ == "__main__":
    main()
