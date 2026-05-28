"""
clean_data.py

Ingests raw shipment records and produces a per-company feature table
using size-invariant metrics. Size-invariance is achieved by expressing
every feature as a ratio, coefficient of variation, entropy, or
concentration index — none of which scale with a company's shipment volume.

Expected input columns (see DATA_DICTIONARY.md):
    manifest_id, shipper_name, shipper_country, consignee_name,
    consignee_country, port_of_loading, port_of_discharge, hs_code,
    declared_weight_kg, declared_value_usd, shipment_date
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_shipments(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    required = {
        "manifest_id", "consignee_name", "shipper_name", "shipper_country",
        "port_of_loading", "port_of_discharge", "hs_code",
        "declared_weight_kg", "declared_value_usd", "shipment_date",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")

    df["declared_weight_kg"] = pd.to_numeric(df["declared_weight_kg"], errors="coerce")
    df["declared_value_usd"] = pd.to_numeric(df["declared_value_usd"], errors="coerce")
    df = df.dropna(subset=["consignee_name", "declared_weight_kg", "declared_value_usd"])
    df = df[df["declared_weight_kg"] > 0]
    return df


# ---------------------------------------------------------------------------
# Size-invariant feature helpers
# ---------------------------------------------------------------------------

def _cv(series: pd.Series) -> float:
    """Coefficient of variation — std / mean. 0 if mean is zero."""
    m = series.mean()
    return series.std() / m if m != 0 else 0.0


def _hhi(series: pd.Series) -> float:
    """Herfindahl-Hirschman Index: sum of squared market shares. Range [1/n, 1]."""
    counts = series.value_counts(normalize=True)
    return float((counts ** 2).sum())


def _diversity_norm(series: pd.Series) -> float:
    """Unique value count normalised by sqrt(n) to remove volume effect."""
    n = len(series)
    return series.nunique() / np.sqrt(n) if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-company aggregation
# ---------------------------------------------------------------------------

def compute_size_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["value_per_kg"] = df["declared_value_usd"] / df["declared_weight_kg"]
    df["route"] = df["port_of_loading"].str.strip() + "→" + df["port_of_discharge"].str.strip()
    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["is_weekend"] = df["shipment_date"].dt.dayofweek >= 5

    records = []
    for company, grp in df.groupby("consignee_name"):
        grp = grp.sort_values("shipment_date")
        n = len(grp)

        # Inter-shipment intervals in days (requires ≥2 shipments)
        intervals = grp["shipment_date"].diff().dt.days.dropna()
        interval_cv = _cv(intervals) if len(intervals) >= 2 else np.nan

        # Weight-value correlation (requires ≥3 shipments with variance)
        if n >= 3 and grp["declared_weight_kg"].std() > 0 and grp["declared_value_usd"].std() > 0:
            weight_value_corr = grp["declared_weight_kg"].corr(grp["declared_value_usd"])
        else:
            weight_value_corr = np.nan

        records.append({
            "company": company,
            "n_shipments": n,

            # Value density — flags mismatches between weight and declared value
            "value_per_kg_mean": grp["value_per_kg"].mean(),
            "value_per_kg_cv": _cv(grp["value_per_kg"]),

            # Weight-value alignment — legitimate goods have consistent unit economics
            "weight_value_corr": weight_value_corr,

            # Declared value volatility
            "value_cv": _cv(grp["declared_value_usd"]),
            "value_skew": float(skew(grp["declared_value_usd"])) if n >= 3 else 0.0,

            # Routing behaviour (normalised so small companies aren't penalised)
            "route_diversity": _diversity_norm(grp["route"]),
            "origin_diversity": _diversity_norm(grp["shipper_country"]),

            # Commodity concentration — high HHI = narrow range of goods
            "hs_diversity": _diversity_norm(grp["hs_code"]),
            "hs_hhi": _hhi(grp["hs_code"]),

            # Counterparty concentration — heavy reliance on one shipper is a flag
            "shipper_hhi": _hhi(grp["shipper_name"]),

            # Timing — legitimate businesses ship predominantly on weekdays
            "weekend_ratio": grp["is_weekend"].mean(),

            # Regularity of shipment cadence
            "interval_cv": interval_cv,
        })

    features = pd.DataFrame(records).set_index("company")
    # Fill NaN interval_cv / weight_value_corr with column median
    features = features.fillna(features.median(numeric_only=True))
    return features


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_features(features: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path)
    print(f"Saved {len(features)} company feature rows → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean shipment data and produce size-invariant company features.")
    parser.add_argument("input", help="Path to raw shipment CSV")
    parser.add_argument("--output", default="data/processed/company_features.csv", help="Output path")
    args = parser.parse_args()

    print(f"Loading shipments from {args.input} …")
    df = load_shipments(args.input)
    print(f"  {len(df):,} valid records across {df['consignee_name'].nunique():,} companies")

    print("Computing size-invariant features …")
    features = compute_size_invariant_features(df)

    save_features(features, args.output)


if __name__ == "__main__":
    main()
