# Fraud Detection — Imports & Shipping

A machine learning pipeline for detecting anomalies and potential fraud in shipping and company import data.

## Overview

This project ingests shipping manifests, customs declarations, and company-level trade records to surface suspicious patterns such as:

- Unusual shipment volumes or frequencies relative to company history
- Mismatches between declared goods and typical commodity codes for a given supplier
- Discrepancies between invoice values and market benchmarks
- Shell company indicators (new entities, thin filing history, mismatched addresses)
- Route anomalies (atypical ports of origin/destination, unusual transshipment hubs)

## Data Sources

| Source | Description |
|--------|-------------|
| Shipping manifests | Bill of lading records including shipper, consignee, cargo description, weight, and value |
| Customs declarations | Import entry filings with HS codes, declared value, and country of origin |
| Company registry | Entity registration data — incorporation date, officers, addresses, filing status |
| Trade benchmarks | Reference pricing and volume norms by commodity code and trade lane |

## Project Structure

```
fraud-detection-imports/
├── data/
│   ├── raw/            # Unprocessed source files (not committed)
│   ├── processed/      # Cleaned and feature-engineered datasets
│   └── reference/      # Static reference tables (HS codes, port lists, etc.)
├── notebooks/          # Exploratory analysis and model prototyping
├── src/
│   ├── ingestion/      # Data loaders and parsers per source
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # Anomaly detection model definitions
│   ├── scoring/        # Batch and real-time scoring logic
│   └── alerts/         # Alert generation and deduplication
├── tests/              # Unit and integration tests
├── configs/            # Model and pipeline configuration files
├── requirements.txt
└── README.md
```

## Approach

The detection pipeline runs in two stages:

1. **Rule-based filters** — fast, interpretable checks that flag obvious red flags (e.g., declared value below a threshold for a high-value commodity code).
2. **Statistical / ML models** — unsupervised anomaly detection (Isolation Forest, Autoencoder) trained on historical clean shipments to score deviations from expected behavior.

Scores from both stages are combined into a single risk score per shipment, which feeds a case management queue for analyst review.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your data source credentials before running any pipeline step.

## Running the Pipeline

```bash
# Ingest and process raw data
python -m src.ingestion.run --date 2025-01-01

# Score new shipments
python -m src.scoring.run --date 2025-01-01

# Run tests
pytest tests/
```

## Contributing

1. Branch from `main` using the naming convention `feature/<short-description>`.
2. Open a pull request with a description of what the change does and why.
3. All new logic should include unit tests.
