# Data Dictionary

## shipping_manifests

| Field | Type | Description |
|-------|------|-------------|
| `manifest_id` | string | Unique bill of lading number |
| `shipper_name` | string | Name of the exporting entity |
| `shipper_country` | string | ISO 3166-1 alpha-2 country code of origin |
| `consignee_name` | string | Name of the importing entity |
| `consignee_country` | string | ISO 3166-1 alpha-2 country code of destination |
| `port_of_loading` | string | UN/LOCODE of departure port |
| `port_of_discharge` | string | UN/LOCODE of arrival port |
| `cargo_description` | string | Free-text description of goods |
| `hs_code` | string | Harmonized System commodity code (6–10 digits) |
| `declared_weight_kg` | float | Gross weight declared on manifest |
| `declared_value_usd` | float | Declared customs value in USD |
| `shipment_date` | date | Date of departure |
| `vessel_name` | string | Carrying vessel or flight number |

## company_registry

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | string | Internal unique identifier |
| `legal_name` | string | Registered legal name |
| `registration_country` | string | Country of incorporation |
| `registration_date` | date | Date of incorporation |
| `registered_address` | string | Official registered address |
| `operating_address` | string | Primary operating address (if different) |
| `officer_count` | int | Number of listed directors/officers |
| `filing_status` | string | `active`, `dissolved`, `suspended` |
| `last_filing_date` | date | Most recent regulatory filing |

## risk_scores (output)

| Field | Type | Description |
|-------|------|-------------|
| `manifest_id` | string | Foreign key to `shipping_manifests` |
| `rule_score` | float | 0–1 score from rule-based checks |
| `ml_score` | float | 0–1 anomaly score from ML model |
| `composite_score` | float | Weighted combination of both scores |
| `triggered_rules` | list[str] | Names of any rule-based flags triggered |
| `scored_at` | timestamp | UTC timestamp of scoring run |
