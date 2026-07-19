# Decarbonizing Business Travel — Summer Analytics 2026 Capstone (in association with Celonis)

A two-part sustainability data science project analyzing ~65,000 business trips to help Celonis move toward its Net Zero by 2030 goal — combining process mining (Celonis) with a machine learning challenge to predict high-carbon trips before they're booked.

## Overview

Celonis' business travel is decentralized and carbon-intensive, contributing heavily to its Scope 3 emissions. This project tackles the problem from two angles:

- **Part 1 — Process Mining & Dashboarding:** Used Celonis to model the end-to-end travel process (booking → approval → departure → reimbursement), uncover inefficiencies, and build an executive-facing dashboard + pitch with quantified recommendations.
- **Part 2 — GreenTravel Intelligence Challenge:** A binary classification task predicting whether a planned trip will be `HighCarbon`, so it can be flagged before booking rather than after the fact.

## Key Findings (Part 1)

- **178.1M kg CO₂e** in total emissions, **$140.5M** in travel spend across the dataset
- Air travel (Business + Economy) accounts for **95%+** of total emissions — rail and EV usage is negligible
- **18.39%** of trips are out-of-policy, responsible for **32.7M kg CO₂e** and **$25.8M** in unregulated spend
- Sales & Marketing are the top emitting business units; Australia and Brazil departures dominate due to ultra-long-haul routes (e.g. Sydney–Frankfurt)
- Quantified recommendations: automate policy enforcement, shift short-haul routes (e.g. Madrid–Berlin) to rail, push virtual-first for internal travel — projected to cut ~35M kg CO₂e and save ~$26M annually

Deliverables: a 3-tab Celonis dashboard (Emissions Overview, Benchmarking, Emissions Drilldown), Process & Variant Explorer, and an executive pitch deck.

## Approach (Part 2 — ML Challenge)

**Task:** Predict `HighCarbon` (0/1) for trips in the private test set, using only features available before booking. Scored on ROC-AUC.

**Feature engineering:**
- Event log aggregations — step count, unique event types, per-event-type frequencies, total trip lifecycle duration
- Attribute-derived features — days preapproved, transportation price difference, extension length, flags for denials / new transport / new hotel selections
- Trip-level features — route (departure–arrival country pair), international flag, cost-per-night
- Excluded all target-leakage columns (`Departure_CO2e`, `Return_CO2e`, `Hotel_CO2e`, `Spend_CO2e`, `TotalCO2e`) per challenge rules

**Modeling:**
- Ensemble of LightGBM (Optuna-tuned, 50 trials), XGBoost, and CatBoost
- 5-fold Stratified Cross-Validation
- Simple averaging across the three models for final predictions

**Result:** Mean ROC-AUC of **0.9994** across folds.

> **Note on the high AUC:** `ShippingTypeDescription` (transport mode) is a near-deterministic predictor of `HighCarbon` in this dataset — rail trips are almost never high-carbon, premium flights almost always are. The score reflects the model correctly learning that strong, legitimate signal rather than any target leakage; leakage columns were explicitly excluded and validated via consistent scores across all 5 folds (0.9993–0.9995).

## Repo Structure

```
├── Part 1/                          # a ppt presenting the solution to the 5 business problems with celonis dashboard
├── Part 2/                          # Full ML pipeline: EDA, feature engineering, training, prediction,;Final predicted probabilities for the private test set
├── Public Dataset/                  # Training data (trip data, event log, attributes)
├── Private dataset/                 # Test data used for scoring
├── GreenTravel_..._ML_Solution.pdf  # ML methodology write-up
└── README.md
```

## Tech Stack

Python · pandas · scikit-learn · LightGBM · XGBoost · CatBoost · Optuna · Celonis (process mining & dashboarding)

## Author

Sujay M S — NITK Surathkal
