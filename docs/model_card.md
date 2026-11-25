# Model card — FraudShield AI fraud scorer v1.0

A short, standard summary of what the trained model is, how well it works and where it
should not be trusted. All numbers come from `models/metadata.json`.

## Model

| | |
|---|---|
| Type | Random Forest (150 trees, depth ≤ 14, ≥ 20 rows per leaf, 30% of rows per tree), balanced class weights |
| Output | Calibrated fraud probability (isotonic regression on the validation period) |
| Decision threshold | 0.49 — the highest F1 on validation data |
| Risk tiers | Low < 30% ≤ Medium < 60% ≤ High < 85% ≤ Critical |
| Size | 3 MB (`models/fraudshield_model.joblib`, git-ignored, rebuilt by `scripts/train_models.py`) |
| Chosen over | Logistic Regression, Decision Tree, Isolation Forest — by validation PR-AUC |

## Intended use

Ranking card transactions so that fraud analysts review the riskiest first. **Not** for
automatically blocking customers or accusing anyone of fraud without human review.

## Data

| Period | Rows | Fraud | Used for |
|---|---|---|---|
| Train: 1 Jan 2019 – 6 Mar 2020 | 1,037,340 | 5,968 | Fitting the model and all learned statistics |
| Validation: 6 Mar – 21 Jun 2020 | 259,335 | 1,538 | Calibration, threshold, model choice |
| Test: 21 Jun – 31 Dec 2020 | 555,719 | 2,145 | Final report only |

Source: Sparkov simulated card transactions (Kaggle, kartik2112), ~1,000 customers and
~800 merchants. Synthetic: no real person's data.

## Features

Amount (log), hour, late-night flag, weekday, customer age at the time of the transaction,
customer–merchant distance, city population (log), spending category, and how unusual the
amount is for that card (z-score against the card's training-period spending).

**Not used:** gender (protected attribute), job, merchant, city/state (too many values),
identifiers. Card ID is only used to look up the card's spending profile.

## Results on the test period

| Metric | Random Forest | Logistic Regression | Decision Tree | Isolation Forest* |
|---|---|---|---|---|
| PR-AUC | **0.823** | 0.249 | 0.337 | 0.102 |
| ROC-AUC | 0.994 | 0.928 | 0.985 | 0.913 |
| Precision | 84.5% | 33.8% | 37.8% | — |
| Recall | 75.4% | 34.4% | 67.9% | — |
| F1 | 0.797 | 0.341 | 0.486 | — |
| Fraud caught when reviewing the same number of alerts | 75.4% | 33.1% | 38.6% | 17.9% |

\*Unsupervised: it never sees fraud labels, so it has no probability threshold.

Confusion matrix at threshold 0.49: 1,617 fraud caught, 528 missed, 297 false alarms,
553,277 legitimate correctly approved. Accuracy 99.85% (approve-everything baseline 99.61%).

**Calibration:** known fraud rate per tier on the test period — Low 0.09%, Medium 36%,
High 76%, Critical 94%.

**What the model relies on** (drop in validation PR-AUC when shuffled): amount 0.84,
category 0.54, hour 0.13, late night 0.09, amount vs. card's usual 0.08, age 0.05; distance,
weekday and city population ≈ 0.

## Limitations and risks

- Trained on simulated data; real fraud patterns differ and drift over time.
- One in four fraud cases (528 of 2,145) is still missed at the chosen threshold.
- Datasets without age, location or city size are scored with typical values; the app warns
  when this happens.
- Explanations are approximations; features interact.
- Age is used as a feature; before real use, outcomes should be checked for unfair
  differences between age groups.

## How to reproduce

```bash
python scripts/train_models.py   # with fraudTrain.csv and fraudTest.csv in data/raw/
```

Random seed 42; scikit-learn 1.9.1, Python 3.14 on the training machine.
