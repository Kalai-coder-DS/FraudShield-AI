# Machine-learning engine (Module 4)

## Run it

```bash
python scripts/train_models.py                       # uses data/raw/fraudTrain.csv + fraudTest.csv
python scripts/train_models.py --train-fraction 0.3  # less memory / faster
```

Outputs: `models/fraudshield_model.joblib` (selected model, git-ignored, rebuilt by
the script) and `models/metadata.json` (all results; read by the Model Performance page).

## Workflow

```
fraudTrain.csv (earlier dates)                              fraudTest.csv (later dates)
|---------------- train 80% ----------------|-- val 20% --|   |---------- test ----------|
 fit pipelines (features, scaling,            calibrate,        reported once; never used
 encoding, card profiles)                     threshold,        for any decision
                                              model choice
```

1. **Load and validate** both files with the same code as the app (`process_dataset`).
2. **Time-based split.** Models learn from earlier transactions and are judged on later
   ones. A random split would leak future information.
3. **One sklearn `Pipeline` per model**: row features → card profile → impute / scale /
   one-hot encode → model. Every learned statistic is fitted on the training period only.
4. **Class imbalance**: `class_weight="balanced"` (no oversampling library needed).
5. **Calibration** (isotonic, on validation) so that probabilities mean what they say;
   the risk tiers use calibrated probabilities.
6. **Decision threshold**: highest F1 on validation.
7. **Model selection**: highest PR-AUC on validation.
8. **Test report**: all metrics once on the test file.
9. **Permutation importance** (validation data) for the selected model.

## Models

| Model | Why it is included | Settings (laptop-friendly) |
|---|---|---|
| Logistic Regression | Simple, explainable baseline | balanced class weights |
| Decision Tree | Easy to explain as rules | depth ≤ 8, ≥ 50 rows per leaf |
| Random Forest | Captures non-linear patterns and interactions | 150 trees, depth ≤ 14, each tree sees 30% of rows |
| Isolation Forest | Unsupervised: finds unusual transactions **without** labels | compared by ranking only |

Isolation Forest outputs an anomaly score, not a fraud probability, so it gets PR-AUC and
ROC-AUC and "fraud caught at the same alert volume", but no precision/recall at a threshold.

## Features

| Feature | How it is made |
|---|---|
| Amount (log) | `log(1 + amount)`; amounts are very skewed |
| Hour, weekday, late night (22:00–03:59) | From the timestamp |
| Customer age | At the time of the transaction |
| Distance to merchant | Haversine distance between customer and merchant |
| City population (log) | |
| Spending category | One-hot encoded |
| Amount unusual for this card | z-score of the amount against the card's typical amount, **learned from the training period only** (`CardProfile`) |
| Card seen in training | 1/0; unknown cards use the overall profile |

Not used, on purpose: gender (protected attribute), job, merchant, city/state (too many
values, overfitting), IDs. "Transactions in the last 24 hours" was not used because a
sample or uploaded extract does not contain every transaction of a card, so the feature
would mean something different at training and scoring time (train/serve skew).

## Metrics

Accuracy is shown with its baseline (always answering "legitimate"). The main metrics are
precision, recall, F1 and **PR-AUC**, whose random baseline equals the fraud rate.
ROC-AUC is reported but looks optimistic when fraud is rare.

## Safety of the saved model

joblib files can run code when loaded. The SHA-256 fingerprint of the model file is stored
in `metadata.json` and checked before loading; a modified file is refused.

## Performance

Generated data with the size of `fraudTrain.csv` (1.3 M rows) plus a 555 k test file, on a
2-core cloud machine: **85 s for all four models, 1.2 GB peak memory**. On a laptop with
little RAM use `--train-fraction 0.3`.
