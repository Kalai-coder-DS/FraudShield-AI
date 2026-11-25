# Prediction, risk intelligence and investigation (Modules 5–7)

All three pages use the saved model (`models/fraudshield_model.joblib`) and the active
dataset. Nothing is retrained in the app.

## Scoring

The first time a page needs scores, the active dataset is scored in chunks of 50,000 rows
and stored in SQLite (`scores` table) together with the model's SHA-256 fingerprint. It is
scored again only when the model changes (after retraining). Each transaction gets:

| Output | Meaning |
|---|---|
| Fraud probability | Calibrated: of transactions scored like this, about this share were fraud |
| Model decision | "Flag for review" when probability ≥ the decision threshold chosen on validation data |
| Risk tier | Low < 30% ≤ Medium < 60% ≤ High < 85% ≤ Critical (`config.RiskThresholds`) |

The probability is a model estimate, not a confirmed fraud decision; every page says so.

| Tier | Recommended action |
|---|---|
| Low | Approve normally |
| Medium | Monitor / batch review |
| High | Analyst review today |
| Critical | Hold and review urgently |

On the Sparkov test period the known fraud rate per tier was 0.09% (Low), 36% (Medium),
76% (High) and 94% (Critical), which matches the probability ranges: the model is well
calibrated.

## Explanations

For one transaction, each feature group (amount, amount vs. the card's usual, time of day,
day of week, category, age, distance, city population) is replaced in turn by values from
200 typical transactions of the active dataset, and the transaction is re-scored:

```
contribution = P(transaction) − average P(transaction with that feature made typical)
```

Positive = pushed towards fraud. This is a simple, model-agnostic cousin of SHAP; it needs
no extra library and takes about 0.3 s. Features interact, so contributions are an
approximation and need not add up to the probability.

## Pages

- **Fraud Prediction**: pick a transaction (riskiest, random, random known fraud/legitimate,
  or by ID) or enter details by hand. Shows probability, decision, tier with action, known
  label if any, plain-language reasons and a contribution chart.
- **Risk Intelligence**: tier counts and amounts, known fraud rate per tier, categories /
  hours / amount ranges over-represented among High and Critical transactions (lift), a
  decision-threshold slider showing the precision/recall trade-off, and the paginated
  high-risk list.
- **Transaction Investigation**: filters (ID text, tier, category, amount, dates, known
  label, flagged only) run as parameterised SQL with pagination; click a row to open its
  explanation.

## Security notes

Search text is passed to SQLite only through `?` placeholders, and `%`/`_` are escaped, so
input such as `'; DROP TABLE transactions; --` is treated as plain text (tested). Card
numbers are masked to the last four digits everywhere in the UI.

## Performance

Scoring 555,000 rows: 1.8 s. One explanation: 0.3 s (2-core cloud machine).
