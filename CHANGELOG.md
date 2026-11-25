# Changelog

## 1.0.0 — 2026-11-25

First complete release.

- **Dataset Management:** CSV upload, bundled sample and large local files; automatic column
  matching; chunked validation with a documented action for every data problem; SQLite storage.
- **Exploratory Analysis:** fraud rates by amount, time, category and customer with 95%
  confidence intervals; plain-language findings that only report clear differences.
- **Machine learning:** time-based split, leak-free pipelines, card spending profiles,
  Logistic Regression / Decision Tree / Random Forest / Isolation Forest, calibration,
  validation-based model and threshold choice. Random Forest: PR-AUC 0.823, recall 75.4%,
  precision 84.5% on unseen data.
- **Fraud Prediction, Risk Intelligence, Transaction Investigation:** stored scores, four
  risk tiers with actions, per-transaction explanations, threshold what-if, filters.
- **Executive Dashboard and Reports:** one shared summary; CSV export with masked cards;
  2–3 page PDF report.
- **Quality:** about 190 tests (unit, integration, Streamlit UI flows), 98% coverage,
  GitHub Actions, benchmark script.

### Development phases

| Version | Phase |
|---|---|
| 0.1 | Requirements, project setup, theme |
| 0.2 | Dataset validation and storage |
| 0.3 | Exploratory data analysis |
| 0.4 | Machine-learning engine |
| 0.5 | Prediction, risk intelligence, investigation |
| 0.6 | Dashboard and reports |
| 0.7 | Testing and performance (fixed: explanations for Logistic Regression, missing-input warning, deprecated pandas calls) |
| 1.0 | Documentation and release |
