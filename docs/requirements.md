# FraudShield AI — Phase 0: Requirements & Feasibility Analysis (Draft SRS)

**Status:** Approved 24 Nov 2026. Hardware-specific limits (§8) are finalised once laptop specs are known.
**Date:** 24 Nov 2026

---

## 1. Project analysis — is this feasible?

Yes, but only if it is scoped honestly. The project you described has two halves:

| Half | What it really is | Risk |
|---|---|---|
| **Data science core** | Data validation → EDA → leak-free preprocessing → model comparison on imbalanced data → threshold/risk tiers → explanations | This is what interviewers care about. Must be done carefully. |
| **Product shell** | Dashboard, upload, investigation, reports | Easy to make look good while hiding a weak core. Must stay a thin layer over real results. |

The biggest design problem in the brief: **"upload any CSV" vs. "a trained model".** A model only works on data with the same columns it was trained on. Real CSVs from different sources all have different columns. So the app will:

1. Define one **canonical transaction schema** (based on the chosen dataset).
2. Validate uploads against it, with a column-mapping step if names differ (e.g. `amt` → `amount`).
3. **Labelled upload** (has a fraud column) → EDA, scoring and evaluation of the saved model; optionally retrain.
4. **Unlabelled upload** → EDA and scoring only, clearly marked "no ground truth, so no accuracy metrics".
5. Reject uploads that can't be mapped, with a clear message listing the missing columns.

This is a strong interview talking point: it shows you understand why ML systems can't take arbitrary input.

---

## 2. The real-world business problem

Card and digital-payment fraud costs banks, fintechs and merchants billions each year. Fraud teams face three problems:

- **Volume:** millions of transactions a day; humans can review only a tiny fraction.
- **Rarity:** fraud is often 0.1–1% of transactions, so it is a needle in a haystack.
- **Cost asymmetry:** missing a fraud (false negative) loses money. Blocking a real customer (false positive) loses trust and revenue.

A fraud-scoring system ranks transactions by risk so analysts review the riskiest first. **It does not decide what is fraud.** A human or a downstream process confirms it. FraudShield AI models this workflow: score → tier → investigate → report.

**Target users**

| User | Needs |
|---|---|
| Fraud analyst | Ranked high-risk list, per-transaction explanation, search/filter |
| Risk manager | KPIs, trends, risk distribution, exportable report |
| Data scientist | Data quality checks, model comparison, metrics, retraining |
| Non-technical viewer (recruiter) | A dashboard that explains itself in plain words |

---

## 3. Functional requirements

Priority: **E** = essential, **O** = optional (if time allows), **X** = unnecessary / cut.

### Module 1 — Executive Dashboard
| ID | Requirement | Priority |
|---|---|---|
| F1.1 | KPIs computed from the scored dataset: total transactions, flagged count, flagged %, total value, value at risk, high+critical count | E |
| F1.2 | Risk-tier distribution chart | E |
| F1.3 | Transactions & flagged transactions over time | E |
| F1.4 | Plain-language caption under every KPI ("what this means") | E |
| F1.5 | Empty state when no dataset is scored yet (no fake numbers) | E |

### Module 2 — Dataset Management
| ID | Requirement | Priority |
|---|---|---|
| F2.1 | CSV upload with size limit and encoding/delimiter checks | E |
| F2.2 | Preview (first N rows), shape, column names, dtypes | E |
| F2.3 | Missing values per column, duplicate rows, invalid values (negative amounts, unparseable dates, out-of-range categories) | E |
| F2.4 | Required-column validation and column mapping | E |
| F2.5 | Data quality score + readable issue list | E |
| F2.6 | "Use bundled sample dataset" button | E |
| F2.7 | Delete dataset with confirmation | E |
| F2.8 | Multiple saved datasets with history | O |

### Module 3 — EDA
| ID | Requirement | Priority |
|---|---|---|
| F3.1 | Fraud vs. legitimate class balance | E |
| F3.2 | Amount distribution (log scale) split by class | E |
| F3.3 | Fraud rate by hour/day | E |
| F3.4 | Fraud rate by category / transaction type | E |
| F3.5 | Correlation heatmap (numeric features) | E |
| F3.6 | Outlier detection (IQR) on amount | E |
| F3.7 | Every chart: title, axis labels, a 1–2 line "what this shows" note | E |
| F3.8 | Automated text summary of key findings | O |

### Module 4 — ML Engine
| ID | Requirement | Priority |
|---|---|---|
| F4.1 | sklearn `Pipeline` + `ColumnTransformer` (impute, encode, scale) fitted on training data only | E |
| F4.2 | Duplicate removal **before** splitting | E |
| F4.3 | Feature engineering (hour, day of week, log amount, customer age, distance, and per-card behaviour features computed from **past transactions only**) | E |
| F4.4 | Time-based split (train on earlier, test on later) where timestamps exist; otherwise stratified split | E |
| F4.5 | Class-imbalance report and handling (`class_weight="balanced"`, threshold tuning) | E |
| F4.6 | Models: Logistic Regression (baseline, explainable), Decision Tree, Random Forest | E |
| F4.7 | Isolation Forest as an **unsupervised** comparison (does not use labels) | E |
| F4.8 | Metrics: precision, recall, F1, ROC-AUC, **PR-AUC**, confusion matrix, accuracy (shown with a warning) | E |
| F4.9 | Model comparison table and PR/ROC curves | E |
| F4.10 | Save/load model and metadata with joblib (no retraining at app start) | E |
| F4.11 | Probability calibration (`CalibratedClassifierCV`) so "70% risk" means something | O (recommended) |
| F4.12 | Hyperparameter search (small, randomized) | O |
| F4.13 | SMOTE / oversampling | X — class weights do the job without an extra dependency; can be discussed as an alternative |
| F4.14 | Gradient boosting (XGBoost/LightGBM), deep learning | X — heavier, not needed to prove the skill. sklearn's `HistGradientBoostingClassifier` can be an optional extra model with no new dependency |

### Module 5 — Fraud Prediction
| ID | Requirement | Priority |
|---|---|---|
| F5.1 | Pick a transaction from the dataset and score it | E |
| F5.2 | Manual entry form with validation | E |
| F5.3 | Show probability, predicted class at chosen threshold, risk tier | E |
| F5.4 | Top contributing factors (see explainability below) | E |
| F5.5 | Permanent disclaimer: "Model risk score, not a confirmed fraud decision" | E |

### Module 6 — Risk Intelligence
| ID | Requirement | Priority |
|---|---|---|
| F6.1 | Four tiers from documented, configurable probability thresholds | E |
| F6.2 | High/Critical transaction table, paginated | E |
| F6.3 | Patterns among flagged transactions (by category, hour, amount band) | E |
| F6.4 | Per-tier explanation text | E |
| F6.5 | Threshold what-if: move the slider, see how flagged count and estimated precision/recall change | O (recommended — great demo) |

**Proposed default tiers** (to be tuned on validation data and documented in `config`):

| Tier | Probability | Business meaning |
|---|---|---|
| Low | < 0.30 | Approve normally |
| Medium | 0.30 – 0.60 | Monitor / batch review |
| High | 0.60 – 0.85 | Analyst review today |
| Critical | ≥ 0.85 | Hold transaction, urgent review |

These cut-offs are a starting point, not a business rule. Phase 4 will report precision and recall at each cut-off so the choice is justified with data.

### Module 7 — Transaction Investigation
| ID | Requirement | Priority |
|---|---|---|
| F7.1 | Search by transaction ID; filter by tier, amount range, date, category | E |
| F7.2 | Detail view: ID, amount, time, features, probability, tier, prediction, explanation | E |
| F7.3 | If labelled: show actual label next to prediction | E |
| F7.4 | Analyst notes / status (reviewed, confirmed fraud) stored in SQLite | O |

### Module 8 — Reports
| ID | Requirement | Priority |
|---|---|---|
| F8.1 | CSV export of scored transactions (filtered view) | E |
| F8.2 | PDF report: dataset summary, fraud stats, model metrics, risk distribution, findings, charts | E |
| F8.3 | Report generated from current results, with timestamp and model version | E |

### Explainability approach (lightweight)
- **Logistic Regression:** exact per-feature contributions (coefficient × scaled value). Fully explainable and interview-friendly.
- **Random Forest / Tree:** global feature importance (permutation importance) plus a simple local explanation (how the score changes when each feature is reset to a typical value).
- **SHAP:** optional, not essential. Adds a dependency and CPU cost. Only add it if your laptop handles it.

---

## 4. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | App starts in under ~10 s; no training at start; dashboard pages render in under 2–3 s on the sample dataset |
| Memory | Peak RAM target is set from your specs (e.g. under 1.5 GB on an 8 GB laptop); large CSVs read in chunks with explicit dtypes |
| Usability | Every page says what it does; non-technical captions; consistent theme |
| Reliability | Every user error produces a clear message, never a raw traceback |
| Correctness | No hardcoded metrics; every number traceable to code and data |
| Maintainability | Modular `src/` package; UI pages contain no ML logic |
| Reproducibility | Fixed random seeds; pinned `requirements.txt`; model metadata records data hash, params, metrics, date |
| Security | See §10 |

---

## 5. Recommended architecture & technology stack

### Architecture: one Python process

```
Browser ──► Streamlit app (UI pages, thin)
                 │
                 ▼
          src/fraudshield  (pure Python, fully testable)
          ├─ data/      load, validate, quality checks
          ├─ features/  feature engineering
          ├─ models/    train, evaluate, registry (joblib)
          ├─ risk/      tiers, explanations
          ├─ reports/   CSV / PDF
          └─ storage/   SQLite access
                 │
       ┌─────────┴──────────┐
   SQLite (scored results,   models/ (*.joblib + metadata.json)
   run history, notes)       data/   (raw CSV, sample, Parquet cache)
```

The key principle: **all logic lives in `src/` and is tested with pytest without the UI.** Streamlit only displays results. This separation is what makes the project look like real engineering.

### Stack decisions

| Technology | Decision | Why |
|---|---|---|
| Python 3.10–3.12 | **Use** | Final version depends on your laptop |
| Pandas, NumPy | **Use** | Core data handling |
| Scikit-learn | **Use** | All four models, pipelines, metrics, calibration — one library |
| **Streamlit** | **Use** | Single process, no separate frontend, built-in caching (`st.cache_data`, `st.cache_resource`), file upload and multi-page navigation. Custom theme via `.streamlit/config.toml` + a small CSS file covers your plum/violet/coral design |
| Django | **Reject** | Would need templates, a JS charting layer, auth, ORM and a separate server. A lot of extra code with no added data science value for a single-user analytics tool |
| **Plotly** | **Use (in the app)** | Your brief requires interactive charts; Streamlit renders Plotly natively |
| Matplotlib | **Use (narrowly)** | Static charts embedded in the PDF report and in notebooks |
| Seaborn | **Drop** | Duplicates Plotly + Matplotlib; nothing it adds is required |
| **SQLite** | **Use** | Built into Python (no install, no server). Stores scored transactions so tables are paginated with `LIMIT/OFFSET` instead of holding everything in memory; also stores run history and analyst notes |
| joblib | **Use** | Save/load models (installed with sklearn) |
| fpdf2 *or* reportlab | **Use one** | PDF reports. fpdf2 is smaller and simpler; reportlab is more powerful. Recommendation: **fpdf2** |
| pytest | **Use** | Unit and integration tests |
| Parquet (pyarrow) | **Optional** | Fast, compact cache of processed data. Use only if the dataset is large |
| SHAP | **Optional** | See explainability above |
| Docker, Redis, Celery, FastAPI, React, PostgreSQL | **Reject** | Not needed for a single-user local app; would strain your laptop |
| Ruff (linter/formatter) | **Optional, recommended** | Tiny, fast, and shows code-quality discipline |

**Estimated runtime dependencies:** about 7 packages (`pandas`, `numpy`, `scikit-learn`, `streamlit`, `plotly`, `matplotlib`, `fpdf2`) + `pytest` for development.

---

## 6. Dataset options

| Dataset | Size | Features | Pros | Cons | Fit |
|---|---|---|---|---|---|
| **Sparkov simulated credit-card transactions** (Kaggle, "Credit Card Transactions Fraud Detection Dataset") | ~1.3M train + ~0.55M test rows, ~0.5% fraud, ~350 MB + ~150 MB | Timestamp, card number, merchant, **category**, amount, customer gender/age/city/location, merchant location | Human-readable features → great EDA, explanations and manual-entry form; real timestamps allow time-based split and per-card behaviour features | Synthetic (simulated) data; large files need sampling | **Recommended** |
| **PaySim mobile-money** (Kaggle) | ~6.3M rows, ~0.13% fraud, ~470 MB | Step (hour), type, amount, origin/destination balances | Readable; simple schema | Known leakage trap: balance columns almost give away the answer, and fraud occurs only in TRANSFER/CASH_OUT. Very large | Good alternative; needs careful leakage discussion |
| **ULB Credit Card Fraud** (Kaggle, European cardholders 2013) | 284,807 rows, 492 frauds (0.17%), ~150 MB | Time, Amount, **V1–V28 (anonymised PCA)**, Class | Real data, famous benchmark | Features meaningless to humans → poor EDA, explanations and manual entry. Bad fit for a "business user" product | Not recommended as main dataset (can be mentioned as a benchmark) |
| **IEEE-CIS Fraud** (Kaggle) | 590k rows, 400+ columns, >1 GB | Many anonymised columns | Realistic | Too heavy for a limited laptop; messy | Reject |

**Plan with Sparkov:**
- A **stratified ~50k-row sample** (all fraud kept proportionally, fixed seed) committed to the repo as `data/sample/` so the app works right after cloning. It is a real subset, not invented data.
- The full dataset is downloaded by you from Kaggle (not committed — too big). A script processes it in chunks into Parquet/SQLite.
- Train on Sparkov's own train file (earlier dates) and evaluate on its test file (later dates) — a genuine time-based holdout.

Please check the licence on each Kaggle page before choosing; Phase 8 documentation will cite the source and licence.

---

## 7. Potential technical challenges

| Challenge | Why it matters | Mitigation |
|---|---|---|
| Extreme class imbalance | 99.5% accuracy by predicting "never fraud" | PR-AUC, recall/precision, class weights, threshold tuning; explain clearly in the UI |
| Data leakage | Inflated metrics that collapse in reality | Fit all transformers inside a Pipeline on train only; dedupe before split; time-based split; behaviour features use only past rows; drop IDs and target-derived columns |
| Uploaded CSVs with different schemas | Model cannot score unknown columns | Canonical schema + mapping + clear rejection |
| Uncalibrated probabilities | Random Forest "0.7" ≠ 70% chance | Calibration on validation data; tiers based on calibrated scores |
| Isolation Forest output isn't a probability | Mixing it into risk tiers would be wrong | Present it separately as an "anomaly score" and compare it only with ranking metrics |
| Large files on limited RAM | Crashes / swapping | Chunked reading, explicit dtypes (`category`, `float32`), sampling, Parquet cache, SQLite pagination |
| Streamlit reruns the script on every click | Slow app | `st.cache_resource` for the model, `st.cache_data` for data, session state |
| Synthetic data | Results may look better than in real life | State this openly in Limitations |
| Explaining Random Forest predictions | Users ask "why?" | Logistic Regression contributions + permutation importance; SHAP optional |

---

## 8. Laptop performance considerations

Final numbers depend on your specs; general rules:

- **Training happens once**, offline via `python -m fraudshield.train`, and saves a `.joblib` file. The app only loads it.
- **Random Forest** with `n_estimators≈100–200`, `max_depth` limited, `n_jobs` set to CPU cores minus one, trained on a sample if RAM is low. Model file kept small (tens of MB, not GBs).
- **Reading CSVs:** `usecols`, explicit `dtype`, `chunksize` for the full file; downcast numbers.
- **Tables:** never send 1M rows to the browser — SQLite query with `LIMIT 50 OFFSET n`.
- **Charts:** aggregate first (e.g. fraud rate per hour), plot the aggregate, not raw points; sample for scatter plots.
- **One process only:** `streamlit run app.py`. No Docker, no background workers.
- **Sample-dataset mode** on by default; full dataset is opt-in.

Rough guide (to confirm once you share specs):

| RAM | Approach |
|---|---|
| 4 GB | Sample only (≤100k rows); LR + Tree + small RF |
| 8 GB | Full Sparkov train file in chunks; RF on ~300–500k rows |
| 16 GB+ | Full dataset comfortably |

---

## 9. Proposed folder structure

```
fraudshield-ai/
├── app.py                      # Streamlit entry point: page config, theme, sidebar navigation
├── views/                      # one file per module, UI only (navigation built in app.py)
│   ├── home.py
│   ├── dashboard.py  dataset.py  eda.py  models.py
│   └── prediction.py  risk.py  investigation.py  reports.py
├── src/fraudshield/
│   ├── __init__.py
│   ├── config.py               # paths, schema, risk thresholds, seeds
│   ├── data/                   # loader.py, validation.py, quality.py, schema.py
│   ├── features/               # engineering.py
│   ├── models/                 # train.py, evaluate.py, registry.py, explain.py
│   ├── risk/                   # tiers.py
│   ├── reports/                # csv_report.py, pdf_report.py
│   ├── storage/                # db.py (SQLite)
│   └── ui/                     # theme.py, components.py (KPI cards, empty states)
├── scripts/
│   ├── make_sample.py          # builds stratified sample from full dataset
│   └── train_models.py         # offline training entry point
├── data/
│   ├── sample/                 # small committed sample
│   ├── raw/                    # full dataset (git-ignored)
│   └── processed/              # cache (git-ignored)
├── models/                     # saved model + metadata.json
├── notebooks/                  # 01_eda.ipynb, 02_modelling.ipynb (optional, for explanation)
├── tests/                      # test_validation.py, test_features.py, test_models.py, ...
├── docs/                       # SRS, architecture, workflow diagrams, screenshots
├── .streamlit/config.toml      # theme colours
├── pyproject.toml              # package, pytest and ruff config
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 10. Security requirements (proportionate to a local portfolio app)

- No secrets in code or Git; `.env` git-ignored (the app currently needs no secrets at all).
- Upload limits: file size cap, CSV only, row limit for in-browser preview.
- Parse uploads as data only; never `eval` cell contents; parameterised SQL queries.
- Escape user text before rendering HTML/CSS in Streamlit.
- Don't commit real personal data; the dataset is synthetic. Card numbers are hashed/masked in the UI.
- Load only model files the project itself produced (joblib/pickle can run code if a file is malicious).

---

## 11. Testing requirements

| Level | Examples |
|---|---|
| Unit | Validation catches missing columns, negative amounts, bad dates; feature functions produce expected values; risk tiers map boundaries correctly |
| Leakage tests | Pipeline is fitted only on train; behaviour features for a row never use future rows |
| Model tests | Saved model loads and predicts with correct shape; probabilities in [0,1]; metrics beat a "predict majority class" baseline on PR-AUC |
| Integration | Upload → validate → score → store → report end-to-end on the sample |
| Report tests | CSV/PDF numbers match the computed statistics |
| UI | Streamlit `AppTest` smoke tests (each page loads; empty states show) + manual checklist with screenshots |
| Performance | Time and peak memory for loading, scoring and page render on the sample and the full file |

---

## 12. Deployment

- **Primary:** runs locally (`streamlit run app.py`).
- **Optional:** Streamlit Community Cloud (free) for a public demo link using the sample dataset and a small model. Has tight memory limits, so the sample mode is what makes this possible.

---

## 13. Development phases & estimated timeline

Assuming ~2–3 focused hours per day:

| Phase | Content | Estimate | Commit |
|---|---|---|---|
| 0 | Requirements & SRS (this doc, finalised after your answers) | 1 day | `docs: add requirements specification` |
| 1 | Repo setup, structure, config, theme, empty pages, CI-ready tests | 1–2 days | `feat: initialize FraudShield AI architecture` |
| 2 | Upload, schema validation, quality checks, sample dataset, SQLite storage | 2–3 days | `feat: implement transaction data validation` |
| 3 | EDA module | 2 days | `feat: add exploratory data analysis` |
| 4 | Features, pipelines, 4 models, evaluation, calibration, registry | 3–4 days | `feat: implement fraud detection models` |
| 5 | Prediction, risk tiers, explanations, investigation | 2–3 days | `feat: add transaction risk prediction` |
| 6 | Dashboard, reports (CSV/PDF), full integration, UI polish | 2–3 days | `feat: implement fraud analytics dashboard` |
| 7 | Testing and performance optimisation | 2 days | `test: add application and model tests` |
| 8 | README, diagrams, screenshots, docs, release tag | 1–2 days | `docs: complete project documentation` |
| | **Total** | **~3–4 weeks part-time** | |

---

## 14. Limitations & future improvements

**Limitations (to state openly in the README):** synthetic training data; a static model (no live retraining on analyst feedback); single-user app with no login; risk thresholds are illustrative, not regulatory; explanations are approximations for non-linear models.

**Future improvements:** analyst feedback loop for retraining; drift monitoring; gradient-boosted models; SHAP; real-time scoring API (FastAPI); user roles; graph features (shared devices/merchants).
