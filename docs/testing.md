# Testing and performance (Phase 7)

## How to run

```bash
pytest                                  # all tests (about 1 minute)
pytest --cov --cov-report=term-missing  # same, plus which lines are covered
ruff check .                            # code quality
python scripts/benchmark.py             # speed and memory on this computer
```

## What is tested

| Level | Files | What it proves |
|---|---|---|
| Unit | `test_config`, `test_schema`, `test_loader`, `test_validation`, `test_eda`, `test_models`, `test_prediction`, `test_reports` | Each function gives the right answer, checked against values worked out by hand (e.g. Wilson interval, haversine distance, tied Spearman ranks, confusion-matrix counts). |
| Data validation | `test_validation`, `test_loader` | 10 planted data problems are each caught and handled as documented; Excel/binary/empty files get clear errors; results do not depend on chunk size. |
| Leakage | `test_models` | Card profiles learn from training rows only; training always ends before validation, validation before test; model choice uses validation, never test. |
| Model | `test_models`, `test_prediction` | Pipelines fit and score; a planted pattern is found; explanations reproduce the saved model's probability for **every** model type. |
| Storage & security | `test_storage`, `test_prediction` | Exact save/load round trip; failed saves leave nothing behind; SQL injection text and `%`/`_` wildcards are harmless; tampered model files are refused. |
| Integration | `test_integration` | The whole path in one go on an isolated workspace: sample file → validation → SQLite → training → saved model → scoring → search → explanation → summary → CSV + PDF → deletion. |
| UI flows | `test_ui`, `test_app_flows` | Every page opens with and without a dataset; one-click sample loading; riskiest-transaction explanation; manual entry; Critical filter; PDF creation; the dataset check-and-save flow. Uses a temporary database and model, never your real data. |
| Reports | `test_reports` | Summary numbers equal pandas calculations; CSV masks card numbers and filters correctly; the PDF is read back and contains the real numbers. |
| Regression | `test_prediction::test_model_parts_reproduce_every_model` and others | Bugs found during development stay fixed (see below). |

## Bugs found by testing (and kept fixed by tests)

| Phase | Bug | Fix | Guarding test |
|---|---|---|---|
| 3 | A page link crashed a page opened outside the app navigation | Safe link helper | `test_ui::test_each_view_renders_without_errors` |
| 3 | `Series.dt.dayofweek` deprecated in newer pandas | `day_of_week`; deprecation warnings now fail the tests | pytest `filterwarnings` |
| 6 | Partly covered first/last weeks looked like sudden drops | Partial edge weeks removed | `test_reports::test_weekly_drops_partial_edge_weeks` |
| 7 | Explanations were wrong when Logistic Regression is the selected model (calibrated on `decision_function`, explained with `predict_proba`) | Use the same score the calibrator was fitted on | `test_prediction::test_model_parts_reproduce_every_model` |
| 7 | Datasets without age/location/city size were scored without any notice | Warning on the dashboard, PDF and prediction page | `test_reports::test_missing_model_inputs_are_reported` |

## Edge cases checked

Only the 5 required columns · renamed columns (`Amount`, `Card Number`, `Date`) · categories
the model never saw · 5-row dataset · all transactions on one day · every row invalid
(0 usable rows; saving is blocked) · dataset without fraud labels · no High/Critical
transactions · deleting the active dataset.

## Manual checklist (run once per release)

| # | Step | Expected |
|---|---|---|
| 1 | Fresh start: `streamlit run app.py`, open every page | No errors; empty pages offer "Load sample dataset" |
| 2 | Click **Load sample dataset** on any empty page | Page reloads with data; Dataset Management shows it as active |
| 3 | Dataset Management → upload a `.xlsx` file | Clear message to save as CSV |
| 4 | Upload a CSV without an amount column | Column matching opens; save is not possible until it is mapped |
| 5 | Exploratory Analysis → all five tabs | Charts with titles, axis labels and "What this shows" notes |
| 6 | Model Performance → all four tabs | Real metrics; accuracy explained as misleading |
| 7 | Fraud Prediction → **Random known fraud**, then **Enter details** → Score | Probability, decision, tier, reasons, contribution chart |
| 8 | Risk Intelligence → move the threshold slider | Flagged/recall/precision update immediately |
| 9 | Transaction Investigation → Risk tier = Critical → click a row | Filtered table; explanation below |
| 10 | Executive Dashboard | Headline, 8 KPIs, charts, findings |
| 11 | Reports → download CSV and PDF | Files open; card numbers masked; PDF shows real numbers |
| 12 | Dataset Management → Saved datasets → Delete (confirm) | Dataset gone; other pages show the empty state |

## Performance

Measured with `python scripts/benchmark.py`. "Peak memory so far" is the highest memory
the process has used up to that step.

**Development machine (2-core cloud VM, small test model):**

| Step | 50k sample | 555k full test file |
|---|---|---|
| Read + validate + clean | 0.98 s | 5.2 s |
| Save to SQLite | 0.40 s | 3.4 s |
| Score all transactions | 0.12 s | 0.6 s |
| EDA summaries | 0.14 s | 0.8 s |
| Explain one transaction | 0.02 s | 0.01 s |
| Build PDF report | 0.46 s | 0.24 s |
| Peak memory | 0.3 GB | 1.0 GB |

**Laptop (real Random Forest model):** to be filled in from `scripts/benchmark.py` output.
