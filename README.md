# FraudShield AI

**Machine-learning fraud detection and risk intelligence platform.**

FraudShield AI helps a fraud team find suspicious financial transactions. It checks
uploaded transaction data, explores fraud patterns, scores every transaction with a
trained model, groups transactions into Low / Medium / High / Critical risk, and
explains each score so an analyst knows what to investigate first.

> A fraud probability is a model's estimate based on past patterns. It is a signal for
> prioritising review, not a confirmed fraud decision.

---

## Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Requirements & feasibility ([docs/requirements.md](docs/requirements.md)) | ✅ Done |
| 1 | Project setup, architecture, theme, navigation | ✅ Done |
| 2 | Dataset upload, validation, data quality | ✅ Done |
| 3 | Exploratory data analysis | ✅ Done |
| 4 | Machine-learning engine | ⏳ Next |
| 5 | Fraud prediction, risk intelligence, investigation | Planned |
| 6 | Executive dashboard, reports, integration | Planned |
| 7 | Testing & performance optimisation | Planned |
| 8 | Final documentation & release | Planned |

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for data science |
| Data | pandas, NumPy | Data loading and processing |
| ML | scikit-learn | Logistic Regression, Decision Tree, Random Forest, Isolation Forest, pipelines, metrics |
| App | Streamlit | One lightweight process; no separate frontend server |
| Charts | Plotly (app), Matplotlib (PDF) | Interactive in the app, static in reports |
| Storage | SQLite (built into Python) | Paginated tables without loading everything into memory |
| Testing | pytest | Unit, integration and Streamlit smoke tests |

Packages are added to `requirements.txt` only in the phase that first uses them.

## Architecture

```
Browser ──► Streamlit (app.py + views/)      UI only: displays results
                  │
                  ▼
           src/fraudshield/                  all logic, tested without the UI
           ├─ config.py   paths, seeds, risk thresholds
           ├─ data/       loading, validation, quality checks
           ├─ features/   feature engineering
           ├─ eda/        EDA summaries and Plotly charts
           ├─ models/     training, evaluation, saved models
           ├─ risk/       risk tiers
           ├─ reports/    CSV / PDF
           ├─ storage/    SQLite
           └─ ui/         theme, components, module registry
```

## Getting started

**Requirements:** Python 3.10 or newer, Git.

```bash
git clone https://github.com/<your-username>/fraudshield-ai.git
cd fraudshield-ai

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies (also installs the project package in editable mode)
pip install -r requirements-dev.txt

# Run the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Running tests

```bash
pytest          # all tests
ruff check .    # code-quality checks
```

## Project structure

```
fraudshield-ai/
├── app.py                 # entry point: page config, theme, sidebar navigation
├── views/                 # one page per module (UI only)
├── src/fraudshield/       # application logic package
├── scripts/               # offline jobs (sample creation, model training)
├── data/sample/           # small committed sample dataset
├── data/raw/              # full dataset, downloaded by you (git-ignored)
├── models/                # saved models and metadata
├── notebooks/             # analysis notebooks
├── tests/                 # pytest suite
├── docs/                  # requirements, diagrams, screenshots
└── .streamlit/config.toml # theme
```

## Dataset

The project uses the **Sparkov simulated credit-card transactions** dataset (Kaggle,
"Credit Card Transactions Fraud Detection Dataset"). It is synthetic, with readable
features such as merchant category, amount, time and customer location.

```bash
# 1. Download the dataset from Kaggle and copy fraudTrain.csv + fraudTest.csv to data/raw/
# 2. Build the sample used by the app (~50,000 rows from the test file)
python scripts/make_sample.py
```

Then open **Dataset Management** in the app, choose *Bundled sample dataset*, check the
data quality and save it. Any CSV can also be uploaded; its columns are matched to
FraudShield's standard columns first.

See [docs/dataset.md](docs/dataset.md) for the schema, validation rules and performance
measurements.

## License

MIT — see [LICENSE](LICENSE).
