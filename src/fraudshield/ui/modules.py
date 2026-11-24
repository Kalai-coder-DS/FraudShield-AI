"""Single registry of the application's modules.

The sidebar navigation, the home page status table and each module page all
read from this list, so a module's name, icon and description are defined once.
When a module is finished, its ``ready`` flag is switched to True.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Module:
    key: str
    title: str
    icon: str  # Streamlit Material icon name
    section: str  # sidebar group
    phase: int  # development phase that delivers it
    summary: str  # one-line, non-technical purpose
    features: tuple[str, ...]  # what the finished module will do
    ready: bool = False


MODULES: tuple[Module, ...] = (
    Module(
        key="dashboard",
        title="Executive Dashboard",
        icon=":material/space_dashboard:",
        section="Overview",
        phase=6,
        summary=(
            "A one-screen summary of how many transactions were checked and how risky they are."
        ),
        features=(
            "Total transactions analysed and total transaction value",
            "Number and percentage of transactions flagged as suspicious",
            "Count of High and Critical risk transactions",
            "Risk distribution and suspicious activity over time",
        ),
    ),
    Module(
        key="dataset",
        title="Dataset Management",
        icon=":material/table_view:",
        section="Data",
        phase=2,
        summary="Upload transaction data and check its quality before any analysis.",
        features=(
            "Upload a CSV file or load the bundled sample dataset",
            "Preview rows, column names, data types and dimensions",
            "Detect missing values, duplicates and invalid values",
            "Validate required columns and map differently named columns",
        ),
        ready=True,
    ),
    Module(
        key="eda",
        title="Exploratory Analysis",
        icon=":material/query_stats:",
        section="Data",
        phase=3,
        summary="Charts that show how fraudulent transactions differ from normal ones.",
        features=(
            "Fraud vs. legitimate transaction balance",
            "Transaction amount distributions",
            "Fraud rate by hour, day and spending category",
            "Correlations and outliers",
        ),
    ),
    Module(
        key="models",
        title="Model Performance",
        icon=":material/model_training:",
        section="Machine Learning",
        phase=4,
        summary="How well each machine-learning model detects fraud, measured on unseen data.",
        features=(
            "Logistic Regression, Decision Tree, Random Forest and Isolation Forest",
            "Precision, recall, F1, ROC-AUC, PR-AUC and confusion matrices",
            "Side-by-side model comparison",
            "Why accuracy alone is misleading for rare fraud",
        ),
    ),
    Module(
        key="prediction",
        title="Fraud Prediction",
        icon=":material/psychology:",
        section="Machine Learning",
        phase=5,
        summary="Score a single transaction and see why the model rated it that way.",
        features=(
            "Pick a transaction or enter details manually",
            "Fraud probability, predicted class and risk tier",
            "Top factors behind the score",
        ),
    ),
    Module(
        key="risk",
        title="Risk Intelligence",
        icon=":material/shield:",
        section="Investigation",
        phase=5,
        summary=(
            "Group transactions into Low, Medium, High and Critical risk and explain the patterns."
        ),
        features=(
            "Documented, configurable risk thresholds",
            "High-risk transaction table",
            "Patterns among suspicious transactions",
        ),
    ),
    Module(
        key="investigation",
        title="Transaction Investigation",
        icon=":material/search:",
        section="Investigation",
        phase=5,
        summary="Search, filter and inspect any individual transaction in detail.",
        features=(
            "Search by transaction ID; filter by risk, amount, date and category",
            "Full transaction details with model score and explanation",
        ),
    ),
    Module(
        key="reports",
        title="Reports",
        icon=":material/description:",
        section="Investigation",
        phase=6,
        summary="Download the analysis results as CSV and PDF reports.",
        features=(
            "Scored transactions as CSV",
            "PDF report with dataset summary, fraud statistics, model performance and findings",
        ),
    ),
)


def get_module(key: str) -> Module:
    for module in MODULES:
        if module.key == key:
            return module
    raise KeyError(f"Unknown module: {key!r}")
