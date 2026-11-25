"""Module 3 - Exploratory Data Analysis of the active dataset.

All numbers are calculated by fraudshield.eda.analysis from the stored
transactions. The summaries are cached per dataset, so switching tabs or
pages does not recalculate them.
"""

from __future__ import annotations

import streamlit as st

from fraudshield.eda import analysis as a
from fraudshield.eda import charts
from fraudshield.features.basic import add_basic_features
from fraudshield.storage import db
from fraudshield.ui.components import escape_markdown, render_empty_state, render_header, render_kpi
from fraudshield.ui.components import render_page_link

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

render_header(
    "Exploratory Analysis",
    "How fraudulent transactions differ from legitimate ones: amounts, timing, "
    "categories and customers.",
    eyebrow="Module 3 · Data",
)

active = db.get_active_dataset()
if active is None:
    render_empty_state(
        "No dataset to analyse",
        "Load and save a dataset in Dataset Management first. "
        "The analysis runs on the active dataset.",
    )
    render_page_link("views/dataset.py", "Go to Dataset Management", ":material/table_view:")
    st.stop()


@st.cache_data(show_spinner="Analysing transactions...", max_entries=3)
def compute(dataset_id: int) -> dict:
    """Load the dataset once and return only the small summary tables."""
    df = add_basic_features(db.load_transactions(dataset_id))
    labels = a.has_labels(df)
    result = {
        "labels": labels,
        "overview": a.overview(df),
        "amount_summary": a.amount_summary(df),
        "amount_hist": a.amount_histogram(df),
        "outliers": a.outliers_iqr(df),
        "daily": a.daily_activity(df),
        "volume_hour": a.volume_by(df, "hour", order=list(range(24))),
        "volume_category": a.volume_by(df, "category"),
        "correlation": a.correlation_matrix(df),
        "box": a.amount_box_stats(df),
        "has_age": "age_years" in df,
        "has_gender": "gender" in df,
    }
    if labels:
        result.update(
            balance=a.class_balance(df),
            findings=a.key_findings(df),
            rate_hour=a.fraud_rate_by_hour(df),
            rate_weekday=a.fraud_rate_by_weekday(df),
            rate_category=a.fraud_rate_by_category(df),
            rate_amount=a.fraud_rate_by_amount_band(df),
            rate_age=a.fraud_rate_by_age_band(df) if "age_years" in df else None,
            rate_gender=a.fraud_rate_by(df, "gender") if "gender" in df else None,
        )
    return result


r = compute(active.id)
ov = r["overview"]
base = ov["fraud_rate"]

st.caption(
    f"Dataset: {escape_markdown(active.name)} · "
    f"{ov['date_min']:%d %b %Y} – {ov['date_max']:%d %b %Y}"
)
kpis = [
    ("Transactions", f"{ov['transactions']:,}", f"Made with {ov['cards']:,} different cards."),
    ("Total value", f"{ov['total_amount']:,.0f}", "Sum of all transaction amounts."),
]
if r["labels"]:
    kpis += [
        ("Known fraud", f"{ov['fraud']:,}", f"{base:.2%} of labelled transactions."),
        ("Fraud value", f"{ov['fraud_amount']:,.0f}", "Total amount of fraud transactions."),
    ]
for col, (label, value, caption) in zip(st.columns(len(kpis)), kpis, strict=True):
    with col:
        render_kpi(label, value, caption)

if not r["labels"]:
    st.warning(
        "This dataset has no fraud labels, so only general patterns are shown. "
        "Fraud-rate charts need a dataset with a fraud label column.",
        icon=":material/warning:",
    )


def explain(text: str) -> None:
    st.caption(f"**What this shows:** {text}")


def show(fig) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG)


summary, amounts, timing, groups, relations = st.tabs(
    ["Summary", "Amounts", "Time patterns", "Categories & customers", "Relationships"]
)

# --- Summary ------------------------------------------------------------------
with summary:
    if r["labels"]:
        st.markdown("##### Key findings")
        for finding in r["findings"]:
            st.markdown(f"- {escape_markdown(finding)}")
        st.caption(
            "Findings are written from the numbers below. A group is only called out when "
            "it has at least 30 transactions and its difference from the overall rate is "
            "statistically clear (95% interval)."
        )
        show(charts.class_balance_chart(r["balance"]))
        explain(
            "how rare fraud is. Because almost every transaction is legitimate, a model "
            "can look very accurate while missing all fraud, so FraudShield judges models "
            "by precision and recall instead of accuracy."
        )
    show(charts.daily_activity_chart(r["daily"]))
    explain(
        "the number of transactions per day (grey line)"
        + (" and fraud transactions per day (red bars, right axis)." if r["labels"] else ".")
    )

# --- Amounts ------------------------------------------------------------------
with amounts:
    show(charts.amount_distribution_chart(r["amount_hist"]))
    explain(
        "how transaction amounts are spread. The amount axis is logarithmic, so small and "
        "very large amounts both fit. Each class adds up to 100%, which makes the shapes "
        "comparable even though fraud is rare."
    )
    st.markdown("##### Amount statistics")
    table = r["amount_summary"].rename(
        columns={
            "group": "Group",
            "transactions": "Transactions",
            "mean": "Mean",
            "median": "Median",
            "p25": "25th percentile",
            "p75": "75th percentile",
            "p95": "95th percentile",
            "max": "Maximum",
        }
    )
    st.dataframe(table.style.format(precision=2, thousands=","), hide_index=True)
    st.caption(
        "The median is the middle value; unlike the mean it is not pulled up by a few huge amounts."
    )

    out = r["outliers"]
    st.markdown("##### Unusually large or small amounts (outliers)")
    cols = st.columns(3)
    with cols[0]:
        render_kpi(
            "Outliers",
            f"{out['outliers']:,}",
            f"{out['outlier_share']:.1%} of transactions are outside "
            f"{max(out['lower_fence'], 0):,.2f} – {out['upper_fence']:,.2f}.",
        )
    if out["fraud_rate_outliers"] is not None:
        with cols[1]:
            render_kpi(
                "Fraud rate of outliers",
                f"{out['fraud_rate_outliers']:.2%}",
                "Among outlier amounts.",
            )
        with cols[2]:
            render_kpi(
                "Fraud rate of the rest",
                f"{out['fraud_rate_others']:.2%}",
                "Among all other amounts.",
            )
    st.caption(
        "Outliers use the 1.5 × IQR rule: values more than 1.5 times the middle 50% range "
        "below the 25th or above the 75th percentile. Outliers are kept, not removed: in "
        "fraud detection an unusual amount is often the signal itself."
    )
    if r["labels"]:
        show(
            charts.fraud_rate_chart(
                r["rate_amount"], "Fraud rate by amount range", "Amount range", base
            )
        )
        explain(
            "the share of transactions that are fraud in each amount range. Black lines are "
            "95% intervals; muted bars have fewer than 30 transactions and are less reliable."
        )

# --- Time patterns ----------------------------------------------------------
with timing:
    if r["labels"]:
        show(
            charts.fraud_rate_chart(
                r["rate_hour"], "Fraud rate by hour of day", "Hour of day", base
            )
        )
        explain(
            "at which hours fraud is most common. The dashed line is the overall fraud rate; "
            "bars well above it are higher-risk hours."
        )
        show(
            charts.fraud_rate_chart(
                r["rate_weekday"], "Fraud rate by day of week", "Day of week", base
            )
        )
        explain("whether fraud is more common on particular days of the week.")
    show(charts.volume_chart(r["volume_hour"], "Transactions by hour of day", "Hour of day"))
    explain(
        "when customers normally transact. "
        "Compare it with the fraud-rate chart to spot unusual hours."
    )

# --- Categories & customers -------------------------------------------------
with groups:
    if r["labels"]:
        show(
            charts.fraud_rate_chart(
                r["rate_category"],
                "Fraud rate by spending category",
                "Category",
                base,
                horizontal=True,
            )
        )
        explain("which spending categories carry the most fraud, from highest to lowest rate.")
        if r["rate_age"] is not None:
            show(
                charts.fraud_rate_chart(
                    r["rate_age"], "Fraud rate by customer age", "Age group", base
                )
            )
            explain(
                "whether some age groups are targeted more often. "
                "Age is calculated at the time of each transaction."
            )
        if r["rate_gender"] is not None:
            show(charts.fraud_rate_chart(r["rate_gender"], "Fraud rate by gender", "Gender", base))
            explain(
                "the fraud rate per gender as recorded in the data. This is shown for analysis "
                "only; using protected attributes like gender in a model needs a fairness review."
            )
    else:
        show(charts.volume_chart(r["volume_category"], "Transactions by category", "Category"))
        explain("how many transactions fall in each spending category.")

# --- Relationships ----------------------------------------------------------
with relations:
    if r["correlation"].shape[0] >= 2:
        show(charts.correlation_heatmap(r["correlation"]))
        explain(
            "whether two numeric features rise and fall together (+1), move in opposite "
            "directions (−1) or are unrelated (0). Spearman correlation is used because amounts "
            "are very skewed. A low correlation with 'Is fraud' does not mean a feature is "
            "useless: the relationship may be non-linear or appear only in combination."
        )
    else:
        st.info("Not enough numeric columns for a correlation analysis.", icon=":material/info:")
    if len(r["box"]):
        show(charts.amount_box_chart(r["box"]))
        explain(
            "the typical amount range per category. The box covers the middle 50% of amounts, "
            "the line inside is the median and the whiskers extend 1.5 × IQR."
            + (" Red boxes are fraud, grey boxes legitimate." if r["labels"] else "")
        )
