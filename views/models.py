"""Module 4 - Model Performance.

Reads models/metadata.json written by scripts/train_models.py. Nothing is
trained here, so the page opens instantly and always shows the results of the
model the app actually uses.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fraudshield.models import charts, registry
from fraudshield.ui.components import render_empty_state, render_header, render_kpi

PLOT_CONFIG = {"displaylogo": False}

render_header(
    "Model Performance",
    "How well each machine-learning model detects fraud, measured on transactions "
    "from a later period that the models never saw while learning.",
    eyebrow="Module 4 · Machine learning",
)

if not registry.metadata_exists():
    render_empty_state(
        "No trained model yet",
        "Put fraudTrain.csv and fraudTest.csv in data/raw/, then run "
        "'python scripts/train_models.py' in the terminal. Results appear here afterwards.",
    )
    st.stop()


@st.cache_data
def load(mtime: float) -> dict:  # mtime makes the cache refresh after retraining
    return registry.load_metadata()


meta = load(registry.metadata_path().stat().st_mtime)
models = meta["models"]
selected = meta["selected_model"]
best = models[selected]["test"]
data = meta["data"]
fraud_total = best["true_positives"] + best["false_negatives"]

st.caption(
    f"Trained {meta['created_at'].replace('T', ' ')} · scikit-learn {meta['sklearn_version']} · "
    f"tested on {data['test']['rows']:,} transactions from {data['test']['start'][:10]} "
    f"to {data['test']['end'][:10]}"
)

# --- Headline ---------------------------------------------------------------
cols = st.columns(4)
kpis = [
    ("Selected model", selected, "Highest PR-AUC on the validation period."),
    (
        "Fraud caught (recall)",
        f"{best['recall']:.1%}",
        f"{best['true_positives']:,} of {fraud_total:,} fraud cases in the test period.",
    ),
    (
        "Alerts that are fraud (precision)",
        f"{best['precision']:.1%}",
        f"{best['false_positives']:,} false alarms for {best['flagged']:,} alerts.",
    ),
    (
        "PR-AUC",
        f"{best['pr_auc']:.3f}",
        f"Ranking quality for rare fraud. Random guessing scores {best['fraud_rate']:.3f}.",
    ),
]
for col, (label, value, caption) in zip(cols, kpis, strict=True):
    with col:
        render_kpi(label, value, caption)

st.warning(
    f"**Why not accuracy?** Only {best['fraud_rate']:.2%} of test transactions are fraud. "
    f"A 'model' that approves everything would be {best['baseline_accuracy']:.2%} accurate "
    f"and catch no fraud at all. The selected model's accuracy is {best['accuracy']:.2%}, "
    "which says little on its own; precision, recall and PR-AUC show what matters.",
    icon=":material/lightbulb:",
)

tab_compare, tab_selected, tab_tiers, tab_details = st.tabs(
    ["Model comparison", "Selected model", "Risk tiers", "Data & method"]
)

# --- Comparison ---------------------------------------------------------------
with tab_compare:
    rows = []
    for name, m in models.items():
        t, v = m["test"], m["validation"]
        supervised = m["kind"] == "supervised"
        rows.append(
            {
                "Model": name + ("  ✓" if name == selected else ""),
                "Type": "Supervised" if supervised else "Unsupervised (anomaly)",
                "Validation PR-AUC": v.get("pr_auc_uncalibrated", v["pr_auc"]),
                "Test PR-AUC": t["pr_auc"],
                "Test ROC-AUC": t["roc_auc"],
                "Precision": t.get("precision"),
                "Recall": t.get("recall"),
                "F1": t.get("f1"),
                "Fraud caught at same alert volume": t["recall_at_best_alert_rate"],
                "Training time (s)": m["train_seconds"],
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {c: "{:.3f}" for c in table.columns[2:9]} | {"Training time (s)": "{:.1f}"},
            na_rep="—",
        ),
        hide_index=True,
    )
    st.caption(
        "The model is chosen by validation PR-AUC; test results are reported once and never "
        "used for choices. 'Fraud caught at same alert volume' compares all models when each "
        f"reviews the same top {best['alert_rate']:.2%} of transactions as the selected model. "
        "Isolation Forest does not use fraud labels, so it has no precision or recall at a "
        "probability threshold."
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.pr_curve_chart(models, best["fraud_rate"]), config=PLOT_CONFIG)
        st.caption(
            "**What this shows:** for each model, how precise its alerts are at every level of "
            "fraud caught. Higher and further right is better. This is the most honest view "
            "when fraud is rare."
        )
    with right:
        st.plotly_chart(charts.roc_curve_chart(models), config=PLOT_CONFIG)
        st.caption(
            "**What this shows:** fraud caught vs. false alarm rate. ROC curves look good for "
            "most models on rare-fraud data, because even many false alarms are a small share "
            "of the huge number of legitimate transactions."
        )

# --- Selected model -----------------------------------------------------------
with tab_selected:
    choice = st.selectbox(
        "Model",
        [n for n, m in models.items() if m["kind"] == "supervised"],
        index=list(models).index(selected),
    )
    test = models[choice]["test"]
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.confusion_chart(test), config=PLOT_CONFIG)
    with right:
        st.markdown(f"##### At decision threshold {test['threshold']:.2f}")
        st.markdown(
            f"- **{test['true_positives']:,}** fraud cases caught\n"
            f"- **{test['false_negatives']:,}** fraud cases missed\n"
            f"- **{test['false_positives']:,}** legitimate transactions flagged (false alarms)\n"
            f"- **{test['true_negatives']:,}** legitimate transactions correctly approved"
        )
        st.caption(
            "The threshold was chosen on the validation period to balance precision and "
            "recall (highest F1). A bank that fears missed fraud more would lower it and "
            "accept more false alarms."
        )
    st.plotly_chart(charts.importance_chart(meta["feature_importance"]), config=PLOT_CONFIG)
    st.caption(
        f"**What this shows:** how much the {selected} model's PR-AUC drops when one feature "
        "is shuffled randomly (permutation importance, measured on validation data). A large "
        "drop means the model depends on that feature."
    )

# --- Risk tiers ---------------------------------------------------------------
with tab_tiers:
    rt = meta["risk_thresholds"]
    st.markdown(
        f"Calibrated fraud probabilities are grouped into four tiers: **Low** below "
        f"{rt['medium']:.0%}, **Medium** from {rt['medium']:.0%}, **High** from {rt['high']:.0%} "
        f"and **Critical** from {rt['critical']:.0%}. The cut-offs are set in `config.py`; the "
        "table shows what they mean on the test period, so they can be justified with data."
    )
    tiers = pd.DataFrame(meta["tiers"])
    st.dataframe(
        tiers.rename(
            columns={
                "tier": "Tier",
                "min_probability": "From probability",
                "transactions": "Transactions",
                "share": "Share of all",
                "fraud": "Fraud in tier",
                "fraud_rate": "Fraud rate in tier",
                "precision_if_flagged_from_here": "Precision if this tier and above are flagged",
                "recall_if_flagged_from_here": "Recall if this tier and above are flagged",
            }
        ).style.format(
            {
                "From probability": "{:.2f}",
                "Share of all": "{:.2%}",
                "Fraud rate in tier": "{:.1%}",
                "Precision if this tier and above are flagged": "{:.1%}",
                "Recall if this tier and above are flagged": "{:.1%}",
            },
            na_rep="—",
        ),
        hide_index=True,
    )
    st.caption(
        "A well-calibrated model has a fraud rate in each tier that roughly matches its "
        "probability range: e.g. most Critical transactions should really be fraud."
    )

# --- Data & method --------------------------------------------------------------
with tab_details:
    periods = pd.DataFrame(
        [
            {
                "Period": label,
                "Rows": d["rows"],
                "Fraud": d["fraud"],
                "From": d["start"][:10],
                "To": d["end"][:10],
                "Used for": use,
            }
            for label, d, use in (
                ("Training", data["train"], "Fitting all models and feature statistics"),
                ("Validation", data["validation"], "Calibration, threshold, model choice"),
                ("Test", data["test"], "Final report only"),
            )
        ]
    )
    st.markdown("##### Time-based split")
    st.dataframe(periods, hide_index=True)
    st.caption(
        "Models always learn from earlier transactions and are tested on later ones, as in "
        "real use. A random split would let the model learn from the future."
    )
    st.markdown("##### Features used")
    st.write(
        ", ".join(charts.FEATURE_LABELS.get(f, f) for f in meta["features"]["numeric"])
        + ", Spending category"
    )
    st.markdown("##### Columns deliberately not used")
    st.dataframe(
        pd.DataFrame([{"Column": k, "Reason": v} for k, v in meta["excluded_features"].items()]),
        hide_index=True,
    )
    st.caption(
        f"Model file: {meta['model_file']} ({meta['model_size_mb']} MB), verified by its "
        "SHA-256 fingerprint before loading."
    )
