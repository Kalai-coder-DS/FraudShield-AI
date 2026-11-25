"""Plotly charts for the Model Performance page, drawn from metadata.json."""

from __future__ import annotations

import plotly.graph_objects as go

from fraudshield.eda.charts import _style
from fraudshield.ui import theme as t

MODEL_COLORS = {
    "Random Forest": t.PRIMARY,
    "Logistic Regression": t.SECONDARY,
    "Decision Tree": "#A78BFA",
    "Isolation Forest": "#8D99AE",
}

FEATURE_LABELS = {
    "log_amount": "Amount",
    "hour": "Hour of day",
    "is_night": "Late night (22:00–03:59)",
    "weekday": "Day of week",
    "age_years": "Customer age",
    "distance_km": "Distance to merchant",
    "log_city_pop": "City population",
    "card_amount_z": "Amount unusual for this card",
    "card_known": "Card seen in training",
    "category": "Spending category",
}


def pr_curve_chart(models: dict, fraud_rate: float) -> go.Figure:
    fig = go.Figure()
    for name, m in models.items():
        pr = m["curves"]["pr"]
        fig.add_trace(
            go.Scatter(
                x=pr["recall"],
                y=pr["precision"],
                name=f"{name} (PR-AUC {m['test']['pr_auc']:.3f})",
                mode="lines",
                line=dict(color=MODEL_COLORS.get(name, t.TEXT_MUTED), width=2),
                hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra>" + name + "</extra>",
            )
        )
    fig.add_hline(
        y=fraud_rate,
        line_dash="dash",
        line_color=t.TEXT_MUTED,
        annotation_text=f"Random guessing ({fraud_rate:.2%})",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return _style(
        fig,
        "Precision vs. recall (test period)",
        "Recall: share of fraud caught",
        "Precision: share of alerts that are fraud",
        height=420,
    )


def roc_curve_chart(models: dict) -> go.Figure:
    fig = go.Figure()
    for name, m in models.items():
        roc = m["curves"]["roc"]
        fig.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                name=f"{name} (ROC-AUC {m['test']['roc_auc']:.3f})",
                mode="lines",
                line=dict(color=MODEL_COLORS.get(name, t.TEXT_MUTED), width=2),
                hovertemplate="False alarm rate %{x:.3f}<br>Fraud caught %{y:.2f}<extra>"
                + name
                + "</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random guessing",
            line=dict(color=t.TEXT_MUTED, dash="dash", width=1),
        )
    )
    return _style(
        fig,
        "ROC curve (test period)",
        "False alarm rate (legitimate flagged)",
        "Share of fraud caught",
        height=420,
    )


def confusion_chart(test: dict) -> go.Figure:
    """2x2 confusion matrix with plain-language labels."""
    z = [
        [test["true_negatives"], test["false_positives"]],
        [test["false_negatives"], test["true_positives"]],
    ]
    labels = [
        ["Correctly approved", "False alarm"],
        ["Fraud missed", "Fraud caught"],
    ]
    text = [[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)]
    # Colour by meaning, not by count: approved = neutral, false alarm = amber,
    # missed fraud = red, caught fraud = teal (the outcome we want).
    colors = [[0.0, 0.55], [0.8, 1.0]]
    fig = go.Figure(
        go.Heatmap(
            z=colors,
            x=["Predicted legitimate", "Predicted fraud"],
            y=["Actually legitimate", "Actually fraud"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color=t.TEXT),
            colorscale=[[0, t.CARD], [0.55, "#6B4A14"], [0.8, "#6B2A33"], [1, "#0F5C55"]],
            showscale=False,
            hoverinfo="skip",
            xgap=3,
            ygap=3,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "What the model decided (test period)", "", "", height=340)


def importance_chart(importance: list[dict]) -> go.Figure:
    rows = [r for r in importance if r["importance"] > 0][::-1] or importance[::-1]
    fig = go.Figure(
        go.Bar(
            x=[r["importance"] for r in rows],
            y=[FEATURE_LABELS.get(r["feature"], r["feature"]) for r in rows],
            orientation="h",
            marker_color=t.PRIMARY,
            hovertemplate="%{y}: PR-AUC drops by %{x:.3f} when shuffled<extra></extra>",
        )
    )
    return _style(
        fig,
        "What the selected model relies on",
        "Drop in PR-AUC when the feature is shuffled",
        "",
        height=max(300, 34 * len(rows) + 120),
    )
