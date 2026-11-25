"""Plotly charts for the EDA module.

Each function receives a small summary table from ``analysis.py`` and returns
a styled figure. Fraud is always drawn in red and legitimate transactions
in a neutral grey-blue, so the colour meaning never changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fraudshield.eda.analysis import FRAUD, LEGIT
from fraudshield.ui import theme as t

LEGIT_COLOR = "#8D99AE"
CLASS_COLORS = {FRAUD: t.FRAUD, LEGIT: LEGIT_COLOR, "All": t.PRIMARY}
UNRELIABLE_COLOR = "#5C3A40"  # muted red: rate based on too few transactions
_CONFIG_FONT = dict(color=t.TEXT_MUTED, size=12)


def _style(fig: go.Figure, title: str, x_title: str, y_title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=t.TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_CONFIG_FONT,
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(bgcolor=t.CARD, bordercolor=t.CARD_BORDER, font_color=t.TEXT),
        bargap=0.2,
    )
    axis = dict(gridcolor=t.CARD_BORDER, zerolinecolor=t.CARD_BORDER, linecolor=t.CARD_BORDER)
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


# ---------------------------------------------------------------------------
# Overview and amounts
# ---------------------------------------------------------------------------


def class_balance_chart(balance: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=balance["class"],
            y=balance["transactions"],
            marker_color=[CLASS_COLORS[c] for c in balance["class"]],
            text=[
                f"{n:,}<br>({s:.2%})"
                for n, s in zip(balance["transactions"], balance["share"], strict=True)
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:,} transactions<extra></extra>",
        )
    )
    return _style(fig, "Fraud vs. legitimate transactions", "", "Number of transactions")


def amount_distribution_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name in (LEGIT, FRAUD, "All"):
        if name not in hist:
            continue
        fig.add_trace(
            go.Scatter(
                x=hist["bin_mid"],
                y=hist[name] * 100,
                name=name,
                mode="lines",
                line=dict(color=CLASS_COLORS[name], width=2, shape="hvh"),
                fill="tozeroy",
                opacity=0.85,
                customdata=np.stack([hist["bin_start"], hist["bin_end"]], axis=-1),
                hovertemplate=(
                    f"{name}<br>Amount %{{customdata[0]:,.2f}}–%{{customdata[1]:,.2f}}"
                    "<br>%{y:.2f}% of transactions<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(type="log")
    return _style(
        fig,
        "Transaction amount distribution",
        "Transaction amount (log scale)",
        "% of transactions in class",
    )


def amount_box_chart(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    order = (
        stats.groupby("group")["median"].max().sort_values(ascending=False).index.tolist()
        if len(stats)
        else []
    )
    for cls in (LEGIT, FRAUD, "All"):
        part = (
            stats[stats["class"] == cls].set_index("group").reindex(order).dropna(subset=["median"])
        )
        if part.empty:
            continue
        fig.add_trace(
            go.Box(
                name=cls,
                x=part.index,
                q1=part["q1"],
                median=part["median"],
                q3=part["q3"],
                lowerfence=part["lower_fence"],
                upperfence=part["upper_fence"],
                marker_color=CLASS_COLORS[cls],
                line_width=1.5,
            )
        )
    fig.update_layout(boxmode="group")
    fig.update_yaxes(type="log")
    return _style(
        fig,
        "Transaction amount by category",
        "Category",
        "Amount (log scale)",
        height=420,
    )


# ---------------------------------------------------------------------------
# Fraud rates
# ---------------------------------------------------------------------------


def fraud_rate_chart(
    table: pd.DataFrame,
    title: str,
    group_title: str,
    overall_rate: float,
    horizontal: bool = False,
) -> go.Figure:
    """Fraud rate per group with 95% intervals and the overall rate as a dashed line.

    Groups with too few transactions for a reliable rate are drawn in a muted colour.
    """
    data = table.dropna(subset=["fraud_rate"])
    if horizontal:
        data = data.iloc[::-1]  # highest rate at the top
    rate = data["fraud_rate"] * 100
    error = dict(
        type="data",
        symmetric=False,
        array=(data["ci_high"] * 100 - rate),
        arrayminus=(rate - data["ci_low"] * 100),
        color=t.TEXT_MUTED,
        thickness=1,
        width=3,
    )
    colors = [t.FRAUD if ok else UNRELIABLE_COLOR for ok in data["reliable"]]
    custom = np.stack(
        [data["transactions"], data["fraud"], data["ci_low"] * 100, data["ci_high"] * 100], axis=-1
    )
    hover = (
        "%{customdata[1]:,} fraud of %{customdata[0]:,} transactions"
        "<br>Fraud rate %{" + ("x" if horizontal else "y") + ":.2f}%"
        "<br>95% interval %{customdata[2]:.2f}%–%{customdata[3]:.2f}%<extra></extra>"
    )
    bar = go.Bar(
        x=rate if horizontal else data["group"],
        y=data["group"] if horizontal else rate,
        orientation="h" if horizontal else "v",
        marker_color=colors,
        customdata=custom,
        hovertemplate=hover,
        name="Fraud rate",
        **({"error_x": error} if horizontal else {"error_y": error}),
    )
    fig = go.Figure(bar)
    line = dict(line_dash="dash", line_color=t.SECONDARY, line_width=1.5)
    label = f"Overall {overall_rate * 100:.2f}%"
    if horizontal:
        fig.add_vline(x=overall_rate * 100, annotation_text=label, **line)
        fig = _style(
            fig, title, "Fraud rate (%)", group_title, height=max(380, 28 * len(data) + 120)
        )
    else:
        fig.add_hline(y=overall_rate * 100, annotation_text=label, **line)
        fig = _style(fig, title, group_title, "Fraud rate (%)")
        fig.update_xaxes(type="category")
    fig.update_layout(showlegend=False)
    return fig


def volume_chart(table: pd.DataFrame, title: str, group_title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=table["group"].astype(str),
            y=table["transactions"],
            marker_color=t.SECONDARY,
            hovertemplate="%{x}: %{y:,} transactions<extra></extra>",
        )
    )
    fig = _style(fig, title, group_title, "Number of transactions", height=320)
    fig.update_xaxes(type="category")
    return fig


def daily_activity_chart(daily: pd.DataFrame) -> go.Figure:
    has_fraud = "fraud" in daily
    fig = make_subplots(specs=[[{"secondary_y": has_fraud}]])
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["transactions"],
            name="All transactions",
            mode="lines",
            line=dict(color=LEGIT_COLOR, width=1.5),
            hovertemplate="%{x|%d %b %Y}: %{y:,} transactions<extra></extra>",
        ),
        secondary_y=False,
    )
    if has_fraud:
        fig.add_trace(
            go.Bar(
                x=daily["date"],
                y=daily["fraud"],
                name="Fraud transactions",
                marker_color=t.FRAUD,
                opacity=0.8,
                hovertemplate="%{x|%d %b %Y}: %{y:,} fraud<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Fraud transactions per day", secondary_y=True, showgrid=False)
    fig = _style(fig, "Transaction activity over time", "Date", "Transactions per day")
    if has_fraud:  # _style resets every y-axis title; restore the second one
        fig.update_yaxes(title_text="Fraud transactions per day", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "amount": "Amount",
    "hour": "Hour of day",
    "age_years": "Customer age",
    "city_pop": "City population",
    "distance_km": "Distance to merchant",
    "is_fraud": "Is fraud",
}


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    labels = [FEATURE_LABELS.get(c, c) for c in corr.columns]
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, t.PRIMARY], [0.5, t.CARD], [1.0, t.FRAUD]],
            texttemplate="%{z:.2f}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Correlation", outlinewidth=0),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Correlation between features (Spearman)", "", "", height=440)
