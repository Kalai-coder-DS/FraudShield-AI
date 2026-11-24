"""Chart builders produce figures with titles and axis labels (needs Plotly)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from fraudshield.eda import analysis as a  # noqa: E402
from fraudshield.eda import charts  # noqa: E402
from fraudshield.features.basic import add_basic_features  # noqa: E402


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 1000
    data = pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(n)],
            "timestamp": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 30 * 86400, n), unit="s"),
            "amount": rng.lognormal(3, 1, n),
            "card_id": pd.Categorical(rng.integers(0, 20, n).astype(str)),
            "category": pd.Categorical(rng.choice(list("abc"), n)),
            "is_fraud": pd.array((rng.random(n) < 0.05).astype(int), dtype="Int8"),
        }
    )
    return add_basic_features(data)


def title(fig):
    return fig.layout.title.text


def test_all_charts_have_titles_and_axes(df):
    base = a.overview(df)["fraud_rate"]
    figures = [
        charts.class_balance_chart(a.class_balance(df)),
        charts.amount_distribution_chart(a.amount_histogram(df)),
        charts.amount_box_chart(a.amount_box_stats(df)),
        charts.fraud_rate_chart(a.fraud_rate_by_hour(df), "By hour", "Hour", base),
        charts.fraud_rate_chart(
            a.fraud_rate_by_category(df), "By cat", "Cat", base, horizontal=True
        ),
        charts.volume_chart(a.volume_by(df, "hour"), "Volume", "Hour"),
        charts.daily_activity_chart(a.daily_activity(df)),
        charts.correlation_heatmap(a.correlation_matrix(df)),
    ]
    for fig in figures:
        assert title(fig), fig
        assert len(fig.data) >= 1
    rate_fig = figures[3]
    assert rate_fig.layout.yaxis.title.text == "Fraud rate (%)"
    assert rate_fig.layout.xaxis.title.text == "Hour"


def test_fraud_is_always_red(df):
    fig = charts.class_balance_chart(a.class_balance(df))
    colors = dict(zip(fig.data[0].x, fig.data[0].marker.color, strict=True))
    assert colors["Fraud"] == charts.CLASS_COLORS["Fraud"]


def test_daily_chart_without_labels(df):
    fig = charts.daily_activity_chart(a.daily_activity(df.drop(columns="is_fraud")))
    assert len(fig.data) == 1


def test_rate_chart_values_are_percentages(df):
    table = a.fraud_rate_by_weekday(df)
    fig = charts.fraud_rate_chart(table, "t", "Day", 0.05)
    assert list(fig.data[0].y) == pytest.approx((table["fraud_rate"] * 100).tolist())
