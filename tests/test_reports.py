"""Phase 6: report summary, CSV export, PDF report and weekly aggregation."""

import io

import numpy as np
import pandas as pd
import pytest

from fraudshield.eda.analysis import weekly_activity
from fraudshield.models import registry, scoring
from fraudshield.reports.csv_report import scored_transactions_csv
from fraudshield.reports.pdf_report import build_pdf
from fraudshield.reports.summary import build_summary, format_probability


@pytest.fixture(scope="module")
def scored(small_model, tmp_path_factory):
    model, test, result = small_model
    meta = registry.save(result, tmp_path_factory.mktemp("meta"))
    data = test.reset_index(drop=True)
    scores = scoring.score_frame(model, data, meta["decision_threshold"])
    return data, scores, meta


@pytest.fixture(scope="module")
def summary(scored):
    data, scores, meta = scored
    return build_summary(data, scores, meta, "Demo <set> & co", quality_score=99.5)


# --- Summary --------------------------------------------------------------------------


def test_summary_numbers_match_pandas(scored, summary):
    data, scores, _ = scored
    df = data.join(scores)
    assert summary.totals["transactions"] == len(df)
    assert summary.totals["total_amount"] == pytest.approx(df["amount"].sum())
    assert summary.scoring["flagged"] == int(df["predicted_fraud"].sum())
    assert summary.scoring["flagged_amount"] == pytest.approx(
        df.loc[df["predicted_fraud"] == 1, "amount"].sum()
    )
    assert sum(t["transactions"] for t in summary.tiers) == len(df)
    assert sum(t["amount"] for t in summary.tiers) == pytest.approx(df["amount"].sum())


def test_summary_detection_counts(scored, summary):
    data, scores, _ = scored
    df = data.join(scores)
    fraud = df["is_fraud"] == 1
    lab = summary.labelled
    assert lab["known_fraud"] == int(fraud.sum())
    assert lab["caught"] == int((fraud & (df["predicted_fraud"] == 1)).sum())
    assert lab["caught"] + lab["missed"] == lab["known_fraud"]
    assert lab["recall"] == pytest.approx(lab["caught"] / lab["known_fraud"])


def test_summary_findings_and_categories(summary):
    assert summary.findings and "flagged for review" in summary.findings[0]
    lifts = [c["lift"] for c in summary.top_categories]
    assert lifts == sorted(lifts, reverse=True)


def test_no_warning_for_data_after_training(summary):
    assert summary.warnings == []


def test_warning_when_data_overlaps_training(scored):
    data, scores, meta = scored
    early = dict(
        meta,
        data={**meta["data"], "validation": {**meta["data"]["validation"], "end": "2100-01-01"}},
    )
    s = build_summary(data, scores, early, "x")
    assert s.warnings and "optimistic" in s.warnings[0]


def test_misaligned_scores_are_rejected(scored):
    data, scores, meta = scored
    with pytest.raises(ValueError):
        build_summary(data, scores.iloc[1:], meta, "x")


def test_unlabelled_dataset(scored):
    data, scores, meta = scored
    s = build_summary(data.drop(columns="is_fraud"), scores, meta, "x")
    assert s.labelled is None
    assert all(t["known_fraud_rate"] is None for t in s.tiers)


@pytest.mark.parametrize(
    "p, text",
    [(1.0, "> 99.9%"), (0.9995, "> 99.9%"), (0.5, "50.0%"), (0.0, "< 0.1%"), (float("nan"), "—")],
)
def test_probability_never_shows_certainty(p, text):
    assert format_probability(p) == text


# --- Weekly activity -------------------------------------------------------------------


def test_weekly_drops_partial_edge_weeks():
    # Wednesday 1 Jan 2020 to Sunday 9 Feb 2020: first week is partial, last week complete.
    ts = pd.date_range("2020-01-01", "2020-02-09 23:00", freq="6h")
    df = pd.DataFrame({"timestamp": ts, "amount": 1.0})
    weekly = weekly_activity(df)
    assert weekly["week"].min() == pd.Timestamp("2020-01-06")  # first full week (Monday)
    assert weekly["week"].max() == pd.Timestamp("2020-02-03")
    assert (weekly["transactions"] == 28).all()


def test_weekly_keeps_short_datasets():
    ts = pd.date_range("2020-01-01", periods=10, freq="D")
    weekly = weekly_activity(pd.DataFrame({"timestamp": ts, "amount": 1.0}))
    assert weekly["transactions"].sum() == 10


# --- CSV ----------------------------------------------------------------------------------


def test_csv_masks_cards_and_sorts_by_risk(scored):
    data, scores, _ = scored
    raw = scored_transactions_csv(data, scores)
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8
    out = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    assert len(out) == len(data)
    assert out["card_id"].str.startswith("••••").all()
    assert not out["card_id"].str.contains(r"\d{5,}").any()
    assert out["fraud_probability"].is_monotonic_decreasing


def test_csv_filters(scored):
    data, scores, _ = scored
    high = pd.read_csv(
        io.BytesIO(scored_transactions_csv(data, scores, tiers=("High", "Critical"))),
        encoding="utf-8-sig",
    )
    assert set(high["risk_tier"]) <= {"High", "Critical"}
    assert len(high) == int(scores["risk_tier"].isin(["High", "Critical"]).sum())
    flagged = pd.read_csv(
        io.BytesIO(scored_transactions_csv(data, scores, flagged_only=True)), encoding="utf-8-sig"
    )
    assert len(flagged) == int(scores["predicted_fraud"].sum())


# --- PDF -----------------------------------------------------------------------------------


def test_pdf_contains_the_real_numbers(summary):
    pypdf = pytest.importorskip("pypdf")
    pdf = build_pdf(summary)
    assert pdf.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    assert 1 <= len(reader.pages) <= 4
    text = " ".join(" ".join(page.extract_text() for page in reader.pages).split())
    assert "Demo <set> & co" in text  # special characters survive escaping
    assert f"{summary.dataset['transactions']:,}" in text
    assert f"{summary.scoring['flagged']:,}" in text
    assert f"{summary.model['pr_auc']:.3f}" in text
    assert "not confirmed fraud decisions" in text


def test_pdf_works_without_labels(scored):
    data, scores, meta = scored
    s = build_summary(data.drop(columns="is_fraud"), scores, meta, "Unlabelled")
    assert build_pdf(s).startswith(b"%PDF")


def test_pdf_with_no_high_risk_transactions(scored):
    data, scores, meta = scored
    calm = scores.assign(risk_tier="Low", predicted_fraud=np.int8(0))
    assert build_pdf(build_summary(data, calm, meta, "Calm")).startswith(b"%PDF")
