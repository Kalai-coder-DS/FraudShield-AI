"""One summary of the active dataset, its scores and the model.

The Executive Dashboard, the CSV export and the PDF report all read from this
single summary, so the same number is never calculated twice in two different
ways. Every value comes from the stored transactions, the stored scores or the
model's metadata - nothing is typed in by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from fraudshield.config import RISK_TIERS
from fraudshield.data.schema import LABEL_FIELD
from fraudshield.eda.analysis import has_labels, key_findings, weekly_activity
from fraudshield.features.basic import add_basic_features
from fraudshield.risk.tiers import TIER_INFO

MIN_GROUP_SHARE = 0.005  # groups smaller than 0.5% of transactions are ignored in "top" lists


@dataclass
class Summary:
    generated_at: str
    dataset: dict
    totals: dict
    scoring: dict
    tiers: list[dict]
    labelled: dict | None
    model: dict
    top_categories: list[dict]
    weekly: pd.DataFrame = field(repr=False)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _lift_by_category(df: pd.DataFrame, risky: pd.Series, top: int = 5) -> list[dict]:
    if risky.sum() == 0:
        return []
    cats = df["category"].astype(str)
    share_all = cats.value_counts(normalize=True)
    share_risky = cats[risky].value_counts(normalize=True)
    counts = cats[risky].value_counts()
    table = pd.DataFrame({"all": share_all, "risky": share_risky, "count": counts}).fillna(0)
    table = table[table["all"] >= MIN_GROUP_SHARE]
    table["lift"] = table["risky"] / table["all"]
    table = table.sort_values("lift", ascending=False).head(top)
    return [
        {
            "category": name,
            "high_risk_transactions": int(row["count"]),
            "share_of_high_risk": float(row["risky"]),
            "share_of_all": float(row["all"]),
            "lift": float(row["lift"]),
        }
        for name, row in table.iterrows()
    ]


def build_summary(
    data: pd.DataFrame,
    scores: pd.DataFrame,
    metadata: dict,
    dataset_name: str,
    quality_score: float | None = None,
    now: datetime | None = None,
) -> Summary:
    """``data`` and ``scores`` must share the same index (row numbers)."""
    if not data.index.equals(scores.index):
        raise ValueError("Transactions and scores do not line up; re-score the dataset.")
    df = add_basic_features(data).join(scores)
    n = len(df)
    flagged = df["predicted_fraud"] == 1
    risky = df["risk_tier"].isin(["High", "Critical"])

    dataset = {
        "name": dataset_name,
        "transactions": n,
        "start": str(df["timestamp"].min())[:10],
        "end": str(df["timestamp"].max())[:10],
        "quality_score": quality_score,
        "has_labels": bool(has_labels(df)),
    }
    totals = {
        "transactions": n,
        "total_amount": float(df["amount"].sum()),
        "cards": int(df["card_id"].nunique()),
    }
    scoring = {
        "flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()) if n else 0.0,
        "flagged_amount": float(df.loc[flagged, "amount"].sum()),
        "high_or_critical": int(risky.sum()),
        "high_or_critical_amount": float(df.loc[risky, "amount"].sum()),
        "threshold": float(metadata["decision_threshold"]),
    }

    tiers = []
    for tier in RISK_TIERS:
        in_tier = df["risk_tier"] == tier
        row = {
            "tier": tier,
            "transactions": int(in_tier.sum()),
            "share": float(in_tier.mean()) if n else 0.0,
            "amount": float(df.loc[in_tier, "amount"].sum()),
            "action": TIER_INFO[tier].action,
            "known_fraud_rate": None,
        }
        if dataset["has_labels"]:
            known = df.loc[in_tier, LABEL_FIELD].dropna()
            row["known_fraud_rate"] = float((known == 1).mean()) if len(known) else None
        tiers.append(row)

    labelled = None
    if dataset["has_labels"]:
        known = df[df[LABEL_FIELD].notna()]
        is_fraud = known[LABEL_FIELD] == 1
        caught = int((is_fraud & (known["predicted_fraud"] == 1)).sum())
        false_alarms = int((~is_fraud & (known["predicted_fraud"] == 1)).sum())
        fraud_total = int(is_fraud.sum())
        labelled = {
            "known_fraud": fraud_total,
            "known_fraud_rate": float(is_fraud.mean()),
            "known_fraud_amount": float(known.loc[is_fraud, "amount"].sum()),
            "caught": caught,
            "missed": fraud_total - caught,
            "false_alarms": false_alarms,
            "recall": caught / fraud_total if fraud_total else None,
            "precision": caught / (caught + false_alarms) if caught + false_alarms else None,
            "fraud_amount_caught": float(
                known.loc[is_fraud & (known["predicted_fraud"] == 1), "amount"].sum()
            ),
        }

    selected = metadata["selected_model"]
    test = metadata["models"][selected]["test"]
    model = {
        "name": selected,
        "trained": metadata["created_at"][:10],
        "test_period": f"{metadata['data']['test']['start'][:10]} to "
        f"{metadata['data']['test']['end'][:10]}",
        "test_transactions": metadata["data"]["test"]["rows"],
        "threshold": float(metadata["decision_threshold"]),
        **{k: test[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "fraud_rate")},
        "comparison": [
            {
                "model": name,
                "pr_auc": m["test"]["pr_auc"],
                "recall_at_same_alerts": m["test"]["recall_at_best_alert_rate"],
            }
            for name, m in metadata["models"].items()
        ],
    }

    warnings = []
    train_end = pd.Timestamp(metadata["data"]["validation"]["end"])
    if df["timestamp"].min() <= train_end:
        warnings.append(
            "Part of this dataset falls inside the period the model learned from "
            f"(up to {train_end:%d %b %Y}), so results on it are optimistic. The honest "
            "test results are in the Model performance section."
        )

    top_categories = _lift_by_category(df, risky)
    findings = _findings(n, totals, scoring, labelled, top_categories, df)
    return Summary(
        generated_at=(now or datetime.now()).strftime("%Y-%m-%d %H:%M"),
        dataset=dataset,
        totals=totals,
        scoring=scoring,
        tiers=tiers,
        labelled=labelled,
        model=model,
        top_categories=top_categories,
        weekly=weekly_activity(df),
        findings=findings,
        warnings=warnings,
    )


def _findings(n, totals, scoring, labelled, top_categories, df) -> list[str]:
    out = [
        f"{scoring['flagged']:,} of {n:,} transactions ({scoring['flagged_share']:.2%}) are "
        f"flagged for review, worth {scoring['flagged_amount']:,.2f} in total "
        f"({scoring['flagged_amount'] / totals['total_amount']:.1%} of all transaction value)."
        if totals["total_amount"]
        else f"{scoring['flagged']:,} of {n:,} transactions are flagged for review."
    ]
    if labelled and labelled["known_fraud"]:
        out.append(
            f"Of {labelled['known_fraud']:,} known fraud cases in this dataset, the model "
            f"flagged {labelled['caught']:,} ({labelled['recall']:.1%}), with "
            f"{labelled['false_alarms']:,} false alarms."
        )
    if top_categories and top_categories[0]["lift"] > 1:
        top = top_categories[0]
        out.append(
            f"'{top['category']}' is the most over-represented category among High and "
            f"Critical transactions: {top['share_of_high_risk']:.1%} of them, against "
            f"{top['share_of_all']:.1%} of all transactions ({top['lift']:.1f}x)."
        )
    out += key_findings(df)[1:] if labelled else []
    return out


def risk_level_text(summary: Summary) -> str:
    """One plain-English headline for non-technical readers."""
    share = summary.scoring["high_or_critical"] / max(summary.dataset["transactions"], 1)
    return (
        f"{summary.scoring['high_or_critical']:,} transactions "
        f"({share:.2%}) need an analyst's attention today; the rest look normal."
    )


def format_probability(p: float) -> str:
    """Never show 100% or 0%: a model is never certain."""
    if p is None or np.isnan(p):
        return "—"
    if p > 0.999:
        return "> 99.9%"
    if p < 0.001:
        return "< 0.1%"
    return f"{p:.1%}"
