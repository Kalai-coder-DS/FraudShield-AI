"""Model evaluation for rare-event (fraud) classification.

Why not accuracy alone: if 0.5% of transactions are fraud, a "model" that
always says "legitimate" is 99.5% accurate and catches nothing. The metrics
that matter are:

* **Precision** - of the transactions we flag, how many are really fraud
  (low precision = many false alarms for analysts).
* **Recall** - of all fraud, how much we catch (low recall = lost money).
* **F1** - one number balancing precision and recall.
* **PR-AUC** (average precision) - quality of the ranking across all
  thresholds, focused on the rare fraud class. Its baseline is the fraud rate,
  not 0.5, so it is honest for imbalanced data. Main selection metric.
* **ROC-AUC** - ranking quality, but looks optimistic when fraud is rare,
  because the huge number of legitimate transactions makes the false-positive
  rate look small.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from fraudshield.config import RISK_TIERS, RiskThresholds

_CURVE_POINTS = 200


def ranking_metrics(y_true, score) -> dict:
    """Threshold-free metrics. ``score`` can be a probability or an anomaly score."""
    y_true = np.asarray(y_true)
    return {
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "fraud_rate": float(y_true.mean()),
    }


def threshold_metrics(y_true, proba, threshold: float) -> dict:
    """Metrics after turning probabilities into yes/no at ``threshold``."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "flagged": int(tp + fp),
    }


def full_metrics(y_true, proba, threshold: float) -> dict:
    out = ranking_metrics(y_true, proba)
    out.update(threshold_metrics(y_true, proba, threshold))
    out["brier"] = float(brier_score_loss(y_true, proba))
    out["baseline_accuracy"] = float(1 - np.mean(y_true))  # always answering "legitimate"
    return out


def best_f1_threshold(y_true, proba) -> float:
    """Threshold with the highest F1 score (chosen on VALIDATION data, never test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def recall_at_alert_rate(y_true, score, alert_rate: float) -> float:
    """Share of fraud caught when the top ``alert_rate`` of transactions are reviewed.

    Lets a model without probabilities (Isolation Forest) be compared with the
    others at the same analyst workload.
    """
    y_true = np.asarray(y_true)
    n_alerts = max(1, int(round(alert_rate * len(y_true))))
    top = np.argsort(-np.asarray(score), kind="stable")[:n_alerts]
    return float(y_true[top].sum() / max(y_true.sum(), 1))


def _downsample(*arrays, points: int = _CURVE_POINTS):
    n = len(arrays[0])
    idx = np.unique(np.linspace(0, n - 1, min(points, n)).astype(int))
    return [np.round(np.asarray(a)[idx], 5).tolist() for a in arrays]


def curves(y_true, score) -> dict:
    """PR and ROC curves, reduced to ~200 points so they are small to store."""
    precision, recall, _ = precision_recall_curve(y_true, score)
    fpr, tpr, _ = roc_curve(y_true, score)
    p, r = _downsample(precision, recall)
    f, t = _downsample(fpr, tpr)
    return {"pr": {"precision": p, "recall": r}, "roc": {"fpr": f, "tpr": t}}


def tier_table(y_true, proba, thresholds: RiskThresholds) -> list[dict]:
    """How many transactions fall in each risk tier, and how often they are fraud.

    Also gives precision and recall if everything from that tier upwards were
    flagged - the numbers that justify where the tier cut-offs are placed.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    cuts = [0.0, thresholds.medium, thresholds.high, thresholds.critical, 1.0000001]
    total_fraud = max(int(y_true.sum()), 1)
    rows = []
    for tier, low, high in zip(RISK_TIERS, cuts[:-1], cuts[1:], strict=True):
        in_tier = (proba >= low) & (proba < high)
        at_or_above = proba >= low
        n_above = int(at_or_above.sum())
        rows.append(
            {
                "tier": tier,
                "min_probability": low,
                "transactions": int(in_tier.sum()),
                "share": float(in_tier.mean()),
                "fraud": int(y_true[in_tier].sum()),
                "fraud_rate": float(y_true[in_tier].mean()) if in_tier.any() else None,
                "precision_if_flagged_from_here": float(y_true[at_or_above].sum() / n_above)
                if n_above
                else None,
                "recall_if_flagged_from_here": float(y_true[at_or_above].sum() / total_fraud),
            }
        )
    return rows
