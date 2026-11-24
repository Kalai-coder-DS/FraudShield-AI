"""EDA calculations.

Every function takes the cleaned transactions (with ``add_basic_features``
applied) and returns a small summary table. Charts are drawn from these
summaries, never from raw rows, so pages stay fast even for a million rows.

Fraud rates are always calculated on labelled rows only. Each rate comes with a
95% confidence interval (Wilson method): a fraud rate measured on 20
transactions is much less certain than one measured on 20,000, and the interval
makes that visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.data.schema import LABEL_FIELD
from fraudshield.features.basic import WEEKDAY_NAMES

FRAUD, LEGIT = "Fraud", "Legitimate"
MIN_GROUP_SIZE = 30  # groups smaller than this are flagged as unreliable

AMOUNT_BANDS = (0, 10, 25, 50, 100, 250, 500, 1000, np.inf)
AGE_BANDS = (0, 25, 35, 45, 55, 65, 200)
AGE_LABELS = ("Under 25", "25–34", "35–44", "45–54", "55–64", "65+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def has_labels(df: pd.DataFrame) -> bool:
    return LABEL_FIELD in df and df[LABEL_FIELD].notna().any()


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[LABEL_FIELD].notna()] if LABEL_FIELD in df else df.iloc[0:0]


def label_mask(df: pd.DataFrame, value: int) -> np.ndarray:
    """True where the label equals ``value``; unlabelled rows are always False."""
    return df[LABEL_FIELD].eq(value).fillna(False).to_numpy(dtype=bool)


def class_name(df: pd.DataFrame) -> np.ndarray:
    return np.where(label_mask(df, 1), FRAUD, LEGIT)


def wilson_interval(k, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% confidence interval for a proportion k/n (Wilson score method)."""
    k = np.asarray(k, dtype="float64")
    n = np.asarray(n, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    low = np.where(n > 0, np.clip(centre - half, 0, 1), np.nan)
    high = np.where(n > 0, np.clip(centre + half, 0, 1), np.nan)
    return low, high


def _band_labels(edges) -> list[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"{lo:,.0f}+" if np.isinf(hi) else f"{lo:,.0f}–{hi:,.0f}")
    return labels


def amount_band(amount: pd.Series) -> pd.Series:
    return pd.cut(amount, AMOUNT_BANDS, labels=_band_labels(AMOUNT_BANDS), right=False)


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, labels=AGE_LABELS, right=False)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def overview(df: pd.DataFrame) -> dict:
    """Headline numbers for the dataset."""
    out = {
        "transactions": len(df),
        "total_amount": float(df["amount"].sum()),
        "cards": int(df["card_id"].nunique()),
        "date_min": df["timestamp"].min(),
        "date_max": df["timestamp"].max(),
        "labelled": 0,
        "fraud": None,
        "fraud_rate": None,
        "fraud_amount": None,
    }
    if has_labels(df):
        lab = labelled(df)
        fraud = label_mask(lab, 1)
        out.update(
            labelled=len(lab),
            fraud=int(fraud.sum()),
            fraud_rate=float(fraud.mean()),
            fraud_amount=float(lab.loc[fraud, "amount"].sum()),
        )
    return out


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    lab = labelled(df)
    counts = pd.Series(class_name(lab)).value_counts().reindex([LEGIT, FRAUD], fill_value=0)
    amounts = lab.groupby(class_name(lab))["amount"].sum().reindex([LEGIT, FRAUD], fill_value=0)
    return pd.DataFrame(
        {
            "class": counts.index,
            "transactions": counts.to_numpy(),
            "share": counts.to_numpy() / max(len(lab), 1),
            "amount": amounts.to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------


def amount_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Amount statistics overall and, if labelled, per class."""
    groups = {"All": df["amount"]}
    if has_labels(df):
        lab = labelled(df)
        groups[LEGIT] = lab.loc[lab[LABEL_FIELD] == 0, "amount"]
        groups[FRAUD] = lab.loc[lab[LABEL_FIELD] == 1, "amount"]
    rows = []
    for name, s in groups.items():
        rows.append(
            {
                "group": name,
                "transactions": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "p25": s.quantile(0.25),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def amount_histogram(df: pd.DataFrame, bins: int = 40) -> pd.DataFrame:
    """Share of each class's transactions in log-spaced amount bins.

    Shares are calculated *within* each class. With raw counts the few fraud
    transactions would be invisible next to the legitimate ones.
    """
    amounts = df["amount"].clip(lower=0.01)
    edges = np.logspace(np.log10(amounts.min()), np.log10(amounts.max() * 1.0001), bins + 1)
    idx = np.clip(np.searchsorted(edges, amounts, side="right") - 1, 0, bins - 1)
    result = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:]})
    result["bin_mid"] = np.sqrt(result["bin_start"] * result["bin_end"])

    groups = {"All": np.ones(len(df), dtype=bool)}
    if has_labels(df):
        groups = {LEGIT: label_mask(df, 0), FRAUD: label_mask(df, 1)}
    for name, mask in groups.items():
        counts = np.bincount(idx[mask], minlength=bins)
        result[name] = counts / max(counts.sum(), 1)
    return result


def outliers_iqr(df: pd.DataFrame, column: str = "amount") -> dict:
    """Outliers by the 1.5 × IQR rule, and how often they are fraud."""
    s = df[column]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_out = (s < low) | (s > high)
    out = {
        "column": column,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(low),
        "upper_fence": float(high),
        "outliers": int(is_out.sum()),
        "outlier_share": float(is_out.mean()),
        "fraud_rate_outliers": None,
        "fraud_rate_others": None,
    }
    if has_labels(df):
        known = df[LABEL_FIELD].notna().to_numpy()
        fraud = label_mask(df, 1)
        is_out = is_out.to_numpy()
        n_out, n_in = (is_out & known).sum(), (~is_out & known).sum()
        out["fraud_rate_outliers"] = float((fraud & is_out).sum() / n_out) if n_out else None
        out["fraud_rate_others"] = float((fraud & ~is_out).sum() / n_in) if n_in else None
    return out


# ---------------------------------------------------------------------------
# Fraud rate by group
# ---------------------------------------------------------------------------


def fraud_rate_by(
    df: pd.DataFrame,
    key: str | pd.Series,
    order: list | None = None,
    sort_by_rate: bool = False,
) -> pd.DataFrame:
    """Transactions, fraud count, fraud rate and 95% interval per group."""
    lab = labelled(df)
    groups = key.loc[lab.index] if isinstance(key, pd.Series) else lab[key]
    fraud = (lab[LABEL_FIELD] == 1).astype(int)
    stats = fraud.groupby(groups, observed=True).agg(["size", "sum"])
    stats.columns = ["transactions", "fraud"]
    if order is not None:
        stats = stats.reindex(order, fill_value=0)
    stats["fraud_rate"] = np.where(
        stats["transactions"] > 0, stats["fraud"] / stats["transactions"].clip(lower=1), np.nan
    )
    stats["ci_low"], stats["ci_high"] = wilson_interval(stats["fraud"], stats["transactions"])
    stats["reliable"] = stats["transactions"] >= MIN_GROUP_SIZE
    stats = stats.rename_axis("group").reset_index()
    stats["group"] = stats["group"].astype(str)
    if sort_by_rate:
        stats = stats.sort_values("fraud_rate", ascending=False, kind="stable")
    return stats.reset_index(drop=True)


def fraud_rate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return fraud_rate_by(df, "hour", order=list(range(24)))


def fraud_rate_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    result = fraud_rate_by(df, "weekday", order=list(range(7)))
    result["group"] = list(WEEKDAY_NAMES)
    return result


def fraud_rate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return fraud_rate_by(df, "category", sort_by_rate=True)


def fraud_rate_by_amount_band(df: pd.DataFrame) -> pd.DataFrame:
    return fraud_rate_by(df, amount_band(df["amount"]), order=_band_labels(AMOUNT_BANDS))


def fraud_rate_by_age_band(df: pd.DataFrame) -> pd.DataFrame:
    return fraud_rate_by(df, age_band(df["age_years"]), order=list(AGE_LABELS))


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def volume_by(df: pd.DataFrame, key: str, order: list | None = None) -> pd.DataFrame:
    """Transaction count and total amount per group (works without labels)."""
    stats = df.groupby(key, observed=True)["amount"].agg(["size", "sum"])
    stats.columns = ["transactions", "amount"]
    if order is not None:
        stats = stats.reindex(order, fill_value=0)
    return stats.rename_axis("group").reset_index()


def daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Transactions per day, and fraud count and rate when labels exist."""
    daily = df.groupby("date")["amount"].agg(["size", "sum"])
    daily.columns = ["transactions", "amount"]
    if has_labels(df):
        lab = labelled(df)
        by_day = (lab[LABEL_FIELD] == 1).astype(int).groupby(lab["date"]).agg(["size", "sum"])
        daily["fraud"] = by_day["sum"].reindex(daily.index, fill_value=0)
        daily["fraud_rate"] = by_day["sum"] / by_day["size"]
    return daily.rename_axis("date").reset_index()


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

CORRELATION_COLUMNS = ("amount", "hour", "age_years", "city_pop", "distance_km", LABEL_FIELD)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation between numeric features.

    Spearman measures whether two values rise and fall together, without
    assuming a straight-line relationship. It is robust to the very skewed
    amounts in transaction data, where a few huge values would dominate the
    usual (Pearson) correlation.
    """
    cols = [c for c in CORRELATION_COLUMNS if c in df and df[c].notna().any()]
    data = df[cols].astype("float64")
    data = data.loc[:, data.nunique() > 1]  # a constant column has no correlation
    return data.corr(method="spearman")


def amount_box_stats(df: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """Box-plot statistics of amount per group (and per class when labelled).

    Calculated here so the chart receives a few numbers per group instead of
    every transaction.
    """
    data = df.assign(_class=class_name(df) if has_labels(df) else "All")
    if has_labels(df):
        data = data[data[LABEL_FIELD].notna()]
    rows = []
    for (group, cls), s in data.groupby([by, "_class"], observed=True)["amount"]:
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        rows.append(
            {
                "group": str(group),
                "class": cls,
                "count": len(s),
                "q1": q1,
                "median": med,
                "q3": q3,
                "lower_fence": max(s.min(), q1 - 1.5 * iqr),
                "upper_fence": min(s.max(), q3 + 1.5 * iqr),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plain-language findings
# ---------------------------------------------------------------------------


def key_findings(df: pd.DataFrame) -> list[str]:
    """Short findings written from the calculated numbers.

    Only groups with at least MIN_GROUP_SIZE labelled transactions are used,
    so a tiny group with 1 fraud out of 2 cannot be reported as "50% fraud",
    and a group is only called out when the difference is statistically clear.
    """
    if not has_labels(df):
        return []
    ov = overview(df)
    base = ov["fraud_rate"]
    findings = [
        f"{ov['fraud']:,} of {ov['labelled']:,} labelled transactions are fraud "
        f"({base:.2%}). A model that always answered 'legitimate' would be "
        f"{1 - base:.2%} accurate and catch no fraud at all."
    ]
    if base == 0 or ov["fraud"] == ov["labelled"]:
        return findings

    summary = amount_summary(df).set_index("group")
    fraud_med, legit_med = summary.loc[FRAUD, "median"], summary.loc[LEGIT, "median"]
    if legit_med > 0:
        findings.append(
            f"The typical (median) fraud transaction is {fraud_med:,.2f}, compared with "
            f"{legit_med:,.2f} for legitimate ones ({fraud_med / legit_med:.1f}× as large)."
        )

    def stand_out(table: pd.DataFrame) -> pd.Series | None:
        """Highest-rate reliable group, only if it is clearly above the overall rate.

        "Clearly" means the lower end of its 95% interval is above the overall
        rate, so random variation alone is unlikely to explain the difference.
        """
        reliable = table[table["reliable"]]
        if reliable.empty:
            return None
        best = reliable.loc[reliable["fraud_rate"].idxmax()]
        return best if best["ci_low"] > base else None

    cat = stand_out(fraud_rate_by_category(df))
    if cat is not None:
        findings.append(
            f"'{cat['group']}' has the highest fraud rate of any category: "
            f"{cat['fraud_rate']:.2%}, which is {cat['fraud_rate'] / base:.1f}× the overall rate."
        )
    else:
        findings.append("No spending category has a fraud rate clearly above the overall rate.")
    hour = stand_out(fraud_rate_by_hour(df))
    if hour is not None:
        h = int(hour["group"])
        findings.append(
            f"Fraud is most likely between {h:02d}:00 and {h:02d}:59, at "
            f"{hour['fraud_rate']:.2%} ({hour['fraud_rate'] / base:.1f}× the overall rate)."
        )
    else:
        findings.append("No hour of the day has a fraud rate clearly above the overall rate.")
    out = outliers_iqr(df)
    if out["fraud_rate_outliers"] is not None and out["fraud_rate_others"]:
        findings.append(
            f"{out['outlier_share']:.1%} of transactions are unusually large (above "
            f"{out['upper_fence']:,.2f}). Their fraud rate is {out['fraud_rate_outliers']:.2%}, "
            f"against {out['fraud_rate_others']:.2%} for the rest."
        )
    return findings
