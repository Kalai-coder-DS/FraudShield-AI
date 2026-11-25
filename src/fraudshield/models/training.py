"""Training workflow.

    training file (earlier dates)                       test file (later dates)
    |------------- train 80% ------------|-- val 20% --|   |------- test -------|
      fit pipelines                        calibrate,         reported once,
      (incl. card profiles, scaling,       pick threshold,    never used for
       encoding - all learned here)        choose best model  any decision

* **Time-based split**: the model is always tested on transactions that happen
  after the ones it learned from - like in real use. A random split would let
  the model "see the future".
* **Everything learned lives inside one sklearn Pipeline**, so imputation,
  scaling, encoding and card profiles are fitted on the training rows only.
* **Class imbalance**: ``class_weight="balanced"`` makes a missed fraud cost as
  much in training as all the legitimate transactions together. This changes
  the scale of the scores, so probabilities are **calibrated** afterwards on
  the validation period: a calibrated 0.7 means about 70% of such transactions
  were fraud.
* **Model choice and threshold** use validation data only. The test set is
  used once, for the final report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fraudshield.config import RANDOM_SEED, SETTINGS
from fraudshield.data.schema import LABEL_FIELD
from fraudshield.features.engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    CardProfile,
    add_model_features,
)
from fraudshield.models import evaluation as ev

SUPERVISED_MODELS = ("Logistic Regression", "Decision Tree", "Random Forest")
ANOMALY_MODEL = "Isolation Forest"


def make_estimator(name: str, n_jobs: int = -1, rf_trees: int = 150):
    """The model at the end of the pipeline. Settings are kept small for a laptop."""
    if name == "Logistic Regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_SEED
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=14,
            min_samples_leaf=20,
            max_samples=0.3,  # each tree sees 30% of rows: much faster, little accuracy lost
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=RANDOM_SEED,
        )
    if name == ANOMALY_MODEL:
        return IsolationForest(n_estimators=150, random_state=RANDOM_SEED, n_jobs=n_jobs)
    raise ValueError(f"Unknown model: {name}")


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(name: str, **kwargs) -> Pipeline:
    return Pipeline(
        [
            ("row_features", FunctionTransformer(add_model_features)),
            ("card_profile", CardProfile()),
            ("preprocess", make_preprocessor()),
            ("model", make_estimator(name, **kwargs)),
        ]
    )


def time_split(
    df: pd.DataFrame, validation_share: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier rows for training, the latest ``validation_share`` for validation."""
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut = int(len(df) * (1 - validation_share))
    return df.iloc[:cut], df.iloc[cut:]


def labelled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[LABEL_FIELD].notna()]


@dataclass
class ModelResult:
    name: str
    kind: str  # "supervised" or "anomaly"
    params: dict
    train_seconds: float
    validation: dict
    test: dict
    curves: dict
    model: object = field(repr=False, default=None)  # calibrated pipeline (supervised only)


@dataclass
class TrainingResult:
    results: dict[str, ModelResult]
    selected: str
    tiers: list[dict]
    importance: list[dict]
    data_info: dict


def _anomaly_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    # IsolationForest: lower score_samples = more abnormal, so negate (higher = riskier)
    return -pipeline.score_samples(X)


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: tuple[str, ...] = (*SUPERVISED_MODELS, ANOMALY_MODEL),
    n_jobs: int = -1,
    rf_trees: int = 150,
    log=print,
) -> TrainingResult:
    train_df, test_df = labelled_only(train_df), labelled_only(test_df)
    fit_df, val_df = time_split(train_df)
    y_fit = fit_df[LABEL_FIELD].astype(int).to_numpy()
    y_val = val_df[LABEL_FIELD].astype(int).to_numpy()
    y_test = test_df[LABEL_FIELD].astype(int).to_numpy()
    for name, y in (("training", y_fit), ("validation", y_val), ("test", y_test)):
        if y.sum() == 0 or y.sum() == len(y):
            raise ValueError(f"The {name} period needs both fraud and legitimate transactions.")

    results: dict[str, ModelResult] = {}
    for name in models:
        log(f"Training {name}...")
        start = time.perf_counter()
        pipe = build_pipeline(name, n_jobs=n_jobs, rf_trees=rf_trees)
        params = {
            k: v
            for k, v in pipe.named_steps["model"].get_params().items()
            if isinstance(v, int | float | str | bool) or v is None
        }
        if name == ANOMALY_MODEL:
            pipe.fit(fit_df)  # unsupervised: labels are NOT used
            seconds = time.perf_counter() - start
            val_score, test_score = _anomaly_scores(pipe, val_df), _anomaly_scores(pipe, test_df)
            results[name] = ModelResult(
                name,
                "anomaly",
                params,
                seconds,
                ev.ranking_metrics(y_val, val_score),
                ev.ranking_metrics(y_test, test_score),
                ev.curves(y_test, test_score),
            )
            results[name].test["_score"] = test_score  # removed before saving
        else:
            pipe.fit(fit_df, y_fit)
            calibrated = CalibratedClassifierCV(
                FrozenEstimator(pipe), method="isotonic", ensemble=False
            )
            calibrated.fit(val_df, y_val)
            seconds = time.perf_counter() - start
            raw_val = pipe.predict_proba(val_df)[:, 1]
            val_proba = calibrated.predict_proba(val_df)[:, 1]
            threshold = ev.best_f1_threshold(y_val, val_proba)
            test_proba = calibrated.predict_proba(test_df)[:, 1]
            validation = ev.full_metrics(y_val, val_proba, threshold)
            validation["pr_auc_uncalibrated"] = float(ev.ranking_metrics(y_val, raw_val)["pr_auc"])
            results[name] = ModelResult(
                name,
                "supervised",
                params,
                seconds,
                validation,
                ev.full_metrics(y_test, test_proba, threshold),
                ev.curves(y_test, test_proba),
                model=calibrated,
            )
            results[name].test["_proba"] = test_proba
        r = results[name]
        log(f"  done in {r.train_seconds:.1f}s, test PR-AUC {r.test['pr_auc']:.3f}")

    supervised = [r for r in results.values() if r.kind == "supervised"]
    if not supervised:
        raise ValueError("At least one supervised model is needed.")
    best = max(supervised, key=lambda r: r.validation["pr_auc_uncalibrated"])
    best_proba = best.test.pop("_proba")

    # Compare every model at the same analyst workload: the share the best model flags.
    alert_rate = best.test["flagged"] / len(y_test)
    for r in results.values():
        score = r.test.pop("_proba", None)
        score = r.test.pop("_score", score)
        source = best_proba if r is best else score
        r.test["recall_at_best_alert_rate"] = ev.recall_at_alert_rate(y_test, source, alert_rate)
    best.test["alert_rate"] = alert_rate

    log("Calculating feature importance...")
    importance = grouped_permutation_importance(best.model, val_df, y_val)
    return TrainingResult(
        results=results,
        selected=best.name,
        tiers=ev.tier_table(y_test, best_proba, SETTINGS.risk_thresholds),
        importance=importance,
        data_info=_data_info(fit_df, val_df, test_df),
    )


def _period(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "fraud": int((df[LABEL_FIELD] == 1).sum()),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }


def _data_info(fit_df, val_df, test_df) -> dict:
    return {"train": _period(fit_df), "validation": _period(val_df), "test": _period(test_df)}


def grouped_permutation_importance(
    calibrated_model, X: pd.DataFrame, y, max_rows: int = 20_000, repeats: int = 3
) -> list[dict]:
    """How much PR-AUC drops when one feature's values are shuffled.

    Shuffling breaks the link between that feature and fraud; a big drop means
    the model relies on it. One-hot category columns are shuffled together as
    one feature ("category"). Calculated on validation data.
    """
    pipe: Pipeline = calibrated_model.calibrated_classifiers_[0].estimator.estimator
    if len(X) > max_rows:
        keep = np.random.default_rng(RANDOM_SEED).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[np.sort(keep)], np.asarray(y)[np.sort(keep)]
    y = np.asarray(y)
    transformed = pipe[:-1].transform(X)
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    model = pipe.named_steps["model"]
    base = ev.ranking_metrics(y, model.predict_proba(transformed)[:, 1])["pr_auc"]

    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        key = "category" if n.startswith("category_") else n
        groups.setdefault(key, []).append(i)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for feature, cols in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = transformed.copy()
            shuffled[:, cols] = shuffled[rng.permutation(len(shuffled))][:, cols]
            score = ev.ranking_metrics(y, model.predict_proba(shuffled)[:, 1])["pr_auc"]
            drops.append(base - score)
        rows.append({"feature": feature, "importance": float(np.mean(drops))})
    return sorted(rows, key=lambda r: r["importance"], reverse=True)
