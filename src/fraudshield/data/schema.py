"""The canonical transaction schema and column mapping.

A trained model can only score data that has the columns it was trained on.
Different CSV files name the same thing differently (``amt`` vs ``amount``), so
every dataset is first *mapped* onto one standard set of column names defined
here. The rest of the application only ever sees these canonical names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["id", "datetime", "date", "numeric", "category", "label"]


@dataclass(frozen=True)
class Field:
    name: str  # canonical column name used everywhere in the app
    kind: Kind  # how the column is parsed and validated
    required: bool  # a dataset cannot be used without it
    description: str  # plain-language meaning, shown in the mapping form
    aliases: tuple[str, ...] = ()  # other common names for the same column
    valid_range: tuple[float, float] | None = None  # numeric limits, inclusive


FIELDS: tuple[Field, ...] = (
    # --- Required: the minimum needed to analyse and score transactions ---
    Field(
        "transaction_id",
        "id",
        True,
        "Unique identifier of the transaction",
        ("trans_num", "txn_id", "transactionid", "transaction_number", "id"),
    ),
    Field(
        "timestamp",
        "datetime",
        True,
        "Date and time of the transaction",
        ("trans_date_trans_time", "transaction_time", "datetime", "transaction_date", "date"),
    ),
    Field(
        "amount",
        "numeric",
        True,
        "Transaction amount",
        ("amt", "transaction_amount", "value"),
        valid_range=(0.0, float("inf")),
    ),
    Field(
        "card_id",
        "id",
        True,
        "Card or account that made the transaction",
        ("cc_num", "card_number", "card", "account_id", "customer_id", "account"),
    ),
    Field(
        "category",
        "category",
        True,
        "Spending category or transaction type",
        ("merchant_category", "transaction_type", "type", "mcc"),
    ),
    # --- Optional: used for analysis and model features when present ---
    Field(
        "is_fraud",
        "label",
        False,
        "Known fraud label (1 = fraud, 0 = legitimate). Needed for training and accuracy checks",
        ("fraud", "isfraud", "is_fraudulent", "label", "class", "target"),
    ),
    Field("merchant", "category", False, "Merchant name", ("merchant_name",)),
    Field("gender", "category", False, "Cardholder gender", ("sex",)),
    Field("dob", "date", False, "Cardholder date of birth", ("date_of_birth", "birth_date")),
    Field("job", "category", False, "Cardholder occupation", ("occupation",)),
    Field("city", "category", False, "Cardholder city"),
    Field("state", "category", False, "Cardholder state or region", ("region",)),
    Field(
        "city_pop",
        "numeric",
        False,
        "Population of the cardholder's city",
        ("city_population", "population"),
        valid_range=(0.0, float("inf")),
    ),
    Field(
        "lat",
        "numeric",
        False,
        "Cardholder latitude",
        ("latitude", "customer_lat"),
        valid_range=(-90.0, 90.0),
    ),
    Field(
        "long",
        "numeric",
        False,
        "Cardholder longitude",
        ("longitude", "lon", "lng", "customer_long"),
        valid_range=(-180.0, 180.0),
    ),
    Field(
        "merch_lat",
        "numeric",
        False,
        "Merchant latitude",
        ("merchant_lat", "merchant_latitude"),
        valid_range=(-90.0, 90.0),
    ),
    Field(
        "merch_long",
        "numeric",
        False,
        "Merchant longitude",
        ("merchant_long", "merchant_lon", "merchant_longitude"),
        valid_range=(-180.0, 180.0),
    ),
)

FIELDS_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
REQUIRED_FIELDS: tuple[str, ...] = tuple(f.name for f in FIELDS if f.required)
LABEL_FIELD = "is_fraud"

# Personal columns in the Sparkov dataset that the app never needs. They are
# never mapped, so they are never loaded into memory or stored.
PERSONAL_COLUMNS: frozenset[str] = frozenset({"first", "last", "street"})

# A column mapping says, for each canonical field, which source column holds it
# (or None if the dataset does not have it).
Mapping = dict[str, str | None]


def _normalise(name: str) -> str:
    """'Trans Date-Trans_Time ' -> 'transdatetranstime' (for fuzzy matching)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def suggest_mapping(columns: list[str]) -> Mapping:
    """Guess which source column holds each canonical field.

    An exact (normalised) match on the canonical name wins over an alias, and a
    source column is never used for two fields.
    """
    normalised = {_normalise(c): c for c in columns}
    mapping: Mapping = {}
    used: set[str] = set()
    for pass_aliases in (False, True):
        for field in FIELDS:
            if mapping.get(field.name):
                continue
            candidates = field.aliases if pass_aliases else (field.name,)
            mapping.setdefault(field.name, None)
            for candidate in candidates:
                source = normalised.get(_normalise(candidate))
                if source is not None and source not in used:
                    mapping[field.name] = source
                    used.add(source)
                    break
    return mapping


def check_mapping(mapping: Mapping) -> list[str]:
    """Return plain-language problems with a mapping (empty list = valid)."""
    problems = [
        f"Required field '{name}' ({FIELDS_BY_NAME[name].description.lower()}) is not mapped."
        for name in REQUIRED_FIELDS
        if not mapping.get(name)
    ]
    chosen = [src for src in mapping.values() if src]
    duplicates = sorted({src for src in chosen if chosen.count(src) > 1})
    problems += [f"Column '{src}' is mapped to more than one field." for src in duplicates]
    unknown = sorted(set(mapping) - set(FIELDS_BY_NAME))
    problems += [f"Unknown field '{name}' in mapping." for name in unknown]
    return problems


def mapped_fields(mapping: Mapping) -> dict[str, str]:
    """Only the fields that are actually mapped: {canonical: source}."""
    return {name: src for name, src in mapping.items() if src}
