from conftest import SPARKOV_COLUMNS

from fraudshield.data.schema import (
    FIELDS,
    PERSONAL_COLUMNS,
    REQUIRED_FIELDS,
    check_mapping,
    suggest_mapping,
)


def test_sparkov_columns_map_automatically():
    mapping = suggest_mapping(SPARKOV_COLUMNS)
    assert mapping["transaction_id"] == "trans_num"
    assert mapping["timestamp"] == "trans_date_trans_time"
    assert mapping["amount"] == "amt"
    assert mapping["card_id"] == "cc_num"
    assert mapping["is_fraud"] == "is_fraud"
    assert check_mapping(mapping) == []


def test_personal_columns_are_never_mapped():
    mapping = suggest_mapping(SPARKOV_COLUMNS)
    assert not set(mapping.values()) & PERSONAL_COLUMNS


def test_matching_ignores_case_spaces_and_symbols():
    mapping = suggest_mapping(["Transaction ID", "TIMESTAMP", "Amount", "Card-ID", "Category"])
    assert check_mapping(mapping) == []


def test_exact_name_wins_over_alias():
    mapping = suggest_mapping(["amount", "amt", "transaction_id", "timestamp", "card_id", "type"])
    assert mapping["amount"] == "amount"


def test_column_is_not_used_twice():
    mapping = suggest_mapping(["type", "transaction_id", "timestamp", "amount", "card_id"])
    used = [v for v in mapping.values() if v]
    assert len(used) == len(set(used))


def test_missing_required_fields_are_reported():
    problems = check_mapping({"transaction_id": "id", "amount": "amt"})
    assert len(problems) == len(REQUIRED_FIELDS) - 2
    assert any("timestamp" in p for p in problems)


def test_duplicate_source_column_is_reported():
    mapping = dict.fromkeys(REQUIRED_FIELDS, None)
    mapping.update(transaction_id="a", timestamp="b", amount="c", card_id="c", category="d")
    assert any("more than one field" in p for p in check_mapping(mapping))


def test_field_names_unique():
    names = [f.name for f in FIELDS]
    assert len(names) == len(set(names))
