"""Score coverage checks.

A run passes when every expected row is present and every missing score is
explained: its row carries a recorded failure reason (e.g. a model that could
not answer one question). Silent gaps (a score missing with no reason) or
missing rows always fail. Recorded failures are allowed up to
MAX_FAILED_FRACTION of the rows, so one unanswerable question out of
thousands no longer fails a run that took days, while a broken detector or
model (many failures) still does. Every failure is listed in
score_completeness.json and in the report.
"""
from __future__ import annotations

import math
from collections import Counter

MAX_FAILED_FRACTION = 0.01


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def inspect_scores(rows: list[dict], columns: list[str], expected_rows: int,
                   key_columns: tuple[str, ...] = (), expected_keys: list[tuple] | None = None,
                   max_failed_fraction: float = MAX_FAILED_FRACTION) -> dict:
    missing = {column: 0 for column in columns}
    errors = []
    unexplained = 0
    failed_rows = 0
    for index, row in enumerate(rows):
        reasons = {key: _text(value) for key, value in row.items()
                   if key == "error" or key.endswith("_error")}
        reasons = {key: value for key, value in reasons.items() if value}
        for key, value in reasons.items():
            errors.append({"row": index, "field": key, "reason": value})
        row_missing = [column for column in columns if not _finite(row.get(column))]
        for column in row_missing:
            missing[column] += 1
        if row_missing or reasons:
            failed_rows += 1
        if row_missing and not reasons:
            unexplained += 1
    missing = {key: count for key, count in missing.items() if count}
    keys_match = True
    if expected_keys is not None:
        keys_match = Counter(tuple(row.get(key) for key in key_columns) for row in rows) == Counter(expected_keys)
    failed_fraction = failed_rows / len(rows) if rows else 0.0
    passed = (len(rows) == expected_rows and keys_match and unexplained == 0
              and failed_fraction <= max_failed_fraction)
    return {"passed": passed, "expected_rows": expected_rows, "recorded_rows": len(rows),
            "expected_answer_keys_match": keys_match, "rows_with_missing_scores": failed_rows,
            "unexplained_missing_rows": unexplained, "failed_fraction": round(failed_fraction, 5),
            "max_failed_fraction": max_failed_fraction,
            "missing_scores": missing, "errors": errors}


def require_scores(values: dict, columns: list[str]) -> dict:
    """Preflight requires every score of one case: no failure is tolerated."""
    missing = [key for key in columns if not _finite(values.get(key))]
    if missing or _text(values.get("__errors__")):
        raise RuntimeError(f"Incomplete detector scores: {missing}; "
                           f"errors: {values.get('__errors__') or 'none supplied'}")
    return values
