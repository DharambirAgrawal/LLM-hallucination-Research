"""Score coverage checks. Missing measurements never count as a successful run."""
from __future__ import annotations

import math
from collections import Counter


def inspect_scores(rows: list[dict], columns: list[str], expected_rows: int,
                   key_columns: tuple[str, ...] = (), expected_keys: list[tuple] | None = None) -> dict:
    missing = {column: 0 for column in columns}
    errors = []
    for index, row in enumerate(rows):
        for column in columns:
            try:
                valid = math.isfinite(float(row.get(column)))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                missing[column] += 1
        for key, value in row.items():
            if key == "error" or key.endswith("_error"):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                if str(value).strip():
                    errors.append({"row": index, "field": key, "reason": str(value)})
    missing = {key: count for key, count in missing.items() if count}
    keys_match = True
    if expected_keys is not None:
        keys_match = Counter(tuple(row.get(key) for key in key_columns) for row in rows) == Counter(expected_keys)
    return {"passed": len(rows) == expected_rows and keys_match and not missing and not errors,
            "expected_rows": expected_rows, "recorded_rows": len(rows),
            "expected_answer_keys_match": keys_match, "missing_scores": missing, "errors": errors}


def require_scores(values: dict, columns: list[str]) -> dict:
    """Preflight uses the same finite-score requirement as measured runs."""
    result = inspect_scores([{**{f"{key}_score": values.get(key) for key in columns},
                              "error": values.get("__errors__")}],
                            [f"{key}_score" for key in columns], 1)
    if not result["passed"]:
        raise RuntimeError(f"Incomplete detector scores: {result['missing_scores']}; "
                           f"errors: {values.get('__errors__') or 'none supplied'}")
    return values
