"""Read-only dataset selection for a no-download dry run."""
from __future__ import annotations

from pathlib import Path


def local_datasets(config: dict) -> tuple[dict, list[str]]:
    """Return datasets that can be loaded without fetching external data.

    File-backed datasets are only loaded when all of their input files exist.
    A Hugging Face source is skipped because its loader may fetch data even
    when a cache exists. The original configuration is left untouched.
    """
    ready, unavailable = [], []
    for dataset in config.get("datasets", []):
        if not dataset.get("enabled", True):
            continue
        paths = [dataset[key] for key in ("path", "responses_path", "sources_path")
                 if dataset.get(key)]
        if dataset.get("source") == "hf" or any(not Path(path).is_file() for path in paths):
            unavailable.append(dataset["name"])
        else:
            ready.append(dataset)
    return {**config, "datasets": ready}, unavailable
