"""Small, deterministic run plans shared by both entry points."""
from __future__ import annotations


def configure_two_question_smoke(config: dict, detectors: tuple[str, ...]) -> str:
    """Keep two questions from the first enabled dataset and every method.

    Keeping only that dataset avoids preparing the other nine data sources.
    The normal run still uses every selected generator model and every
    configured reduction method.
    """
    first = next((ds for ds in config.get("datasets", []) if ds.get("enabled", True)), None)
    if first is None:
        raise ValueError("--smoke-2q needs at least one enabled dataset")
    config["datasets"] = [{**first, "max_samples": 2}]
    config.setdefault("run", {})["detectors"] = list(detectors)
    config["run"]["reduce"] = True
    return first["name"]
