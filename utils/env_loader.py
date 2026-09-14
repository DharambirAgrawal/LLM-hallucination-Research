"""Minimal `.env` reader for API secrets; no third-party dependency required."""
from __future__ import annotations

import os
import re
from pathlib import Path


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(path: str | Path = ".env") -> set[str]:
    """Load simple KEY=VALUE entries without replacing existing environment values."""
    env_path = Path(path)
    if not env_path.is_file():
        return set()

    loaded = set()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _KEY.fullmatch(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded.add(key)
    return loaded
