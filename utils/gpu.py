"""How big the GPU is, read without starting CUDA in this process.

nvidia-smi is asked instead of torch: once torch has touched CUDA, hiding the
GPU from this process (CUDA_VISIBLE_DEVICES) no longer works.
"""
from __future__ import annotations

import subprocess
from typing import Optional, Tuple

# Below this, `device: auto` leaves the GPU to Ollama. The detectors' torch
# models take ~6 GB together (SelfCheckGPT NLI + BERTScore, UQLM NLI +
# BERTScore + cosine); a 7B generator or the judge takes 4-5 GB more.
AUTO_GPU_MIN_GIB = 16


def nvidia_gpu() -> Optional[Tuple[str, float]]:
    """(name, total GiB) of the largest NVIDIA GPU, or None if there is none."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        name, _, mib = line.rpartition(",")
        try:
            gpus.append((name.strip(), float(mib) / 1024))
        except ValueError:
            continue
    return max(gpus, key=lambda g: g[1]) if gpus else None
