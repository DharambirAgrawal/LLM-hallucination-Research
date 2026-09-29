"""Terminal output for long runs: short readable lines on screen, full
detail (tracebacks, third-party warnings, retries) in the run's log file.

Everything that prints during a run goes through this module so the
terminal stays one consistent format: section headers, aligned key/value
lines, one progress bar per model/detector with an ETA, and one-line
warnings. Nothing is hidden: whatever is kept off the screen is written to
``run.log`` next to the results.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger
from tqdm import tqdm

WIDTH = 78
_IS_TTY = sys.stderr.isatty()


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Route loguru to a compact terminal sink plus a detailed log file.

    The terminal sink writes through ``tqdm.write`` so log lines never
    break an active progress bar.
    """
    # Quiet third-party chatter on screen; it still reaches the log file
    # through the warnings hook below.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.remove()
    logger.add(
        lambda message: tqdm.write(message, end="", file=sys.stderr),
        level=level.upper(),
        colorize=_IS_TTY,
        format=(
            "<dim>{time:HH:mm:ss}</dim> "
            "<level>{level.name:<7}</level> {message}\n"
        ),
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{line} | {message}",
            backtrace=False,
            diagnose=False,
            encoding="utf-8",
        )

    def _warning_to_log(message, category, filename, lineno, file=None, line=None):
        logger.debug(f"{category.__name__}: {message} ({filename}:{lineno})")

    warnings.showwarning = _warning_to_log


def header(title: str) -> None:
    line = f"══ {title} "
    tqdm.write("\n" + line + "═" * max(0, WIDTH - len(line)), file=sys.stderr)


def section(title: str) -> None:
    line = f"── {title} "
    tqdm.write("\n" + line + "─" * max(0, WIDTH - len(line)), file=sys.stderr)


def kv(key: str, value: object, width: int = 17) -> None:
    tqdm.write(f"  {key:<{width}} {value}", file=sys.stderr)


def line(text: str = "") -> None:
    tqdm.write(f"  {text}" if text else "", file=sys.stderr)


def progress(iterable: Iterable, desc: str, total: Optional[int] = None, unit: str = "case"):
    """One progress bar with elapsed time, ETA and rate.

    When output is redirected to a file (``nohup``/``tee``), the bar only
    refreshes every 30 s so the log does not fill with carriage returns.
    """
    return tqdm(
        iterable,
        desc=f"  {desc}",
        total=total,
        unit=unit,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.5 if _IS_TTY else 30,
        bar_format="{desc} {percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
