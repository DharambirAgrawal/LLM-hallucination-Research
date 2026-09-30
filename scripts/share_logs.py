"""Collect what is needed to diagnose a run into one small text report and,
only with --upload, put it on paste.rs so it can be read elsewhere.

    python scripts/share_logs.py               # write debug_report.txt, upload nothing
    python scripts/share_logs.py --upload      # also upload; prints a short link
    python scripts/share_logs.py results/smoke-20260930-1200 --upload

Standard library only, so it works even when an environment is broken.

The report holds: the git commit, Python/OS, GPU (nvidia-smi), Ollama
(version, loaded and installed models), the config, every distinct error or
warning with a count, the last tracebacks, per-run CSV row and error counts,
and the end of each log. The home folder is replaced by ~. No API keys or
.env content are read. An uploaded report is readable by anyone who has the
link.
"""
from __future__ import annotations

import argparse
import platform
import re
import socket
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 400_000          # keep the report small enough to upload and read
PROBLEM = re.compile(r"out of memory|error|failed|traceback|warning|✗|⚠", re.IGNORECASE)
LOG_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d[ T][\d:.,]+\s*\|\s*")   # loguru time stamp


def sh(*cmd: str) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=ROOT)
        return (done.stdout + done.stderr).strip() or "(no output)"
    except FileNotFoundError:
        return f"({cmd[0]} not found)"
    except Exception as exc:
        return f"({' '.join(cmd)} failed: {exc})"


def latest_results() -> Path | None:
    folders = [p for p in (ROOT / "results").glob("*") if p.is_dir()]
    return max(folders, key=lambda p: p.stat().st_mtime) if folders else None


def read(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def error_summary(logs: list[Path]) -> list[str]:
    """Every distinct problem line, most frequent first, with its count."""
    counts: Counter[str] = Counter()
    for log in logs:
        for line in read(log):
            if PROBLEM.search(line):
                counts[LOG_PREFIX.sub("", line).strip()[:300]] += 1
    return [f"{n:>6} × {line}" for line, n in counts.most_common(60)]


def tracebacks(log: Path, keep: int = 4, length: int = 45) -> list[str]:
    lines = read(log)
    starts = [i for i, line in enumerate(lines) if line.startswith("Traceback")]
    out = []
    for i in starts[-keep:]:
        out += [f"--- {log.name} line {i + 1}"] + lines[i:i + length]
    return out


def csv_status(folder: Path) -> list[str]:
    """Rows and failed cases per CSV that has *_error columns."""
    import csv
    out = []
    for path in sorted(folder.glob("run_*/**/*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            out.append(f"{path.relative_to(folder)}: unreadable ({exc})")
            continue
        errors = {c: sum(1 for r in rows if (r.get(c) or "").strip())
                  for c in (rows[0].keys() if rows else []) if c.endswith("_error")}
        failed = ", ".join(f"{c}={n}" for c, n in errors.items() if n)
        out.append(f"{str(path.relative_to(folder)):<60} {len(rows):>5} rows"
                   + (f"   failed: {failed}" if failed else ""))
    return out


def build(folder: Path | None) -> str:
    parts: list[tuple[str, str]] = []
    add = lambda title, body: parts.append((title, body if isinstance(body, str) else "\n".join(body)))

    add("Run", [f"created   {datetime.now().isoformat(timespec='seconds')}",
                f"results   {folder or '(no results folder found)'}",
                f"commit    {sh('git', 'rev-parse', '--short', 'HEAD')}",
                f"changes   {sh('git', 'status', '--short') if (ROOT / '.git').exists() else '(not a git checkout)'}",
                f"python    {sys.version.split()[0]} ({sys.executable})",
                f"system    {platform.platform()}"])
    add("GPU (nvidia-smi)", sh("nvidia-smi"))
    add("Ollama", [f"version: {sh('ollama', '--version')}", "", "loaded (ollama ps):", sh("ollama", "ps"),
                   "", "installed (ollama list):", sh("ollama", "list")])

    logs = sorted(folder.glob("logs/*.log")) + sorted(folder.glob("*.log")) if folder else []
    logs += sorted((ROOT / "logs").glob("pip-*.log")) if (ROOT / "logs").is_dir() else []
    add("Errors and warnings (count × line)", error_summary(logs) or ["(none found)"])
    add("Last tracebacks", [line for log in logs for line in tracebacks(log)] or ["(none)"])
    if folder:
        add("Result files", csv_status(folder) or ["(no run CSVs yet)"])
        used = next(iter(sorted(folder.glob("run_*/**/config_used.yaml"))), None)
        add(f"Config ({used.relative_to(folder) if used else 'config.yaml'})",
            read(used) if used else read(ROOT / "config.yaml"))
    else:
        add("Config (config.yaml)", read(ROOT / "config.yaml"))
    for log in logs:
        add(f"End of {log.name}", read(log)[-80:])

    text = "\n\n".join(f"{'=' * 78}\n{title}\n{'=' * 78}\n{body}" for title, body in parts)
    home = str(Path.home())
    text = text.replace(home, "~")
    if len(text.encode()) > MAX_BYTES:
        text = text.encode()[:MAX_BYTES].decode(errors="ignore") + "\n\n[cut: report longer than the limit]"
    return text


def upload(text: str) -> str:
    """paste.rs over HTTPS; termbin.com as a fallback. Returns the link."""
    try:
        request = urllib.request.Request("https://paste.rs/", data=text.encode(), method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            link = response.read().decode().strip()
            if response.status == 206:
                link += "   (cut short by paste.rs size limit)"
            return link
    except Exception as exc:
        print(f"paste.rs failed ({exc}); trying termbin.com", file=sys.stderr)
    with socket.create_connection(("termbin.com", 9999), timeout=60) as conn:
        conn.sendall(text.encode())
        conn.shutdown(socket.SHUT_WR)
        return conn.recv(1024).decode().strip("\x00\n ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?", type=Path, help="results folder (default: the newest)")
    parser.add_argument("--upload", action="store_true",
                        help="upload the report and print a link (readable by anyone with the link)")
    args = parser.parse_args()

    folder = args.results or latest_results()
    text = build(folder)
    out = ROOT / "debug_report.txt"
    out.write_text(text, encoding="utf-8")
    print(f"Report written: {out} ({len(text.encode()) // 1024} KB)")
    if not args.upload:
        print("Nothing uploaded. To share it: python scripts/share_logs.py --upload")
        return
    link = upload(text)
    print()
    print("=" * 50)
    print(f"  Link to send:  {link}")
    print("=" * 50)


if __name__ == "__main__":
    main()
