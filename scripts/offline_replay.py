#!/usr/bin/env python3
"""Two-question, two-run synthetic replay of the real pipeline and reports.

No Ollama, neural weights, installs, downloads, or environment preparation.
Prepared responses and synthetic detector backends exercise the actual local
adapters, reducers, metrics, CSV joins, completeness gates, and exporters.
This validates plumbing and reporting, NOT published detector accuracy.

    python scripts/offline_replay.py

--docx-python optionally selects an already installed python-docx runtime.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")  # replay metrics are serial; no process trackers needed

import numpy as np
import pandas as pd
import yaml

from utils.replay_fixture import QUESTIONS, ScriptedReplayModel, synthetic_backends


def no_network(*args, **kwargs):
    raise RuntimeError("Offline replay forbids network connections")


def export_bridge(python):
    def export(report, out):
        blocks = []
        for block in report.blocks:
            item = {"kind": block.kind, "text": block.text, "items": block.items,
                    "path": str(block.path.resolve()) if block.path else None}
            if block.frame is not None:
                item.update(records=block.frame.to_dict("records"), columns=list(block.frame.columns))
            blocks.append(item)
        source = out.with_name("report_blocks.json")
        source.write_text(json.dumps({"title": report.title, "subtitle": report.subtitle, "blocks": blocks}), encoding="utf-8")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # compiled packages must match the export runtime
        subprocess.run([python, str(ROOT / "scripts/offline_replay_docx.py"), str(source), str(out)],
                       check=True, env=env, cwd=ROOT)
        return out
    return export


def verify_report(folder, models, expected_runs):
    tables = folder / "tables"
    scores = pd.read_csv(tables / "answer_scores_by_model.csv")
    assert len(scores) == 13 * 5 * 6, "Incomplete detector/model/condition matrix"
    assert scores.mean_risk.notna().all() and np.isfinite(scores.mean_risk).all()
    assert (scores.missing_scores == 0).all(), "Replay contains missing measurements"
    assert set(scores.model) == set(models)
    assert (scores.observations == 2 * expected_runs).all()
    comparisons = pd.read_csv(tables / "matched_comparisons_by_model.csv")
    assert len(comparisons) == 13 * 5 * 5
    np.testing.assert_allclose(comparisons.improvement, comparisons.baseline_risk - comparisons.method_risk)
    np.testing.assert_allclose(comparisons.change, -comparisons.improvement)
    assert (comparisons.observations == 2 * expected_runs).all()
    assert (comparisons.questions == 2).all()
    pngs = list((folder / "charts").glob("answer_scores_*.png"))
    assert len(pngs) == 13 * 6, "Missing comparison charts"
    assert not any("latency" in p.name.lower() for p in (folder / "charts").glob("*.png"))
    html = (folder / "report.html").read_text(encoding="utf-8")
    assert "SYNTHETIC OFFLINE REPLAY" in html
    assert "All expected answer rows and finite detector scores passed" in html
    assert "FAILED · incomplete scores" not in html
    assert len(re.findall(r"<figure>", html)) >= len(pngs)
    with ZipFile(folder / "report.docx") as archive:
        assert archive.testzip() is None
        xml = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text = "\n".join(node.text or "" for node in xml.findall(".//w:t", ns))
        assert "SYNTHETIC OFFLINE REPLAY" in text
        assert "All expected answer rows and finite detector scores passed" in text
        from reporting.build import pretty
        for model in models:
            for detector in scores.detector.unique():
                assert f"{model} · {pretty(detector)}" in text
        pictures = [name for name in archive.namelist() if name.startswith("word/media/")]
        image_hashes = {hashlib.sha256(archive.read(name)).hexdigest() for name in pictures}
        for png in pngs:
            assert hashlib.sha256(png.read_bytes()).hexdigest() in image_hashes
        section = xml.find(".//w:sectPr/w:pgSz", ns)
        assert section.get(f"{{{ns['w']}}}orient") == "landscape"
    return {"passed": True, "detector_scores": 13, "model_views": 5,
            "answer_score_rows": len(scores), "matched_comparison_rows": len(comparisons),
            "comparison_charts": len(pngs), "docx_embedded_images": len(pictures),
            "html_figures": len(re.findall(r"<figure>", html))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--docx-python", help="Existing Python with pandas/python-docx; never installed by this command")
    args = parser.parse_args()
    output = (args.output or ROOT / "results" / f"offline-replay-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "SYNTHETIC_ONLY.txt").write_text("Synthetic offline software test. Prepared answers and toy detector outputs. NOT research measurements.\n")
    fixture = output / "questions.json"
    fixture.write_text(json.dumps(QUESTIONS, indent=2), encoding="utf-8")
    checkpoint = output / "fake-alignscore-checkpoint.txt"
    checkpoint.write_text("Synthetic backend sentinel. Not a neural checkpoint.\n")
    started = time.monotonic()
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    models = base["selected_models"]
    assert len(models) == 5, "Replay expects the smoke plan's five configured models"
    base["models"] = [{"name": name, "model": name, "provider": "replay"} for name in models]
    base["datasets"] = [{"name": "synthetic_replay_qa", "source": "json", "path": str(fixture), "max_samples": 2}]
    base["judge"] = {"model": "synthetic-replay-judge"}
    base["reduction"] = {"methods": ["closed_book", "greedy", "self_refine_adapted", "cove_adapted", "uqlm_best_response"], "max_iterations": 1}
    base["detectors"]["selfcheckgpt"].update(enabled=True, n_samples=2, methods=["ngram", "bertscore", "nli", "prompt"], device="cpu")
    base["detectors"]["uqlm"].update(enabled=True, scorers=["semantic_negentropy", "noncontradiction", "entailment", "cosine_sim", "bert_score"], device="cpu")
    base["detectors"]["alignscore"].update(enabled=True, checkpoint_path=str(checkpoint), device="cpu")
    groups = {"core": ["selfcheckgpt", "uqlm", "uqlm_judge"], "minicheck": ["minicheck"], "summac": ["summac"], "alignscore": ["alignscore"]}
    for name in ("uqlm_judge", "minicheck", "summac"):
        base["detectors"][name].update(enabled=True, device="cpu")
    base.setdefault("run", {}).update(runs=2, reduce=True)
    events, calls, checks = [], [], {}

    with synthetic_backends(), patch.object(socket.socket, "connect", no_network), \
            patch.object(socket.socket, "connect_ex", no_network), patch.object(socket, "getaddrinfo", no_network):
        from benchmark.runner import BenchmarkRunner
        from benchmark.preflight import run_preflight
        from data.datasets import DatasetLoader
        from reporting import generate
        from reporting.document import Report
        from utils import console
        from main import run_once
        if args.docx_python:
            Report.to_docx = export_bridge(args.docx_python)
        console.setup_logging("WARNING", output / "run.log")
        datasets = DatasetLoader(base, seed=42).load_all()
        assert sum(map(len, datasets.values())) == 2
        for run_index in (1, 2):
            run_name = f"run_{run_index:02d}"
            generators = [ScriptedReplayModel(name, index, run_index) for index, name in enumerate(models)]
            for group, families in groups.items():
                config = copy.deepcopy(base)
                config["benchmark"]["output_dir"] = str(output / run_name / group)
                for family, setting in config["detectors"].items():
                    setting["enabled"] = family in families
                runner = BenchmarkRunner(config)
                active = generators if group == "core" else []
                if run_index == 1:
                    run_preflight(config, runner, datasets, active, group == "core")
                    for g in active:
                        g.calls.clear()
                runner.bank.reset()
                run_once(Path(config["benchmark"]["output_dir"]), f"{run_name}/{group}", config,
                         runner, datasets, active, group == "core",
                         score_reduction_from=output if group != "core" else None,
                         run_name=run_name, make_report=False)
                manifest_path = Path(config["benchmark"]["output_dir"]) / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                assert manifest["score_completeness"]["passed"]
                manifest["execution_mode"] = "synthetic_offline_replay"
                manifest["run_protocol"]["detector_backend"] = "synthetic_fixture_not_neural_inference"
                manifest["huggingface_models"] = {}  # no cached models participated
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                events.append({"kind": "group_complete", "run": run_index, "group": group})
            for g in generators:
                calls.extend(g.calls)
                assert sum(c["kind"] == "sample_set" for c in g.calls) == 2
            generate(output / run_name, title=f"SYNTHETIC OFFLINE REPLAY · {run_name} · NOT research results")
            checks[run_name] = verify_report(output / run_name, models, 1)
            events.append({"kind": "report_complete", "run": run_index})
        generate(output, out_dir=output / "combined", title="SYNTHETIC OFFLINE REPLAY · combined · NOT research results")
        checks["combined"] = verify_report(output / "combined", models, 2)
        events.append({"kind": "combined_report_complete"})
        from reporting.aggregate import discover, reduction_table, score_columns
        reduction = reduction_table(discover(output))
        assert len(reduction) == 120 and len(score_columns(reduction)) == 13
        first = reduction[reduction.run == "run_01"].set_index(["sample_id", "model", "condition"])
        second = reduction[reduction.run == "run_02"].set_index(["sample_id", "model", "condition"])
        assert (first.answer != second.answer).any(), "Run fixtures were accidentally identical"
        assert any((first[c] != second[c]).any() for c in score_columns(reduction)), "Scores did not reflect distinct run inputs"
    (output / "prepared_generation_calls.json").write_text(json.dumps(calls, indent=2), encoding="utf-8")
    elapsed = round(time.monotonic() - started, 2)
    summary = {"mode": "synthetic_offline_replay", "passed": True, "seconds": elapsed,
               "questions_total": 2, "runs": 2, "models": models, "families": 6, "scores": 13,
               "reduction_answers": 120, "missing_scores": 0, "report_checks": checks, "events": events,
               "limitations": "Neural inference is mocked; report exports/metrics/CSV processing are real. Rendering must be checked separately."}
    (output / "replay_checks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nPASS · synthetic offline replay · {elapsed:.1f}s · {output}", flush=True)


if __name__ == "__main__":
    main()
