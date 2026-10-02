"""Real plotting checks using numpy/pandas/matplotlib, without detector packages."""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReportChartTests(unittest.TestCase):
    def setUp(self):
        # Isolate plotting code from detector imports; plotting and pairing
        # themselves use the real installed numerical/graphics libraries.
        aggregate = types.ModuleType("reporting.aggregate")
        aggregate.score_columns = lambda frame: [c for c in frame.columns if c.endswith("_score")]
        parent = types.ModuleType("reporting")
        parent.aggregate = aggregate
        document = types.ModuleType("reporting.document")
        document.Report = object
        loguru = types.ModuleType("loguru")
        loguru.logger = MagicMock()
        self.modules = patch.dict(sys.modules, {
            "reporting": parent, "reporting.aggregate": aggregate,
            "reporting.document": document, "loguru": loguru,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            "report_build_under_test", Path(__file__).resolve().parents[1] / "reporting" / "build.py")
        self.build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.build)
        self.addCleanup(plt.close, "all")

    def test_repeated_runs_do_not_reweight_question_model_pairs(self):
        rows = []
        for run, question, baseline, method in (
                ("run_01", "q1", 0.8, 0.4), ("run_02", "q1", 0.8, 0.4),
                ("run_01", "q2", 0.2, 0.0)):
            for condition, value in (("baseline", baseline), ("greedy", method)):
                rows.append({"run": run, "sample_id": question, "model": "m", "condition": condition,
                             "judge_score": value})
        row = self.build.paired_risk_table(pd.DataFrame(rows), "judge").iloc[0]
        self.assertAlmostEqual(row["baseline_risk"], .5)
        self.assertAlmostEqual(row["method_risk"], .2)
        self.assertEqual((row["pairs"], row["questions"], row["observations"]), (2, 2, 3))

    def test_direct_comparison_has_upright_matched_bars_and_labels(self):
        paired = pd.DataFrame({"condition": ["greedy", "closed_book"],
                               "baseline_risk": [.5, .4], "method_risk": [.2, .6]})
        with patch.object(self.build, "_save", side_effect=lambda fig, path: fig):
            fig = self.build.absolute_risk_chart(paired, "minicheck", Path("unused.png"))
        ax = fig.axes[0]
        self.assertEqual([round(bar.get_height(), 2) for bar in ax.patches], [.4, .5, .6, .2])
        self.assertTrue(all(np.isclose(bar.get_width(), .36) for bar in ax.patches))
        self.assertIn("Lower bars", ax.get_ylabel())
        self.assertEqual(len(ax.texts), 4)
        self.assertEqual(ax.get_ylim()[0], 0)

    def test_small_smoke_effects_do_not_draw_fake_zero_width_intervals(self):
        effects = pd.DataFrame({"detector": ["minicheck"] * 2,
                                "condition": ["greedy", "closed_book"], "mean_change": [-.2, .1],
                                "ci_low": [np.nan, np.nan], "ci_high": [np.nan, np.nan]})
        with patch.object(self.build, "_save", side_effect=lambda fig, path: fig):
            fig = self.build.effects_chart(effects, ["minicheck"], {"minicheck": .8}, Path("unused.png"))
        ax = fig.axes[0]
        self.assertFalse(any(type(c).__name__ == "ErrorbarContainer" for c in ax.containers))
        self.assertIn("Below 0", ax.get_ylabel())

    def test_answer_scores_keep_native_scales_missing_counts_and_equal_pair_weights(self):
        frame = pd.DataFrame([
            {"run": run, "sample_id": question, "model": "m", "condition": condition, "ngram_score": value}
            for run, question, baseline, method in (("run_01", "q1", .8, .4),
                ("run_02", "q1", .8, .4), ("run_01", "q2", .2, 0.), ("run_01", "q3", 1.2, np.nan))
            for condition, value in (("baseline", baseline), ("greedy", method))])
        table = self.build.answer_score_table(frame).set_index("condition")
        self.assertAlmostEqual(table.loc["baseline", "mean_risk"], (.8 + .2 + 1.2) / 3)
        self.assertAlmostEqual(table.loc["greedy", "mean_risk"], .2)
        self.assertEqual(table.loc["greedy", "missing_scores"], 1)
        self.assertEqual(table.loc["greedy", "pairs"], 2)
        models = self.build.answer_score_table(frame, by_model=True)
        self.assertEqual(models["model"].unique().tolist(), ["m"])

    def test_before_after_chart_shows_positive_improvement_and_missing_measurements(self):
        scores = pd.DataFrame({"condition": ["baseline", "greedy", "closed_book"],
                               "detector": ["ngram"]*3, "mean_risk": [1.2, .8, np.nan]})
        matched = pd.DataFrame({"condition": ["greedy"], "detector": ["ngram"], "improvement": [.4]})
        with patch.object(self.build, "_save", side_effect=lambda fig, path: fig):
            fig = self.build.score_comparison_chart(scores, matched, "ngram", "Model A", Path("unused.png"))
        left, right = fig.axes
        self.assertGreater(left.get_ylim()[1], 1.2)  # no clipping to a false 0–1 scale
        self.assertTrue(any(text.get_text() == "missing" for text in left.texts))
        self.assertTrue(any(text.get_text() == "no pairs" for text in right.texts))
        self.assertTrue(any(text.get_text() == "+0.400" for text in right.texts))
        self.assertIn("Positive = improvement", right.get_ylabel())

    def test_ranking_and_run_comparison_generate_real_pngs(self):
        ranking = pd.DataFrame({"detector": ["MiniCheck", "UQLM judge"], "AUROC": [.8, .6],
                                "AUROC low": [.8, .6], "AUROC high": [.8, .6],
                                "compares with": ["the context"] * 2})
        by_run = pd.DataFrame({"run": ["run_01", "run_02"] * 2,
                              "detector": ["MiniCheck"] * 2 + ["UQLM judge"] * 2,
                              "roc_auc": [.8, .8, .6, .7]})
        with tempfile.TemporaryDirectory() as temp:
            for function, frame, name in ((self.build.ranking_chart, ranking, "ranking.png"),
                                          (self.build.consistency_chart, by_run, "runs.png")):
                path = Path(temp) / name
                self.assertEqual(function(frame, path), path)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_complete_html_report_has_new_charts_and_marks_legacy_runs(self):
        """Exercise report composition/aggregation with real tables and PNGs.

        Supplied summaries avoid sklearn, and DOCX export is excluded because
        this check is for HTML composition, not a claim of Word layout QA.
        """
        root = Path(__file__).resolve().parents[1]
        yaml = types.ModuleType("yaml")
        validator = types.ModuleType("benchmark.detector_validation")
        validator.DetectorValidator = object
        with patch.dict(sys.modules, {"yaml": yaml, "benchmark": types.ModuleType("benchmark"),
                                      "benchmark.detector_validation": validator}):
            spec = importlib.util.spec_from_file_location("aggregate_for_chart_test", root / "reporting/aggregate.py")
            aggregate = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = aggregate
            spec.loader.exec_module(aggregate)
            spec = importlib.util.spec_from_file_location("document_for_chart_test", root / "reporting/document.py")
            document = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = document
            spec.loader.exec_module(document)
            self.build.agg = aggregate
            self.build.Report = document.Report
            with tempfile.TemporaryDirectory() as temp:
                runs, deltas = [], []
                for index in (1, 2):
                    name = f"run_{index:02d}"
                    raw = pd.DataFrame([{"case_id": f"qa:{i}", "sample_id": f"q{i}", "dataset": "qa",
                                         "label": i - 1, "question": "When?", "answer": "answer",
                                         "minicheck_score": .1 if i == 1 else .9,
                                         "summac_score": .2 if i == 1 else .8} for i in (1, 2)])
                    summary = pd.DataFrame([{"run": name, "detector": detector, "model": aggregate.MODEL_INDEPENDENT,
                        "n_cases": 2, "n_failed": 0, "threshold": .5, **{m: 1.0 for m in aggregate.METRICS}}
                        for detector in ("minicheck", "summac")])
                    reduction = pd.DataFrame([{"run": name, "sample_id": f"q{i}", "dataset": "qa", "model": model,
                        "condition": cond, "answer": "answer", "minicheck_score": value, "summac_score": value / 2}
                        for model in ("fake-model", "second-model") for i in (1, 2)
                        for cond, value in (("baseline", .6), ("greedy", .2), ("closed_book", .8))])
                    runs.append(aggregate.RunData(Path(temp) / name, name, summary, raw, reduction,
                                                  None, None, None))
                    for model in ("fake-model", "second-model"):
                        for detector in ("minicheck", "summac"):
                            for i in (1, 2):
                                for condition, delta in (("greedy", -.4), ("closed_book", .2)):
                                    deltas.append({"run": name, "sample_id": f"q{i}", "dataset": "qa", "model": model,
                                        "condition": condition, "detector": detector,
                                        "delta": delta if detector == "minicheck" else delta / 2})
                with patch.object(aggregate, "discover", return_value=runs), \
                        patch.object(aggregate, "per_dataset_by_run", return_value=pd.DataFrame()), \
                        patch.object(aggregate, "reduction_deltas", return_value=pd.DataFrame(deltas)), \
                        patch.object(document.Report, "to_docx", return_value=Path(temp) / "excluded.docx"):
                    result = self.build.generate(Path(temp), Path(temp) / "combined")
                    html = re.sub("src='data:image/png;base64,[^']*'", "",
                                  result["report (html)"].read_text(encoding="utf-8"))
                    self.assertIn("Run independence is unverified", html)
                    self.assertIn("observations = completed pairs across runs", html)
                    self.assertIn("Before reduction · baseline scores", html)
                    self.assertIn("After reduction and improvement", html)
                    self.assertIn("positive green bar", html)
                    self.assertIn("Complete scores and comparisons", html)
                    self.assertIn("fake-model · MiniCheck", html)
                    self.assertIn("fake-model · SummaC", html)
                    self.assertIn("second-model · MiniCheck", html)
                    self.assertIn("second-model · SummaC", html)
                    self.assertEqual(len(list(result["charts"].glob("answer_scores_*.png"))), 6)
                    scores = pd.read_csv(result["tables"] / "answer_scores_by_model.csv")
                    self.assertEqual(len(scores), 12)  # 2 detectors × 2 models × 3 conditions
                    matched = pd.read_csv(result["tables"] / "matched_comparisons_by_model.csv")
                    self.assertEqual(len(matched), 8)  # 2 detectors × 2 models × 2 methods
                    self.assertNotIn("zero by design", html)
                    self.assertFalse(any("latency" in p.name for p in result["charts"].glob("*.png")))
                    self.assertTrue((result["charts"] / "baseline_vs_methods.png").is_file())
                    for run in runs:
                        run.manifest = {"run_protocol": {"detector_scores": "recomputed_each_run",
                            "generator_samples": "fresh_each_run", "preflight_samples": "discarded"}}
                    self.build.generate(Path(temp), Path(temp) / "combined")
                    html = re.sub("src='data:image/png;base64,[^']*'", "",
                                  result["report (html)"].read_text(encoding="utf-8"))
                    self.assertIn("all detector scores were recomputed", html)
                    self.assertNotIn("Run independence is unverified", html)


if __name__ == "__main__":
    unittest.main()
