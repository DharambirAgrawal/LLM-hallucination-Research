"""Timing and scheduling checks without Ollama, downloads or sleeping."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.runtime_estimate import build_estimate, combine_estimates, reduction_answers_per_question, save_estimate


class RuntimeEstimateTests(unittest.TestCase):
    def test_load_once_per_worker_not_once_per_question(self):
        warm = {"detector_validation": 40, "reduction_scoring": 60}
        probe = {"fixed_detector_startup_seconds": 28}
        standalone = build_estimate(warm, 2, calibration=probe)
        worker = build_estimate(warm, 2, calibration=probe, reload_fixed_detectors=True)
        self.assertEqual(standalone["remaining_compute_seconds"], 200)
        self.assertEqual(worker["remaining_compute_seconds"], 256)
        self.assertEqual(warm, {"detector_validation": 40, "reduction_scoring": 60})

    def test_elapsed_setup_is_separate_and_json_is_portable(self):
        estimate = build_estimate({"reduction": 60}, 3, elapsed_seconds=120)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime_estimate.json"
            save_estimate(path, estimate)
            loaded = json.loads(path.read_text())
        self.assertEqual(loaded["remaining_compute_seconds"], 180)
        self.assertEqual(loaded["elapsed_setup_preflight_seconds"], 120)
        self.assertEqual(loaded["planning_range_seconds"], [135, 360])
        self.assertTrue(any("not a guaranteed minimum" in s for s in loaded["notes"]))

    def test_invalid_timing_is_not_reported_as_a_duration(self):
        for value in (-1, float("nan"), float("inf"), 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_estimate({"validation": value}, 2)
        with self.assertRaises(ValueError):
            combine_estimates({}, 2)

    def test_reduction_row_count_includes_baseline_and_selected_models(self):
        config = {"models": [{"name": "a"}, {"name": "b"}], "selected_models": ["b"],
                  "reduction": {"methods": ["closed_book", "greedy"]}}
        self.assertEqual(reduction_answers_per_question(config), 3)
        config["reduction"]["methods"] = []
        self.assertEqual(reduction_answers_per_question(config), 1)
        config["reduction"] = {"method": "greedy"}
        self.assertEqual(reduction_answers_per_question(config), 2)
        config["reduction"] = {}
        self.assertEqual(reduction_answers_per_question(config), 6)

    def test_fixed_preflight_accounts_for_post_reduction_and_median_input(self):
        import utils
        console = types.ModuleType("utils.console")
        console.duration = str
        console.line = console.section = MagicMock()
        data = types.ModuleType("data.datasets")
        samples = [types.SimpleNamespace(sample_id=str(i), dataset="qa", question="q", context="x" * size)
                   for i, size in enumerate([1, 10, 30])]
        data.BenchmarkSample = object
        data.DatasetLoader = types.SimpleNamespace(detection_cases=lambda group:
            [{"sample_id": s.sample_id, "answer": answer} for s in group for answer in ("right", "wrong")])
        loguru = types.ModuleType("loguru")
        loguru.logger = MagicMock()
        stubs = {"utils.console": console, "data.datasets": data, "loguru": loguru,
                 "docx": types.ModuleType("docx"), "matplotlib": types.ModuleType("matplotlib"),
                 "sklearn": types.ModuleType("sklearn")}
        spec = importlib.util.spec_from_file_location("timed_preflight_test", Path(__file__).resolve().parents[1] / "benchmark/preflight.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs), patch.object(utils, "console", console, create=True):
            spec.loader.exec_module(module)
            def timed_check(self, label, action):
                action()
                return 30 if "loads + scores" in label else 2 if label.endswith("timing") else 0
            scorer = MagicMock(return_value={"fixed": .2})
            runner = types.SimpleNamespace(families={"fixed": types.SimpleNamespace(
                name="fixed", needs_generator=False, columns=["fixed"], score=scorer)})
            with tempfile.TemporaryDirectory() as tmp, patch.object(module.Preflight, "check", timed_check):
                estimate = module.run_preflight({"benchmark": {"output_dir": tmp}}, runner,
                                               {"qa": samples}, [], False, reduction_answers_per_question=30)
        # Three questions, six labeled cases, five generators × six conditions.
        self.assertEqual(estimate["detector_validation"], 12)
        self.assertEqual(estimate["reduction_scoring"], 180)
        self.assertEqual(runner.preflight_timing["fixed_detector_startup_seconds"], 28)
        self.assertEqual(runner.preflight_timing["probe"]["sample_id"], "1")
        self.assertEqual(scorer.call_count, 2)  # no extra calls just for the estimate

    def test_reducer_loading_is_not_multiplied_by_questions_or_models(self):
        import utils
        console = types.ModuleType("utils.console")
        console.duration = str
        console.line = console.section = MagicMock()
        data = types.ModuleType("data.datasets")
        data.BenchmarkSample = object
        samples = [types.SimpleNamespace(sample_id=str(i), dataset="qa", question="q", context="c") for i in range(3)]
        data.DatasetLoader = types.SimpleNamespace(detection_cases=lambda group:
            [{"answer": a} for s in group for a in ("right", "wrong")])
        loguru = types.ModuleType("loguru")
        loguru.logger = MagicMock()
        best = types.SimpleNamespace(_load=MagicMock())
        reducer = types.SimpleNamespace(_best=best,
            _produce=MagicMock(return_value={"baseline": {"error": None, "answer": "answer"}}),
            _score=MagicMock(return_value={"sample_score": .2}))
        reduction_module = types.ModuleType("benchmark.reduction_runner")
        reduction_module.ReductionRunner = MagicMock(return_value=reducer)
        stubs = {"utils.console": console, "data.datasets": data, "loguru": loguru,
                 "benchmark.reduction_runner": reduction_module,
                 "docx": types.ModuleType("docx"), "matplotlib": types.ModuleType("matplotlib"),
                 "sklearn": types.ModuleType("sklearn")}
        spec = importlib.util.spec_from_file_location("core_timed_preflight_test", Path(__file__).resolve().parents[1] / "benchmark/preflight.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs), patch.object(utils, "console", console, create=True):
            spec.loader.exec_module(module)
            def timed_check(self, label, action):
                action()
                if label == "reduction tools load":
                    return 50
                if label.endswith("reduction round"):
                    return 10
                if label.endswith("answers"):
                    return 3
                if label.endswith("samples"):
                    return 4
                return 2 if label.endswith("timing") else 22 if " · sample" in label else 0
            generators = [types.SimpleNamespace(name=n, generate=MagicMock(return_value="answer"), release=MagicMock()) for n in ("a", "b")]
            scorer = MagicMock(return_value={"sample": .2})
            runner = types.SimpleNamespace(families={"sample": types.SimpleNamespace(
                name="sample", needs_generator=True, columns=["sample"], score=scorer)},
                bank=types.SimpleNamespace(get=MagicMock(return_value=["answer", "other"])))
            with tempfile.TemporaryDirectory() as tmp, patch.object(module.Preflight, "check", timed_check):
                estimate = module.run_preflight({"benchmark": {"output_dir": tmp}}, runner, {"qa": samples}, generators, True)
        self.assertEqual(estimate["reduction"], 60)
        self.assertEqual(estimate["reducer_loading"], 50)
        self.assertEqual(estimate["generator_restarts"], 12)
        self.assertEqual(estimate["sampling_detector_loading"], 20)
        self.assertEqual(estimate["detector_validation"], 48)
        reduction_module.ReductionRunner.assert_called_once()
        best._load.assert_called_once()
        self.assertEqual(reducer._produce.call_count, 2)


if __name__ == "__main__":
    unittest.main()
