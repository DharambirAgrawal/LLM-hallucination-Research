"""Run isolation and report timing checks; stdlib only, no installs or model calls."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.runtime_estimate import build_estimate

import utils

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("core", "minicheck", "summac", "alignscore")
METHODS = ["closed_book", "greedy", "self_refine_adapted", "cove_adapted", "uqlm_best_response"]


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    return result


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class Frame:
    """Capture validate() outputs without importing pandas."""
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)

    def to_csv(self, path, index=False):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


class Progress(list):
    def close(self):
        pass

    def set_postfix_str(self, text):
        pass


@contextmanager
def dependencies(config):
    console = module("utils.console", WIDTH=78, duration=lambda seconds: "0s",
                     progress=lambda rows, **kwargs: Progress(rows))
    for name in ("header", "section", "line", "kv", "setup_logging"):
        setattr(console, name, MagicMock())

    class Loader:
        def __init__(self, config, seed):
            pass

        def load_all(self):
            return {"qa": [
                {"case_id": "qa:1", "question": "q1", "context": "c1", "answer": "a1", "label": 0},
                {"case_id": "qa:2", "question": "q2", "context": "c2", "answer": "a2", "label": 1},
            ]}

        @staticmethod
        def detection_cases(samples):
            return copy.deepcopy(samples)

    stubs = {
        "utils.console": console,
        "pandas": module("pandas", DataFrame=Frame),
        "yaml": module("yaml", safe_load=lambda text: copy.deepcopy(config)),
        "loguru": module("loguru", logger=MagicMock()),
        "data": module("data"),
        "data.datasets": module("data.datasets", DatasetLoader=Loader, BenchmarkSample=object),
        "detectors": module("detectors", **{name: object for name in (
            "AlignScoreDetector", "MiniCheckDetector", "SelfCheckGPTDetector", "SummaCDetector")}),
        "models": module("models"),
        "models.base_model": module("models.base_model", BaseModel=object),
        "models.prompts": module("models.prompts", grounded_prompt=lambda q, c: q + c),
        "utils.resources": module("utils.resources", prepare=MagicMock()),
        "utils.gpu": module("utils.gpu", nvidia_gpu=lambda: None),
    }
    with patch.dict(sys.modules, stubs), patch.object(utils, "console", console, create=True):
        sampling = load_script("detectors.sampling", "detectors/sampling.py")
        yield stubs, sampling.SampleBank


def config():
    return {
        "benchmark": {"seed": 42}, "detectors": {}, "judge": {"model": "fake-judge"},
        "datasets": [{"name": "qa", "source": "synthetic", "enabled": True},
                     {"name": "other", "source": "synthetic", "enabled": True}],
        "run": {"runs": 2, "reduce": True}, "reduction": {"methods": METHODS},
        "selected_models": ["fake-generator"], "models": [{"name": "fake-generator"}],
    }


class FullRunSequenceTests(unittest.TestCase):
    def exercise(self, root, *, failure=None, report_failure=None, skip_report=False,
                 preflight_only=False, write_timings=False):
        events, commands, titles = [], {}, {}

        def child(cmd, **kwargs):
            name = cmd[cmd.index("--part") + 1]
            if "--preflight" in cmd:
                events.append(("preflight", name))
                if write_timings:
                    folder = root / "logs"
                    folder.mkdir(exist_ok=True)
                    (folder / f"preflight-timing-{name}.json").write_text(json.dumps(
                        build_estimate({"detector_validation": 10, "reduction": 20 if name == "core" else 0},
                                       2, group=name, calibration={"questions": 2})))
                    if name != "core":
                        flag = "--estimate-reduction-answers-per-question"
                        self.assertEqual(cmd[cmd.index(flag) + 1], "6")
                return types.SimpleNamespace(returncode=0)
            index = int(cmd[cmd.index("--run-index") + 1])
            commands[index, name] = cmd
            self.assertIn("--skip-preflight", cmd)
            self.assertIn("--smoke-2q", cmd)
            self.assertEqual(cmd[cmd.index("--runs") + 1], "2")
            if index == 2 and not skip_report and report_failure != "run_01":
                for suffix in ("docx", "html"):
                    self.assertTrue((root / "run_01" / f"report.{suffix}").is_file())
            events.append(("worker", index, name))
            folder = root / f"run_{index:02d}" / name
            folder.mkdir(parents=True)
            return types.SimpleNamespace(returncode=1 if failure == (index, name) else 0)

        def report(source, out_dir=None, title=None):
            folder = out_dir or source
            titles[folder.name] = title
            events.append(("report", folder.name))
            if folder.name == report_failure:
                raise RuntimeError("fake report failure")
            if folder.name != "combined":
                self.assertTrue(all((folder / name).is_dir() for name in GROUPS))
            folder.mkdir(exist_ok=True)
            paths = {f"report ({suffix})": folder / f"report.{suffix}" for suffix in ("docx", "html")}
            for path in paths.values():
                path.write_text("test report placeholder", encoding="utf-8")
            return paths

        with dependencies(config()):
            controller = load_script("full_run_under_test", "scripts/run_full.py")
            args = ["run_full.py", "--smoke-2q", "--output", str(root)]
            if skip_report:
                args.append("--skip-report")
            if preflight_only:
                args.append("--preflight")
            with patch.object(controller, "ensure_python", return_value=Path(sys.executable)), \
                    patch.object(controller.subprocess, "run", side_effect=child), \
                    patch.dict(sys.modules, {"reporting": module("reporting", generate=report)}), \
                    patch.object(sys, "argv", args):
                if failure or report_failure:
                    with self.assertRaises(SystemExit) as caught:
                        controller.main()
                    self.assertEqual(caught.exception.code, 1)
                else:
                    controller.main()
        return events, commands, titles

    def test_all_groups_then_report_before_next_run_and_combined_last(self):
        with tempfile.TemporaryDirectory() as temp:
            events, _, _ = self.exercise(Path(temp))
        expected = [("preflight", name) for name in GROUPS]
        for index in (1, 2):
            expected += [("worker", index, name) for name in GROUPS]
            expected.append(("report", f"run_{index:02d}"))
        expected.append(("report", "combined"))
        self.assertEqual(events, expected)

    def test_failed_core_only_disables_reduction_scoring_in_its_own_run(self):
        with tempfile.TemporaryDirectory() as temp:
            _, commands, titles = self.exercise(Path(temp), failure=(1, "core"))
        for name in GROUPS[1:]:
            self.assertNotIn("--score-reduction-from", commands[1, name])
            self.assertIn("--score-reduction-from", commands[2, name])
        self.assertIn("incomplete", titles["run_01"])
        self.assertNotIn("incomplete", titles["run_02"])
        self.assertIn("incomplete", titles["combined"])

    def test_report_failure_is_nonzero_but_other_run_reports_are_attempted(self):
        with tempfile.TemporaryDirectory() as temp:
            events, _, _ = self.exercise(Path(temp), report_failure="run_01")
        self.assertEqual([event for event in events if event[0] == "report"],
                         [("report", "run_01"), ("report", "run_02"), ("report", "combined")])

    def test_skip_report_skips_all_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            events, _, _ = self.exercise(Path(temp), skip_report=True)
        self.assertFalse(any(event[0] == "report" for event in events))

    def test_total_estimate_adds_groups_and_all_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events, _, _ = self.exercise(root, write_timings=True)
            estimate = json.loads((root / "runtime_estimate.json").read_text())
        self.assertEqual(estimate["seconds_per_run"], 60)
        self.assertEqual(estimate["remaining_compute_seconds"], 120)
        self.assertEqual(estimate["planning_range_seconds"], [90, 240])
        self.assertEqual(events[-1], ("report", "combined"))

    def test_full_preflight_only_estimates_without_any_measured_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events, commands, _ = self.exercise(root, write_timings=True, preflight_only=True)
            self.assertTrue((root / "runtime_estimate.json").is_file())
        self.assertFalse(commands)
        self.assertEqual(events, [("preflight", name) for name in GROUPS])

    def test_missing_worker_timings_do_not_invent_total(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.exercise(root, preflight_only=True)
            self.assertFalse((root / "runtime_estimate.json").exists())


class IndependentScoreTests(unittest.TestCase):
    def test_same_runner_recomputes_all_fixed_detector_scores_on_second_run(self):
        with tempfile.TemporaryDirectory() as temp, dependencies(config()) as (stubs, _):
            harness = load_script("benchmark_runner_under_test", "benchmark/runner.py")
            runner = harness.BenchmarkRunner.__new__(harness.BenchmarkRunner)
            runner.output_dir = Path(temp)
            runner.generator = None
            scorers = {name: MagicMock(side_effect=[{name: .1}, {name: .2},
                                                    {name: .3}, {name: .4}])
                       for name in ("uqlm_judge", "minicheck", "summac", "alignscore")}
            runner.families = {name: harness.Family(name, [name], {name: .5}, False, score)
                               for name, score in scorers.items()}
            datasets = stubs["data.datasets"].DatasetLoader({}, 42).load_all()
            first = runner.validate(datasets)
            second = runner.validate(datasets)
            for name, score in scorers.items():
                self.assertEqual(score.call_count, 4)
                self.assertEqual(first.rows[0][f"{name}_score"], .1)
                self.assertEqual(second.rows[0][f"{name}_score"], .3)

    def exercise_main(self, root, *, worker_index=None, preflight_only=False,
                      mode="--smoke-2q", timed=False):
        fixture = config()
        if mode != "--smoke-2q":
            fixture["datasets"] = fixture["datasets"][:1]
            fixture["run"].update(detectors=["selfcheckgpt", "uqlm"], runs=3)
        with dependencies(fixture) as (_, Bank):
            generator = types.SimpleNamespace(name="fake-generator", config={})
            generator.sample_n = MagicMock(side_effect=lambda *args, **kwargs:
                                           [f"generation-{generator.sample_n.call_count}"] * kwargs["n"])
            factory = types.SimpleNamespace(ensure_ollama_models=MagicMock(),
                build_all=lambda cfg: [generator], active_configs=lambda cfg: [{"name": generator.name}])
            runner = types.SimpleNamespace(bank=Bank(n_samples=2), attach=MagicMock())
            runner_factory = MagicMock(return_value=runner)
            observed = []

            def preflight(*args):
                runner.bank.get(generator, "q", "c")
                return {"detector_validation": 60} if timed else {}

            def run_once(folder, *args, **kwargs):
                folder.mkdir(parents=True)
                observed.append((folder, runner.bank.get(generator, "q", "c")))

            with patch.dict(sys.modules, {
                "models": module("models", ModelFactory=factory),
                "benchmark": module("benchmark", BenchmarkRunner=runner_factory),
                "benchmark.preflight": module("benchmark.preflight", run_preflight=MagicMock(side_effect=preflight)),
                "reporting": module("reporting", generate=lambda *args, **kwargs: {"report (docx)": "fake"}),
            }):
                worker = load_script("main_under_test", "main.py")
                argv = ["main.py", "--device", "cpu", "--output", str(root)]
                if mode:
                    argv.append(mode)
                if preflight_only:
                    argv.append("--preflight")
                if worker_index is not None:
                    argv += ["--part", "core", "--run-index", str(worker_index), "--skip-preflight",
                             "--detectors", "selfcheckgpt", "uqlm", "uqlm_judge", "--runs", "2"]
                with patch.object(sys, "argv", argv), \
                        patch.object(worker, "run_once", side_effect=run_once), \
                        patch.object(worker, "load_config", side_effect=lambda path: copy.deepcopy(fixture)):
                    worker.main()
            return observed

    def test_smoke_2q_smoke_and_full_preflight_save_correct_number_of_runs(self):
        for mode, expected_runs in (("--smoke-2q", 2), ("--smoke", 2), (None, 3)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                observed = self.exercise_main(root, mode=mode, timed=True, preflight_only=True)
                estimate = json.loads((root / "runtime_estimate.json").read_text())
                self.assertFalse(observed)
                self.assertEqual(estimate["remaining_compute_seconds"], expected_runs * 60)
                self.assertEqual(estimate["runs"], expected_runs)

    def test_standalone_discards_preflight_samples_and_resamples_every_run(self):
        with tempfile.TemporaryDirectory() as temp:
            observed = self.exercise_main(Path(temp))
        self.assertEqual([folder.name for folder, _ in observed], ["run_01", "run_02"])
        self.assertEqual([samples for _, samples in observed], [["generation-2"] * 2, ["generation-3"] * 2])

    def test_worker_runs_only_requested_index_and_protects_existing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_01" / "core").mkdir(parents=True)
            observed = self.exercise_main(root, worker_index=2)
            self.assertEqual([folder for folder, _ in observed], [root / "run_02" / "core"])
            with self.assertRaisesRegex(SystemExit, "already holds results"):
                self.exercise_main(root, worker_index=2)
            with self.assertRaisesRegex(SystemExit, "index between 1 and --runs"):
                self.exercise_main(root, worker_index=3)


if __name__ == "__main__":
    unittest.main()
