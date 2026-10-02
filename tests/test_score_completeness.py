"""Offline scoring repair and completeness gates; no downloads/model calls."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from utils.completeness import inspect_scores, require_scores

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    return result


class Tensor:
    """Only the tensor operations used by upstream's best-sentence formula."""
    def __init__(self, values):
        self.values = np.asarray(values)

    def reshape(self, *shape):
        return Tensor(self.values.reshape(*shape))

    def max(self, axis):
        return types.SimpleNamespace(values=types.SimpleNamespace(numpy=lambda: self.values.max(axis=axis)))


class ScoreCompletenessTests(unittest.TestCase):
    def test_scores_require_all_columns_finite_values_rows_and_no_errors(self):
        self.assertTrue(inspect_scores([{"x_score": 2.4, "y_score": -.2}], ["x_score", "y_score"], 1)["passed"])
        for value in (None, float("nan"), float("inf"), "bad"):
            result = inspect_scores([{"x_score": value}], ["x_score"], 1)
            self.assertFalse(result["passed"])
            self.assertEqual(result["missing_scores"], {"x_score": 1})
        self.assertFalse(inspect_scores([{"x_score": .1}], ["x_score", "y_score"], 1)["passed"])
        self.assertFalse(inspect_scores([], ["x_score"], 1)["passed"])
        self.assertFalse(inspect_scores([{"x_score": .1, "x_error": "partial: broken"}], ["x_score"], 1)["passed"])
        wrong_keys = inspect_scores([{"id": "q1", "x_score": .1}] * 2, ["x_score"], 2,
                                    ("id",), [("q1",), ("q2",)])
        self.assertFalse(wrong_keys["passed"])
        self.assertFalse(wrong_keys["expected_answer_keys_match"])

    def test_preflight_rejects_even_one_partial_scorer_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete detector scores"):
            require_scores({"ngram": 1.5, "__errors__": "bertscore: IndexError"}, ["ngram", "bertscore"])
        self.assertEqual(require_scores({"x": .2}, ["x"]), {"x": .2})

    def selfcheck(self):
        stubs = {
            "detectors": module("detectors"),
            "detectors.sampling": module("detectors.sampling", SampleBank=object),
            "selfcheckgpt": module("selfcheckgpt"),
            "selfcheckgpt.modeling_selfcheck": module("selfcheckgpt.modeling_selfcheck"),
        }
        context = patch.dict(sys.modules, stubs)
        context.start()
        self.addCleanup(context.stop)
        adapter = load("selfcheck_adapter_test", "detectors/selfcheckgpt_detector.py")
        return adapter.SelfCheckGPTDetector(method="bertscore", bank=object()), stubs["selfcheckgpt.modeling_selfcheck"]

    def test_bertscore_normal_inputs_keep_upstream_result_and_samples(self):
        detector, upstream = self.selfcheck()
        span = type("Span", (), {"text": "a normal long sentence", "__len__": lambda self: 5})()
        scorer = types.SimpleNamespace(nlp=lambda s: types.SimpleNamespace(sents=[span]),
                                       predict=MagicMock(return_value=np.array([.15, .25])))
        detector._scorers["bertscore"] = scorer
        sentences, samples = ["sentence one", "sentence two"], ["sample one", "sample two"]
        np.testing.assert_array_equal(detector._bertscore(sentences, samples), [.15, .25])
        scorer.predict.assert_called_once_with(sentences=sentences, sampled_passages=samples)

    def test_short_sample_uses_original_text_best_match_and_sample_weights(self):
        detector, upstream = self.selfcheck()

        class Span:
            def __init__(self, text, size):
                self.text, self.size = text, size
            def __len__(self):
                return self.size

        scorer = types.SimpleNamespace(
            nlp=lambda s: types.SimpleNamespace(sents=[Span("Yes.", 2), Span("1991.", 2)] if s == "short" else [Span(s, 5)]),
            default_model="en", rescale_with_baseline=True,
            predict=MagicMock(return_value=np.array([.15, .25])))
        upstream.bert_score = types.SimpleNamespace(score=MagicMock(return_value=(None, None, Tensor([.4, .7, .2, .1]))))
        detector._scorers["bertscore"] = scorer
        result = detector._bertscore(["answer one", "answer two"], ["normal one", "short", "normal two"])
        np.testing.assert_allclose(result, [(.15 * 2 + .3) / 3, (.25 * 2 + .8) / 3])
        scorer.predict.assert_called_once_with(sentences=["answer one", "answer two"], sampled_passages=["normal one", "normal two"])
        upstream.bert_score.score.assert_called_once_with(
            ["Yes.", "1991.", "Yes.", "1991."], ["answer one", "answer one", "answer two", "answer two"],
            lang="en", verbose=False, rescale_with_baseline=True)

    def test_short_samples_only_produce_scores_without_upstream_empty_call(self):
        detector, upstream = self.selfcheck()
        span = type("Span", (), {"text": "1991.", "__len__": lambda self: 2})()
        scorer = types.SimpleNamespace(nlp=lambda s: types.SimpleNamespace(sents=[span]),
                                       default_model="en", rescale_with_baseline=True, predict=MagicMock())
        detector._scorers["bertscore"] = scorer
        upstream.bert_score = types.SimpleNamespace(score=MagicMock(return_value=(None, None, Tensor([.8]))))
        np.testing.assert_allclose(detector._bertscore(["Released in 1991."], ["1991.", "1991."]), [.2])
        scorer.predict.assert_not_called()
        self.assertEqual(upstream.bert_score.score.call_count, 2)
        with self.assertRaisesRegex(ValueError, "requires sentences"):
            detector._bertscore(["answer"], [])

    def test_empty_or_incomplete_generated_sample_sets_are_rejected(self):
        with patch.dict(sys.modules, {"models": module("models"), "models.prompts": module("models.prompts", grounded_prompt=lambda q, c: q + c)}):
            sampling = load("sample_bank_test", "detectors/sampling.py")
            for samples in (["ok", ""], ["ok"]):
                generator = types.SimpleNamespace(name="m", sample_n=lambda *a, **k: samples)
                with self.assertRaisesRegex(RuntimeError, "2 nonempty samples"):
                    sampling.SampleBank(n_samples=2).get(generator, "q", "c")

    def test_real_run_once_writes_failure_gate_and_diagnostic_report(self):
        import utils
        console = module("utils.console")
        for name in ("section", "line"):
            setattr(console, name, MagicMock())
        cases = [{"case_id": "q:faithful"}, {"case_id": "q:hallu"}]
        detector = types.SimpleNamespace(columns=["x"])
        runner = types.SimpleNamespace(families={"x": detector}, generator_columns=lambda: [],
                                       validate=lambda *a, **k: pd.DataFrame([{"case_id": "q:faithful", "x_score": .1}, {"case_id": "q:hallu", "x_score": np.nan}]))
        report = MagicMock(return_value={"report (docx)": Path("report.docx")})
        stubs = {"yaml": module("yaml"), "loguru": module("loguru", logger=MagicMock()),
                 "utils.console": console, "benchmark": module("benchmark"),
                 "data": module("data"), "data.datasets": module("data.datasets", DatasetLoader=types.SimpleNamespace(detection_cases=lambda group: cases)),
                 "benchmark.reduction_runner": module("benchmark.reduction_runner", ReductionRunner=object),
                 "reporting": module("reporting", generate=report),
                 "utils.run_manifest": module("utils.run_manifest", write_run_files=MagicMock())}
        with patch.dict(sys.modules, stubs), patch.object(utils, "console", console, create=True):
            main = load("main_gate_test", "main.py")
            self.assertFalse(hasattr(main, "DatasetLoader"))  # run_once must import its own dependency
            main.print_table = MagicMock()
            main.summarise = lambda *a: pd.DataFrame([{"n_cases": 1, "model": "m"}])
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                with self.assertRaisesRegex(SystemExit, "this run FAILED"):
                    main.run_once(folder, "run_01", {"benchmark": {}}, runner, {"qa": ["q"]}, [], False)
                result = json.loads((folder / "score_completeness.json").read_text())
                self.assertFalse(result["passed"])
                self.assertEqual(result["detector_validation"]["missing_scores"], {"x_score": 1})
                report.assert_called_once()
                runner.validate = lambda *a, **k: pd.DataFrame([{"case_id": "q:faithful", "x_score": .1}, {"case_id": "q:hallu", "x_score": .9}])
                main.run_once(folder, "run_01", {"benchmark": {}}, runner, {"qa": ["q"]}, [], False)
                self.assertTrue(json.loads((folder / "score_completeness.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
