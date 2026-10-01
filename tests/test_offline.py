"""Offline tests: no LLM calls and no detector/model downloads."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from benchmark.detector_validation import DetectorValidator
from benchmark.reduction_runner import ReductionRunner
from benchmark.runner import BenchmarkRunner
from data.datasets import DatasetLoader
from detectors.alignscore_detector import AlignScoreDetector
from detectors.minicheck_detector import MiniCheckDetector
from detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from detectors.summac_detector import SummaCDetector
from models.replay_model import ReplayModel
from reducers.self_refine import SelfRefineReducer


ROOT = Path(__file__).resolve().parents[1]


def fake_upstream_modules() -> dict[str, types.ModuleType]:
    selfcheck_package = types.ModuleType("selfcheckgpt")
    selfcheck_module = types.ModuleType("selfcheckgpt.modeling_selfcheck")

    class FakeSelfCheck:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            if "passage" in kwargs:
                value = 4.0 if "1985" in kwargs["passage"] else 1.0
                return {"sent_level": {"avg_neg_logprob": [value]}}
            return [0.8 if "1985" in sentence else 0.1 for sentence in kwargs["sentences"]]

    selfcheck_module.SelfCheckNgram = FakeSelfCheck
    selfcheck_module.SelfCheckNLI = FakeSelfCheck
    selfcheck_module.SelfCheckBERTScore = FakeSelfCheck

    minicheck_package = types.ModuleType("minicheck")
    minicheck_module = types.ModuleType("minicheck.minicheck")

    class FakeMiniCheck:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, docs, claims):
            probabilities = [0.1 if "1985" in claim else 0.9 for claim in claims]
            return [int(value >= 0.5) for value in probabilities], probabilities, None, None

    minicheck_module.MiniCheck = FakeMiniCheck

    summac_package = types.ModuleType("summac")
    summac_module = types.ModuleType("summac.model_summac")

    class FakeSummaC:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, documents, claims):
            return {"scores": [0.1 if "1985" in claims[0] else 0.9]}

    summac_module.SummaCConv = FakeSummaC
    summac_module.SummaCZS = FakeSummaC

    alignscore_module = types.ModuleType("alignscore")

    class FakeAlignScore:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, contexts, claims):
            return [0.1 if "1985" in claims[0] else 0.9]

    alignscore_module.AlignScore = FakeAlignScore

    # No spaCy model offline: the SelfCheckGPT adapter falls back to its
    # regex splitter. (Loading real spaCy inside patch.dict would unload
    # its C extensions afterwards and crash the interpreter at exit.)
    spacy_module = types.ModuleType("spacy")

    def missing_model(name):
        raise OSError(f"offline test: {name} not loaded")

    spacy_module.load = missing_model

    # UQLM: confidences that are high for 1991 answers, low for 1985 ones.
    # Records its constructor arguments so tests can check use_best etc.
    uqlm_module = types.ModuleType("uqlm")
    uqlm_judges = types.ModuleType("uqlm.judges")

    class FakeResult:
        def __init__(self, data):
            self.data = data

    class FakeBlackBoxUQ:
        init_kwargs = {}

        def __init__(self, **kwargs):
            FakeBlackBoxUQ.init_kwargs = kwargs
            self.scorers = kwargs["scorers"]

        def score(self, responses, sampled_responses, show_progress_bars=True):
            conf = 0.1 if "1985" in responses[0] else 0.9
            return FakeResult({name: [conf] for name in self.scorers})

    class FakeSemanticEntropy:
        init_kwargs = {}

        def __init__(self, **kwargs):
            FakeSemanticEntropy.init_kwargs = kwargs

        def score(self, responses, sampled_responses, show_progress_bars=True, prompts=None):
            candidates = [responses[0], *sampled_responses[0]]
            best = next((c for c in candidates if "1985" not in c), candidates[0])
            return FakeResult({"responses": [best]})

    class FakeJudge:
        def __init__(self, llm, scoring_template="true_false_uncertain"):
            self.llm = llm

        async def judge_responses(self, prompts, responses):
            return {"scores": [0.0 if "1985" in r else 1.0 for r in responses]}

    uqlm_module.BlackBoxUQ = FakeBlackBoxUQ
    uqlm_module.SemanticEntropy = FakeSemanticEntropy
    uqlm_judges.LLMJudge = FakeJudge
    langchain_ollama = types.ModuleType("langchain_ollama")
    langchain_ollama.ChatOllama = lambda **kwargs: kwargs

    return {
        "uqlm": uqlm_module,
        "uqlm.judges": uqlm_judges,
        "langchain_ollama": langchain_ollama,
        "spacy": spacy_module,
        "selfcheckgpt": selfcheck_package,
        "selfcheckgpt.modeling_selfcheck": selfcheck_module,
        "minicheck": minicheck_package,
        "minicheck.minicheck": minicheck_module,
        "summac": summac_package,
        "summac.model_summac": summac_module,
        "alignscore": alignscore_module,
    }


class DataAndMetricsTests(unittest.TestCase):
    def test_synthetic_fixture_generates_balanced_fixed_cases(self):
        config = {"datasets": [{"name": "synthetic", "source": "synthetic", "max_samples": 8}]}
        datasets = DatasetLoader(config, seed=42).load_all()
        cases = DatasetLoader.detection_cases(datasets["synthetic"])
        self.assertEqual(len(cases), 16)
        self.assertEqual(sum(case["label"] for case in cases), 8)

    def test_ragtruth_loader_keeps_every_labeled_answer_of_the_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sources = Path(temp_dir) / "source_info.jsonl"
            responses = Path(temp_dir) / "response.jsonl"
            sources.write_text(json.dumps({
                "source_id": "7", "task_type": "QA", "source": "MARCO",
                "source_info": {"question": "When?", "passages": "passage 1: in 1991"},
                "prompt": "Briefly answer the following question:\nWhen?\n..."}) + "\n")
            rows = [
                {"id": "1", "source_id": "7", "model": "m1", "split": "test", "quality": "good",
                 "labels": [], "response": "In 1991."},
                {"id": "2", "source_id": "7", "model": "m2", "split": "test", "quality": "good",
                 "labels": [{"text": "1985"}], "response": "In 1985."},
                {"id": "3", "source_id": "7", "model": "m3", "split": "train", "quality": "good",
                 "labels": [], "response": "train split, skipped"},
                {"id": "4", "source_id": "7", "model": "m4", "split": "test", "quality": "truncated",
                 "labels": [], "response": "truncated, skipped"},
            ]
            responses.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            config = {"datasets": [{"name": "rt", "source": "ragtruth", "task_type": "QA",
                                    "responses_path": str(responses), "sources_path": str(sources),
                                    "max_samples": 5}]}
            samples = DatasetLoader(config, seed=42).load_all()["rt"]
            self.assertEqual(len(samples), 1)
            self.assertEqual((samples[0].question, samples[0].context), ("When?", "passage 1: in 1991"))
            cases = DatasetLoader.detection_cases(samples)
            self.assertEqual([(c["answer"], c["label"]) for c in cases], [("In 1991.", 0), ("In 1985.", 1)])

    def test_halubench_loader_maps_pass_fail_to_labels(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hb.parquet"
            pd.DataFrame([
                {"id": "a1", "passage": "P", "question": "Q", "answer": "right", "label": "PASS", "source_ds": "DROP"},
                {"id": "a2", "passage": "P", "question": "Q", "answer": "wrong", "label": "FAIL", "source_ds": "DROP"},
                {"id": "a3", "passage": "P", "question": "Q", "answer": "['x', 'y']", "label": "FAIL", "source_ds": "DROP"},
                {"id": "a4", "passage": "P2", "question": "Q2", "answer": "ok", "label": "PASS", "source_ds": "DROP"},
                {"id": "b1", "passage": "X", "question": "Y", "answer": "other", "label": "PASS", "source_ds": "covidQA"},
            ]).to_parquet(path)
            config = {"datasets": [{"name": "hb", "source": "halubench", "source_ds": "DROP",
                                    "path": str(path), "max_samples": 5}]}
            samples = DatasetLoader(config, seed=42).load_all()["hb"]
            self.assertEqual(len({s.sample_id for s in samples}), 2)   # ids unique per question
            cases = DatasetLoader.detection_cases(samples)
            # the list-literal answer (a format artifact) is excluded
            self.assertEqual(sorted((c["answer"], c["label"]) for c in cases), [("ok", 0), ("right", 0), ("wrong", 1)])

    def test_metrics_have_expected_direction(self):
        result = DetectorValidator.evaluate(
            labels=[0, 0, 1, 1], scores=[0.1, 0.2, 0.8, 0.9], threshold=0.5
        )
        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["f1"], 1.0)


class AdapterContractTests(unittest.TestCase):
    context = "Python was released in 1991 by Guido van Rossum."
    factual = "Python was released in 1991."
    hallucinated = "Python was released in 1985."

    def setUp(self):
        self.modules = patch.dict(sys.modules, fake_upstream_modules())
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_all_official_adapter_contracts(self):
        replay = ReplayModel([self.factual, "Guido van Rossum released Python in 1991."])
        selfcheck = SelfCheckGPTDetector(replay, method="ngram", n_samples=2, threshold=3.0)
        self.assertLess(
            selfcheck.detect("When?", self.context, self.factual).score,
            selfcheck.detect("When?", self.context, self.hallucinated).score,
        )

        minicheck = MiniCheckDetector()
        self.assertLess(
            minicheck.detect(self.context, self.factual).score,
            minicheck.detect(self.context, self.hallucinated).score,
        )

        summac = SummaCDetector()
        self.assertLess(
            summac.detect(self.context, self.factual).score,
            summac.detect(self.context, self.hallucinated).score,
        )

        with tempfile.NamedTemporaryFile() as checkpoint:
            alignscore = AlignScoreDetector(checkpoint_path=checkpoint.name)
            self.assertLess(
                alignscore.detect(self.context, self.factual).score,
                alignscore.detect(self.context, self.hallucinated).score,
            )

    def test_runner_writes_complete_fixed_case_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "alignscore.ckpt"
            checkpoint.write_bytes(b"offline fake checkpoint")
            config = {
                "datasets": [{
                    "name": "synthetic",
                    "enabled": True,
                    "source": "synthetic",
                    "max_samples": 2,
                }],
                "detectors": {
                    "selfcheckgpt": {
                        "enabled": True,
                        "method": "ngram",
                        "n_samples": 2,
                        "threshold": 3.0,
                    },
                    "minicheck": {"enabled": True},
                    "summac": {"enabled": True},
                    "alignscore": {
                        "enabled": True,
                        "checkpoint_path": str(checkpoint),
                    },
                },
                "benchmark": {"output_dir": temp_dir, "seed": 42},
            }
            datasets = DatasetLoader(config, seed=42).load_all()
            generator = ReplayModel([self.factual, "Python first appeared in 1991."])
            frame = BenchmarkRunner(config, generator=generator).validate(datasets)
            self.assertEqual(len(frame), 4)
            for column in ("selfcheckgpt_ngram", "minicheck", "summac", "alignscore"):
                self.assertTrue(frame[f"{column}_score"].notna().all(), column)
            for family in ("selfcheckgpt", "minicheck", "summac", "alignscore"):
                self.assertTrue(frame[f"{family}_error"].isna().all(), family)
            self.assertTrue((Path(temp_dir) / "detector_validation_raw.csv").exists())

    def test_runner_scores_every_selected_generator(self):
        """Every model passed via `generators=` is scored, not only the first."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "datasets": [{
                    "name": "synthetic",
                    "enabled": True,
                    "source": "synthetic",
                    "max_samples": 1,
                }],
                "detectors": {
                    "selfcheckgpt": {
                        "enabled": True,
                        "method": "ngram",
                        "n_samples": 1,
                        "threshold": 3.0,
                    },
                },
                "benchmark": {"output_dir": temp_dir, "seed": 42},
            }
            datasets = DatasetLoader(config, seed=42).load_all()
            model_a = ReplayModel([self.factual], name="model-a")
            model_b = ReplayModel(["unrelated text"], name="model-b")
            frame = BenchmarkRunner(config, generator=model_a).validate(
                datasets, generators=[model_a, model_b]
            )
            self.assertEqual(sorted(frame["model"].unique()), ["model-a", "model-b"])
            self.assertEqual(len(frame), 4)  # 2 cases (factual/hallucinated) x 2 models
            self.assertTrue(frame["selfcheckgpt_ngram_score"].notna().all())

    def test_gpu_out_of_memory_moves_the_detector_to_cpu(self):
        """An OOM that survives one cache-free retry reloads the detector on
        the CPU; the case is still scored and the change is in the config."""
        config = {"detectors": {"summac": {"enabled": True, "device": "cuda"}},
                  "benchmark": {"output_dir": tempfile.mkdtemp()}}
        runner = BenchmarkRunner(config)
        calls = []

        def out_of_memory(context, answer):
            calls.append(1)
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        runner.detectors["summac"].detect = out_of_memory
        case = {"question": "q", "context": self.context, "answer": self.factual}
        scores = runner.families["summac"].score(case, None)
        self.assertEqual(len(calls), 2)                 # first try + one retry on the GPU
        self.assertIsNotNone(scores["summac"])          # then scored on the CPU
        self.assertEqual(runner.config["detectors"]["summac"]["device"], "cpu")
        self.assertIsNot(runner.detectors["summac"].detect, out_of_memory)

    def test_other_errors_are_not_retried(self):
        runner = BenchmarkRunner({"detectors": {"summac": {"enabled": True}},
                                  "benchmark": {"output_dir": tempfile.mkdtemp()}})

        def broken(context, answer):
            raise ValueError("bad input")
        runner.detectors["summac"].detect = broken
        with self.assertRaises(ValueError):
            runner.families["summac"].score({"question": "q", "context": "c", "answer": "a"}, None)

    def test_validate_requires_a_generator_when_selfcheckgpt_is_enabled(self):
        config = {
            "detectors": {"selfcheckgpt": {"enabled": True, "method": "ngram"}},
        }
        runner = BenchmarkRunner({**config, "benchmark": {"output_dir": tempfile.mkdtemp()}})
        with self.assertRaises(ValueError):
            runner.validate({"synthetic": []}, generators=[])


class DocumentationTests(unittest.TestCase):
    def test_documentation_local_links_exist(self):
        documents = [ROOT / "README.md", ROOT / "METHOD_SOURCES.md"]
        documents.extend(sorted((ROOT / "docs").glob("*.md")))
        missing = []
        for document in documents:
            content = document.read_text()
            targets = re.findall(
                r"\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)", content
            )
            for target in targets:
                if not (document.parent / target).resolve().exists():
                    missing.append(f"{document.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_readme_and_architecture_have_diagrams(self):
        for doc in ("README.md", "docs/ARCHITECTURE.md"):
            blocks = re.findall(r"```mermaid\n(.*?)```", (ROOT / doc).read_text(), re.S)
            self.assertGreaterEqual(len(blocks), 2, doc)
            for block in blocks:
                self.assertRegex(block.lstrip(), r"^(flowchart|sequenceDiagram)\b", doc)

    def test_provenance_has_full_revisions_and_existing_adapters(self):
        registry = yaml.safe_load((ROOT / "provenance" / "sources.yaml").read_text())
        for source in registry["sources"].values():
            revision = source.get("revision")
            if revision:
                self.assertRegex(revision, r"^[0-9a-f]{40}$")
            adapter = source.get("local_adapter")
            if adapter:
                self.assertTrue((ROOT / adapter).exists(), adapter)


class ScriptedModel:
    """Minimal duck-typed generator returning canned responses in order."""

    def __init__(self, name: str, responses):
        self.name = name
        self._responses = list(responses)

    def generate(self, prompt, **kwargs):
        return self._responses.pop(0)

    def generate_batch(self, prompts, **kwargs):
        return ["sample"] * len(prompts)


class SelfRefineReducerTests(unittest.TestCase):
    def test_stops_when_model_reports_no_issues(self):
        model = ScriptedModel("m", ["NO_ISSUES"])
        reducer = SelfRefineReducer(model=model, max_iterations=3)
        result = reducer.reduce("When?", "context", initial_answer="Python was released in 1991.")
        self.assertEqual(result.final_answer, "Python was released in 1991.")
        self.assertEqual(result.initial_answer, "Python was released in 1991.")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.stopped_reason, "no_issues_reported")

    def test_revises_until_iteration_budget_is_spent(self):
        model = ScriptedModel("m", [
            "Unsupported claim here.",
            "Revised answer 1.",
            "Still an issue.",
            "Revised answer 2.",
        ])
        reducer = SelfRefineReducer(model=model, max_iterations=2)
        result = reducer.reduce("When?", "context", initial_answer="Initial answer.")
        self.assertEqual(result.final_answer, "Revised answer 2.")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.stopped_reason, "max_iterations")
        self.assertEqual(len(result.feedback_history), 2)

    def test_rejects_non_positive_iteration_budget(self):
        with self.assertRaises(ValueError):
            SelfRefineReducer(model=ScriptedModel("m", ["x"]), max_iterations=0)


class ScriptedChatModel:
    """Returns realistic replies by recognising which prompt it was given."""

    def __init__(self, name="model-a"):
        self.name = name
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        if prompt.startswith("Review the candidate answer"):
            return "The year 1985 is not supported by the context."
        if prompt.startswith("Rewrite the candidate answer"):
            return "Python was released in 1991."
        if prompt.startswith("Below is a question and a draft answer"):
            return "1. When was Python first released?\n2. Who created Python?"
        if prompt.startswith("Revise the draft answer"):
            return "Python was first released in 1991 by Guido van Rossum."
        if "Context:" not in prompt:
            return "Python was released in 1985."          # closed book: no context
        if kwargs.get("temperature") == 0.0:
            return "Python was released in 1991."          # greedy
        return "Python was released in 1985."              # grounded baseline (a hallucination)

    def sample_n(self, prompt, n=5, temperature=1.0):
        return ["Python was first released in 1991."] * n


class ReductionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, fake_upstream_modules())
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def _config(self, temp_dir, methods=None):
        return {
            "datasets": [{"name": "synthetic", "enabled": True, "source": "synthetic", "max_samples": 1}],
            "detectors": {"selfcheckgpt": {"enabled": True, "method": "ngram", "n_samples": 2, "threshold": 3.0},
                          "uqlm": {"enabled": True, "scorers": ["noncontradiction"]},
                          "uqlm_judge": {"enabled": True}},
            "judge": {"model": "judge:7b"},
            "reduction": {"methods": methods if methods is not None else
                          ["closed_book", "greedy", "self_refine_adapted", "cove_adapted", "uqlm_best_response"],
                          "max_iterations": 1},
            "benchmark": {"output_dir": temp_dir, "seed": 42},
        }

    def test_every_method_is_paired_with_the_same_baseline_and_scored(self):
        from benchmark.reduction_runner import paired_deltas
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            datasets = DatasetLoader(config, seed=42).load_all()
            runner = BenchmarkRunner(config)
            frame = ReductionRunner(config, runner).run(datasets, generators=[ScriptedChatModel()])
            self.assertEqual(list(frame["condition"]), ["baseline", "closed_book", "greedy",
                                                        "self_refine_adapted", "cove_adapted",
                                                        "uqlm_best_response"])
            self.assertTrue(frame["error"].isna().all(), frame["error"].tolist())
            for col in ("selfcheckgpt_ngram_score", "uqlm_noncontradiction_score", "uqlm_judge_score"):
                self.assertTrue(frame[col].notna().all(), col)
            answers = dict(zip(frame["condition"], frame["answer"]))
            self.assertIn("1985", answers["baseline"])
            self.assertIn("1991", answers["self_refine_adapted"])
            self.assertIn("1991", answers["cove_adapted"])
            self.assertIn("1991", answers["uqlm_best_response"])
            self.assertIn("1991", answers["greedy"])
            # the "not upstream" facts survive in the raw CSV
            status = dict(zip(frame["condition"], frame["reproduction_status"]))
            self.assertEqual(status["self_refine_adapted"], "local_inspired_baseline_NOT_an_upstream_reproduction")
            self.assertEqual(status["uqlm_best_response"], "official_uqlm_implementation")
            deltas = paired_deltas(frame, ["uqlm_judge_score"])
            by_cond = dict(zip(deltas["condition"], deltas["delta"]))
            self.assertLess(by_cond["self_refine_adapted"], 0)   # 1985 → 1991 lowers the risk
            self.assertEqual(by_cond["closed_book"], 0)          # still 1985
            self.assertTrue((Path(temp_dir) / "reduction_comparison.csv").exists())

    def test_uqlm_best_response_uses_official_class_with_prompts_out_of_nli(self):
        import uqlm
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir, ["uqlm_best_response"])
            ReductionRunner(config, BenchmarkRunner(config))._best._load()
            self.assertTrue(uqlm.SemanticEntropy.init_kwargs["use_best"])
            self.assertFalse(uqlm.SemanticEntropy.init_kwargs["prompts_in_nli"])

    def test_uqlm_detection_never_swaps_the_scored_answer(self):
        import uqlm
        from detectors.sampling import SampleBank
        from detectors.uqlm_detector import UQLMConsistencyDetector
        detector = UQLMConsistencyDetector(SampleBank(2), ["noncontradiction"])
        risk = detector.detect("When?", "ctx", "Python was released in 1985.", ReplayModel(["x"]))
        self.assertFalse(uqlm.BlackBoxUQ.init_kwargs["use_best"])
        self.assertAlmostEqual(risk["noncontradiction"], 0.9)   # risk = 1 - confidence

    def test_rejects_an_unqualified_method_name(self):
        """Config must say 'self_refine_adapted', never bare 'self_refine'."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir, ["self_refine"])
            with self.assertRaises(ValueError):
                ReductionRunner(config, BenchmarkRunner(config))

    def test_best_response_is_scored_leave_one_out(self):
        """The picked sample must not be part of its own evidence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir, ["uqlm_best_response"])
            datasets = DatasetLoader(config, seed=42).load_all()
            runner = BenchmarkRunner(config)
            seen = []
            original = runner.score_answer

            def spy(case, model):
                seen.append((case["case_id"], list(runner.bank.get(model, case["question"], case["context"]))))
                return original(case, model)
            runner.score_answer = spy
            model = ScriptedChatModel()
            frame = ReductionRunner(config, runner).run(datasets, generators=[model])
            best = frame[frame["condition"] == "uqlm_best_response"].iloc[0]
            evidence = dict(seen)[f"{best['sample_id']}:uqlm_best_response"]
            self.assertIn("1991", best["answer"])                  # a sample was picked
            self.assertNotIn(best["answer"], evidence)              # every copy removed, not just one
            self.assertIn("Python was released in 1985.", evidence)  # the baseline joins the evidence
            # the original samples are back in place after scoring
            sample = datasets["synthetic"][0]
            self.assertEqual(runner.bank.get(model, sample.question, sample.context),
                             ["Python was first released in 1991."] * 2)

    def test_one_failing_selfcheck_scorer_keeps_the_others(self):
        import selfcheckgpt.modeling_selfcheck as upstream
        detector = SelfCheckGPTDetector(ReplayModel(["Python was released in 1991."]),
                                        method=["ngram", "bertscore"], n_samples=2)

        class Broken:
            def __init__(self, *a, **k):
                pass

            def predict(self, **kwargs):
                raise IndexError("list index out of range")
        with patch.object(upstream, "SelfCheckBERTScore", Broken):
            result = detector.detect("When?", "ctx", "Python was released in 1991.")
        self.assertIn("ngram", result.scores)
        self.assertIn("bertscore", result.errors)

    def test_selfcheck_bertscore_model_is_loaded_once_on_the_configured_device(self):
        from detectors.selfcheckgpt_detector import _LoadedBERTScore
        built = []

        class FakeBERTScorer:
            def __init__(self, lang, rescale_with_baseline, device):
                built.append((lang, rescale_with_baseline, device))

            def score(self, cands, refs, verbose=False):
                return ("P", "R", "F1")
        fake = types.ModuleType("bert_score")
        fake.BERTScorer = FakeBERTScorer
        loaded = _LoadedBERTScore("cpu")
        with patch.dict(sys.modules, {"bert_score": fake}):
            for _ in range(5):   # upstream calls it once per sample
                result = loaded.score(["a"], ["b"], lang="en", verbose=False,
                                      rescale_with_baseline=True)
        self.assertEqual(result, ("P", "R", "F1"))
        self.assertEqual(built, [("en", True, "cpu")])

    def test_uqlm_cosine_model_gets_the_configured_device(self):
        from detectors.uqlm_detector import _sentence_transformer_on
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = lambda name, **kwargs: kwargs
        with patch.dict(sys.modules, {"sentence_transformers": fake}):
            with _sentence_transformer_on("cpu"):
                from sentence_transformers import SentenceTransformer   # as uqlm imports it
                self.assertEqual(SentenceTransformer("m", trust_remote_code=True),
                                 {"device": "cpu", "trust_remote_code": True})
            self.assertEqual(fake.SentenceTransformer("m"), {})        # restored afterwards

    def test_cove_parses_numbered_questions(self):
        from reducers.cove import ChainOfVerificationReducer
        self.assertEqual(ChainOfVerificationReducer.parse_questions("1. Who?\n- When was it?\n\nQ3: Where?"),
                         ["Who?", "When was it?", "Where?"])
        self.assertEqual(ChainOfVerificationReducer.parse_questions("Here are the questions:\n1. Who?"),
                         ["Who?"])


class CountingModel(ReplayModel):
    def __init__(self, responses, name="counting"):
        super().__init__(responses, name=name)
        self.calls = 0

    def sample_n(self, prompt, n=5, temperature=1.0):
        self.calls += 1
        return super().sample_n(prompt, n=n, temperature=temperature)


class ResourceTests(unittest.TestCase):
    def _remote(self, source: Path, sha: str):
        from utils.resources import RemoteFile
        return RemoteFile("test file", source.as_uri(), sha, source.stat().st_size)

    def test_download_verifies_checksum_and_keeps_good_file(self):
        import hashlib
        from utils.resources import download
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            source.write_text('{"a": 1}\n')
            target = Path(temp_dir) / "out" / "data.json"
            download(self._remote(source, hashlib.sha256(source.read_bytes()).hexdigest()), target)
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_download_discards_a_file_with_the_wrong_checksum(self):
        from utils.resources import download
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            source.write_text("tampered\n")
            target = Path(temp_dir) / "data.json"
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                download(self._remote(source, "0" * 64), target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("data.json.part").exists())

    def test_every_configured_dataset_file_has_a_pinned_download(self):
        from utils.resources import REMOTE_FILES
        config = yaml.safe_load((ROOT / "config.yaml").read_text())
        for ds in config["datasets"]:
            for key in ("path", "responses_path", "sources_path"):
                if ds.get(key):
                    self.assertIn(ds[key], REMOTE_FILES, ds["name"])
        self.assertIn(config["detectors"]["alignscore"]["checkpoint_path"], REMOTE_FILES)


class RunIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, fake_upstream_modules())
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_factual_and_hallucinated_answers_share_one_sample_set(self):
        model = CountingModel(["Python was released in 1991."])
        detector = SelfCheckGPTDetector(model, method="ngram", n_samples=2, threshold=3.0)
        detector.detect("When?", "ctx", "Python was released in 1991.")
        detector.detect("When?", "ctx", "Python was released in 1985.")
        self.assertEqual(model.calls, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "samples.jsonl"
            self.assertEqual(detector.export_samples(path), 1)
            record = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(record["model"], "counting")
            self.assertEqual(len(record["samples"]), 2)

    def test_empty_answer_is_a_failure_not_a_score(self):
        detector = SelfCheckGPTDetector(ReplayModel(["x"]), method="ngram")
        with self.assertRaises(ValueError):
            detector.detect("When?", "ctx", "   ")

    def test_summary_counts_failed_cases(self):
        import pandas as pd
        frame = pd.DataFrame({
            "label": [0, 1, 0, 1],
            "x_score": [0.1, 0.9, None, 0.8],
        })
        row = DetectorValidator().evaluate_frame(frame, ["x_score"], {"x_score": 0.5}).iloc[0]
        self.assertEqual(row["n_cases"], 3)
        self.assertEqual(row["n_failed"], 1)

    def _write_run(self, folder: Path, detectors: dict, generators=None) -> None:
        import pandas as pd
        config = {
            "datasets": [{"name": "synthetic", "enabled": True, "source": "synthetic", "max_samples": 4}],
            "detectors": detectors,
            "benchmark": {"output_dir": str(folder), "seed": 42},
        }
        datasets = DatasetLoader(config, seed=42).load_all()
        runner = BenchmarkRunner(config)
        raw = runner.validate(datasets, generators=generators)
        frames = []
        for column, threshold in runner.thresholds().items():
            frame = DetectorValidator().evaluate_frame(
                raw.drop_duplicates("case_id"), [column], {column: threshold})
            frame["model"] = ("replay" if column in runner.generator_columns()
                              else "n/a (model-independent detector)")
            frames.append(frame)
        pd.concat(frames).to_csv(folder / "detector_validation_summary.csv", index=False)
        # the report's run-plan section reads these, as in a real run folder
        (folder / "config_used.yaml").write_text(yaml.safe_dump({**config, "judge": {"model": "judge:7b"},
                                                                 "reduction": {"methods": []}}))

    def test_single_run_folder_gets_the_full_report_set(self):
        from reporting import generate

        with tempfile.TemporaryDirectory() as temp_dir:
            run = Path(temp_dir) / "run_01"
            run.mkdir()
            self._write_run(run, {"selfcheckgpt": {"enabled": True, "method": "ngram",
                                                   "n_samples": 2, "threshold": 3.0}},
                            [ReplayModel(["Python was released in 1991."])])
            produced = generate(run)
            self.assertIn("4 questions from 1 dataset", (run / "report.html").read_text(encoding="utf-8"))
            for name in ("REPORT.md", "report.html", "report.docx", "takeaways.md"):
                self.assertTrue((run / name).is_file(), name)
            self.assertTrue(any((run / "charts").glob("*.png")))
            self.assertTrue((run / "tables" / "per_dataset.csv").is_file())
            self.assertEqual(produced["report (md)"], run / "REPORT.md")

    def test_combined_report_spans_every_detector_and_run(self):
        import pandas as pd
        from reporting import generate

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run in ("run_01", "run_02"):
                sc = root / "selfcheckgpt" / run
                sc.mkdir(parents=True)
                self._write_run(sc, {"selfcheckgpt": {"enabled": True, "method": "ngram",
                                                      "n_samples": 2, "threshold": 3.0}},
                                [ReplayModel(["Python was released in 1991."])])
                mc = root / "minicheck" / run
                mc.mkdir(parents=True)
                self._write_run(mc, {"minicheck": {"enabled": True}, "summac": {"enabled": True}})
            generate(root, out_dir=root / "combined")
            ms = pd.read_csv(root / "combined" / "tables" / "summary_mean_std.csv")
            self.assertEqual(sorted(ms["detector"]), ["minicheck", "selfcheckgpt_ngram", "summac"])
            self.assertTrue((ms["n_runs"] == 2).all())
            by_run = pd.read_csv(root / "combined" / "tables" / "summary_by_run.csv")
            self.assertEqual(sorted(by_run["run"].unique()), ["run_01", "run_02"])
            for name in ("REPORT.md", "report.html", "report.docx", "takeaways.md", "raw_all_runs.csv"):
                self.assertTrue((root / "combined" / name).is_file(), name)

    def test_run_full_layout_one_folder_per_run_plus_combined(self):
        """results/<name>/run_XX/<group>/ → each run's report covers every
        group; combined/ covers every run."""
        import pandas as pd
        from reporting import generate

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run in ("run_01", "run_02"):
                core, mini = root / run / "core", root / run / "minicheck"
                core.mkdir(parents=True)
                mini.mkdir(parents=True)
                self._write_run(core, {"selfcheckgpt": {"enabled": True, "method": "ngram",
                                                        "n_samples": 2, "threshold": 3.0}},
                                [ReplayModel(["Python was released in 1991."])])
                self._write_run(mini, {"minicheck": {"enabled": True}})
                rows = [{"sample_id": f"q{i}", "dataset": "synthetic", "model": "replay",
                         "question": "When?", "condition": cond, "answer": "a",
                         "selfcheckgpt_ngram_score": 1.0 + (0.5 if cond == "closed_book" else 0) + i / 10}
                        for i in range(12) for cond in ("baseline", "closed_book")]
                pd.DataFrame(rows).to_csv(core / "reduction_comparison.csv", index=False)
                pd.DataFrame([{"sample_id": r["sample_id"], "model": "replay", "condition": r["condition"],
                               "minicheck_score": 0.2 + (0.3 if r["condition"] == "closed_book" else 0)}
                              for r in rows]).to_csv(mini / "reduction_scores.csv", index=False)
            for run in ("run_01", "run_02"):
                generate(root / run)
                ranking = pd.read_csv(root / run / "tables" / "detector_ranking.csv")
                self.assertEqual(sorted(ranking["detector"]), ["MiniCheck", "SelfCheckGPT n-gram"])
                report_html = (root / run / "report.html").read_text(encoding="utf-8")
                self.assertIn("Baseline versus each method", report_html)
                self.assertIn("lower measured hallucination risk", report_html)
                self.assertTrue((root / run / "charts" / "baseline_vs_methods.png").is_file())
                self.assertFalse((root / run / "charts" / "answer_latency.png").exists())
                for name in ("report.docx", "report.html", "REPORT.md", "takeaways.md"):
                    self.assertTrue((root / run / name).is_file(), f"{run}/{name}")
            generate(root, out_dir=root / "combined")
            ms = pd.read_csv(root / "combined" / "tables" / "summary_mean_std.csv")
            self.assertTrue((ms["n_runs"] == 2).all())
            both = pd.read_csv(root / "combined" / "reduction_all_runs.csv")
            self.assertTrue(both["minicheck_score"].notna().all())   # joined from the other group
            self.assertEqual(sorted(both["run"].unique()), ["run_01", "run_02"])

    def test_smoke_fills_only_missing_numbers(self):
        import argparse
        import main
        args = argparse.Namespace(smoke=True, runs=None, max_samples=5, n_samples=None, max_iterations=None)
        main.apply_smoke(args)
        self.assertEqual((args.runs, args.max_samples, args.n_samples, args.max_iterations), (2, 5, 2, 1))

    def test_reduction_scores_from_every_environment_are_kept(self):
        """MiniCheck's and SummaC's reduction_scores.csv share the same keys;
        both environments' columns must survive the merge, for every run."""
        import pandas as pd
        from reporting import aggregate as agg
        keys = {"sample_id": "q1", "model": "m", "condition": "baseline"}

        def run(folder, run_name, reduction=None, scores=None):
            return agg.RunData(folder=Path(folder) / run_name, run=run_name,
                               summary=pd.DataFrame(), raw=None,
                               reduction=reduction, reduction_scores=scores,
                               manifest=None, config=None)
        runs = []
        for name in ("run_01", "run_02"):
            runs.append(run("core", name, reduction=pd.DataFrame([{**keys, "run": name, "answer": "a",
                                                                   "uqlm_judge_score": 0.1}])))
            runs.append(run("minicheck", name, scores=pd.DataFrame([{**keys, "run": name, "minicheck_score": 0.2}])))
            runs.append(run("summac", name, scores=pd.DataFrame([{**keys, "run": name, "summac_score": 0.3}])))
        table = agg.reduction_table(runs)
        self.assertEqual(len(table), 2)
        self.assertTrue(table["minicheck_score"].notna().all())
        self.assertTrue(table["summac_score"].notna().all())

    def _plan(self, run=None, datasets=None, sections=None, reduction=None, **flags):
        """apply_plan on a fresh config; `sections` are the config's detector
        sections, `flags` the command-line arguments."""
        import argparse
        import main
        config = {
            "run": dict(run or {"runs": 3, "samples_per_dataset": 50, "detectors": ["selfcheckgpt", "minicheck"],
                                "selfcheckgpt_samples": 5, "reduce": True, "reduction_iterations": 3}),
            "datasets": [dict(d) for d in (datasets or [{"name": "a", "source": "json"},
                                                        {"name": "b", "source": "synthetic", "max_samples": 8}])],
            "detectors": {k: dict(v) for k, v in (sections or {}).items()},
            "reduction": dict(reduction or {}), "benchmark": {},
        }
        args = argparse.Namespace(**{"output": None, "detectors": None, "max_samples": None, "n_samples": None,
                                     "max_iterations": None, "device": "cpu", "runs": None, "reduce": False,
                                     "no_reduce": False, **flags})
        return main.apply_plan(config, args), config

    def test_auto_device_leaves_a_small_gpu_to_ollama(self):
        import os
        import main
        main.DEVICE_NOTE.clear()
        with patch("utils.gpu.nvidia_gpu", return_value=("NVIDIA GeForce RTX 5060", 7.9)), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            plan, config = self._plan(device=None)
            self.assertEqual(config["detectors"]["selfcheckgpt"]["device"], "cpu")
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")   # libraries cannot take the GPU
            self.assertIn("RTX 5060", main.DEVICE_NOTE[0])
        main.DEVICE_NOTE.clear()
        with patch("utils.gpu.nvidia_gpu", return_value=("NVIDIA A100", 40.0)), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            plan, config = self._plan(device=None)
            self.assertEqual(config["detectors"]["selfcheckgpt"]["device"], "cuda")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
            plan, config = self._plan(device="cpu")   # an explicit flag wins
            self.assertEqual(config["detectors"]["uqlm"]["device"], "cpu")

    def test_run_plan_precedence_flag_over_section_over_plan(self):
        # run block as the default
        plan, config = self._plan()
        self.assertEqual(plan["runs"], 3)
        self.assertEqual(plan["samples_per_dataset"], {"a": 50, "b": 8})   # a dataset's own cap wins
        self.assertEqual(plan["selfcheckgpt_samples"], 5)
        self.assertEqual(plan["reduction_iterations"], 3)
        self.assertEqual(plan["detectors"], ["selfcheckgpt", "minicheck"])
        self.assertTrue(plan["reduce"])
        self.assertTrue(config["detectors"]["minicheck"]["enabled"])
        self.assertFalse(config["detectors"]["summac"]["enabled"])
        # a detector section's own setting wins over the run block
        plan, _ = self._plan(sections={"selfcheckgpt": {"n_samples": 7}}, reduction={"max_iterations": 1})
        self.assertEqual((plan["selfcheckgpt_samples"], plan["reduction_iterations"]), (7, 1))
        # flags win over everything
        plan, _ = self._plan(sections={"selfcheckgpt": {"n_samples": 7}}, n_samples=2, max_samples=2,
                             max_iterations=4, runs=1)
        self.assertEqual(plan["selfcheckgpt_samples"], 2)
        self.assertEqual(plan["samples_per_dataset"], {"a": 2, "b": 2})
        self.assertEqual((plan["reduction_iterations"], plan["runs"]), (4, 1))
        # reduce: flag > run block; the section's `enabled` is ignored; needs a sampling detector
        plan, _ = self._plan(no_reduce=True)
        self.assertFalse(plan["reduce"])
        run_off = {"runs": 1, "detectors": ["selfcheckgpt"], "reduce": False}
        plan, _ = self._plan(run=run_off, reduction={"enabled": True})
        self.assertFalse(plan["reduce"])
        plan, _ = self._plan(run=run_off, reduce=True)
        self.assertTrue(plan["reduce"])
        plan, _ = self._plan(detectors=["minicheck"])
        self.assertFalse(plan["reduce"])  # reduction needs selfcheckgpt or uqlm
        # without a flag or run.detectors, the sections' `enabled` flags decide
        plan, _ = self._plan(run={"runs": 1}, sections={"summac": {"enabled": True}})
        self.assertEqual(plan["detectors"], ["summac"])

    def test_summary_keeps_a_row_for_every_detector_and_model(self):
        import pandas as pd
        import main

        class Runner:
            def thresholds(self):
                return {"sc_score": 0.5, "judge_score": 0.5, "dead_score": 0.5}

            def generator_columns(self):
                return ["sc_score"]
        rows = []
        for model in ("m1", "m2"):
            for case, label in (("q:0", 0), ("q:1", 1)):
                rows.append({"case_id": case, "label": label, "model": model,
                             "sc_score": None if model == "m2" else (0.9 if label else 0.1),
                             "judge_score": 0.8 if label else 0.2, "dead_score": None})
        summary = main.summarise(pd.DataFrame(rows), Runner())
        got = {(r.detector, r.model): (r.n_cases, r.n_failed) for r in summary.itertuples()}
        self.assertEqual(got[("sc", "m1")], (2, 0))
        self.assertEqual(got[("sc", "m2")], (0, 2))            # the failed model keeps its row
        self.assertEqual(got[("judge", main.MODEL_INDEPENDENT)], (2, 0))   # scored once per case
        self.assertEqual(got[("dead", main.MODEL_INDEPENDENT)], (0, 2))

    def test_confidence_interval_needs_enough_pairs(self):
        from reporting import aggregate as agg
        self.assertTrue(all(v != v for v in agg.bootstrap_ci([0.1, -0.2, 0.3])))   # NaN: too few
        low, high = agg.bootstrap_ci([-0.2] * 8 + [-0.1] * 8)
        self.assertLess(high, 0)            # clearly below zero
        self.assertLessEqual(low, high)

    def test_reduction_interval_needs_distinct_questions(self):
        import pandas as pd
        from reporting import aggregate as agg

        tiny = pd.DataFrame({"condition": ["greedy"] * 10, "detector": ["mini"] * 10,
                             "sample_id": ["q1"] * 5 + ["q2"] * 5,
                             "model": [f"m{i}" for i in range(5)] * 2,
                             "delta": [-0.2] * 10})
        row = agg.method_effects(tiny, ["mini"]).iloc[0]
        self.assertEqual((row.n_pairs, row.n_questions), (10, 2))
        self.assertTrue(pd.isna(row.ci_low) and pd.isna(row.ci_high))

        enough = pd.concat([tiny.assign(sample_id=f"q{i}") for i in range(10)], ignore_index=True)
        row = agg.method_effects(enough, ["mini"]).iloc[0]
        self.assertLess(row.ci_high, 0)

    def test_constant_detector_is_flagged_as_no_signal(self):
        import pandas as pd
        from reporting import aggregate as agg
        raw = pd.DataFrame({"a_score": [0.1, 0.9, 0.4], "flat_score": [1.0, 1.0, 1.0]})
        self.assertEqual(agg.constant_detectors(raw), ["flat"])

    def test_mean_std_across_runs(self):
        import pandas as pd
        from reporting import aggregate as agg
        frame = pd.DataFrame({"run": ["run_01", "run_02", "run_03"], "detector": ["d"] * 3,
                              "roc_auc": [0.6, 0.7, 0.8]})
        row = agg.mean_std(frame, ["detector"], ["roc_auc"]).iloc[0]
        self.assertAlmostEqual(row["roc_auc_mean"], 0.7)
        self.assertAlmostEqual(row["roc_auc_std"], 0.1)          # sample std (n − 1)
        self.assertEqual(row["n_runs"], 3)

    def test_ollama_failure_raises_instead_of_returning_empty_text(self):
        fake_ollama = types.ModuleType("ollama")

        class FailingClient:
            def __init__(self, host=None, **kwargs):
                pass

            def chat(self, **kwargs):
                return {"message": {"content": "<think>only reasoning</think>"}}

        fake_ollama.Client = FailingClient
        with patch.dict(sys.modules, {"ollama": fake_ollama}), \
                patch("models.ollama_model.time.sleep"):
            from models.ollama_model import OllamaModel
            model = OllamaModel("m", {"model": "m:1b"})
            with self.assertRaises(RuntimeError):
                model.generate("hi")

    def test_model_that_does_not_fit_gets_room_and_retries(self):
        """Ollama's "failed to load … resource limitations": other loaded
        models are unloaded, the benchmark frees GPU memory, and the retry
        succeeds."""
        fake_ollama = types.ModuleType("ollama")
        unloaded, freed = [], []

        class TightClient:
            calls = 0

            def __init__(self, host=None, **kwargs):
                pass

            def chat(self, **kwargs):
                TightClient.calls += 1
                if TightClient.calls == 1:
                    raise RuntimeError("model failed to load, this may be due to resource "
                                       "limitations or an internal error")
                return {"message": {"content": "Four."}}

            def ps(self):
                return {"models": [{"model": "mistral:7b"}, {"model": "gpt-oss:20b"}]}

            def generate(self, model, prompt, keep_alive):
                unloaded.append(model)

        fake_ollama.Client = TightClient
        with patch.dict(sys.modules, {"ollama": fake_ollama}), \
                patch("models.ollama_model.time.sleep"):
            from models.ollama_model import OllamaModel
            model = OllamaModel("gpt-oss-20b", {"model": "gpt-oss:20b"})
            model.on_memory_pressure = lambda: freed.append(1)
            self.assertEqual(model.generate("2 + 2?"), "Four.")
        self.assertEqual(unloaded, ["mistral:7b"])   # not the model being loaded
        self.assertEqual(freed, [1])


if __name__ == "__main__":
    unittest.main()
