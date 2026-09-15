"""Offline tests: no LLM calls and no detector/model downloads."""
from __future__ import annotations

import json
import os
import re
import struct
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
from models.openai_compatible_model import OpenAICompatibleModel
from models.replay_model import ReplayModel
from models.transformers_model import TransformersModel
from reducers.self_refine import SelfRefineReducer
from utils.env_loader import load_env_file


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

    return {
        "selfcheckgpt": selfcheck_package,
        "selfcheckgpt.modeling_selfcheck": selfcheck_module,
        "minicheck": minicheck_package,
        "minicheck.minicheck": minicheck_module,
        "summac": summac_package,
        "summac.model_summac": summac_module,
        "alignscore": alignscore_module,
    }


class DataAndMetricsTests(unittest.TestCase):
    def test_default_config_generates_balanced_fixed_cases(self):
        config = yaml.safe_load((ROOT / "config.yaml").read_text())
        datasets = DatasetLoader(config, seed=42).load_all()
        cases = DatasetLoader.detection_cases(datasets["synthetic"])
        self.assertEqual(len(cases), 16)
        self.assertEqual(sum(case["label"] for case in cases), 8)

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
            for name in ("selfcheckgpt", "minicheck", "summac", "alignscore"):
                self.assertTrue(frame[f"{name}_score"].notna().all())
                self.assertTrue(frame[f"{name}_error"].isna().all())
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
            self.assertTrue(frame["selfcheckgpt_score"].notna().all())

    def test_validate_requires_a_generator_when_selfcheckgpt_is_enabled(self):
        config = {
            "detectors": {"selfcheckgpt": {"enabled": True, "method": "ngram"}},
        }
        runner = BenchmarkRunner({**config, "benchmark": {"output_dir": tempfile.mkdtemp()}})
        with self.assertRaises(ValueError):
            runner.validate({"synthetic": []}, generators=[])


class RemoteAdapterTests(unittest.TestCase):
    def test_local_transformers_model_requires_immutable_revision(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            TransformersModel(
                "moving-model",
                {"model": "example/model", "revision": "main"},
            )

    def test_local_transformers_model_uses_full_tokenizer_inputs(self):
        generated = {}

        class FakeBatch(dict):
            def to(self, device):
                generated["input_device"] = device
                return self

        class FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                generated["template_kwargs"] = kwargs
                return FakeBatch(
                    input_ids=np.array([[1, 2, 3]]),
                    attention_mask=np.array([[1, 1, 1]]),
                )

            def decode(self, token_ids, **kwargs):
                return "local response"

        class FakeModel:
            def to(self, device):
                return self

            def eval(self):
                return self

            def generate(self, **kwargs):
                generated["generate_kwargs"] = kwargs
                return np.array([[1, 2, 3, 4, 5]])

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return FakeTokenizer()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                generated["load_kwargs"] = kwargs
                return FakeModel()

        class InferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.inference_mode = InferenceMode
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoModelForCausalLM = FakeAutoModel

        config = {
            "model": "official/model",
            "revision": "a" * 40,
            "max_tokens": 8,
        }
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            result = TransformersModel("local", config).generate("hello")

        self.assertEqual(result, "local response")
        self.assertIn("attention_mask", generated["generate_kwargs"])
        self.assertTrue(generated["template_kwargs"]["return_dict"])
        self.assertEqual(generated["load_kwargs"]["attn_implementation"], "eager")

    def test_env_file_loading_without_overwriting_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "# comment\nGROQ_API_KEY='from-file'\nexport SECOND_KEY=two\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"GROQ_API_KEY": "already-set"}, clear=False):
                loaded = load_env_file(env_file)
                self.assertEqual(os.environ["GROQ_API_KEY"], "already-set")
                self.assertEqual(os.environ["SECOND_KEY"], "two")
                self.assertEqual(loaded, {"SECOND_KEY"})

    def test_openai_compatible_payload_and_response(self):
        received = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {"choices": [{"message": {"content": "test answer"}}]}
                ).encode()

        def fake_urlopen(request, timeout):
            received["path"] = request.full_url
            received["authorization"] = request.headers.get("Authorization")
            received["json"] = json.loads(request.data)
            received["timeout"] = timeout
            return FakeResponse()

        config = {
            "model": "test-model",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "OFFLINE_TEST_API_KEY",
            "request_fields": {"reasoning_effort": "minimal"},
        }
        with patch.dict(os.environ, {"OFFLINE_TEST_API_KEY": "secret"}):
            with patch("models.openai_compatible_model.urlopen", fake_urlopen):
                model = OpenAICompatibleModel("test", config)
                self.assertEqual(model.generate("hello", seed=42), "test answer")
        self.assertEqual(received["path"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(received["authorization"], "Bearer secret")
        self.assertEqual(received["json"]["model"], "test-model")
        self.assertEqual(received["json"]["seed"], 42)
        self.assertEqual(received["json"]["reasoning_effort"], "minimal")
        self.assertEqual(received["timeout"], 120)

    def test_provider_smoke_defaults_are_bounded_and_free_tier(self):
        from scripts.provider_selfcheck_smoke import (
            CASES,
            DEFAULT_PROVIDERS,
            PROVIDERS,
            parse_providers,
        )

        self.assertEqual(
            set(PROVIDERS), {"openrouter", "gemini", "gemini-flash", "mistral"}
        )
        self.assertEqual(PROVIDERS["openrouter"]["model"], "openrouter/free")
        self.assertEqual(PROVIDERS["gemini"]["model"], "gemini-3.5-flash-lite")
        self.assertEqual(PROVIDERS["gemini-flash"]["model"], "gemini-3.5-flash")
        self.assertEqual(PROVIDERS["mistral"]["model"], "mistral-small-latest")
        self.assertEqual(len(CASES) * 2 * len(parse_providers(DEFAULT_PROVIDERS)), 8)


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

    def test_diagrams_are_valid_wide_pngs(self):
        for name in ("research-pipeline.png", "official-detectors.png"):
            data = (ROOT / "assets" / "diagrams" / name).read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", data[16:24])
            self.assertGreater(width, height)
            self.assertGreaterEqual(width, 1600)

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


class ReductionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, fake_upstream_modules())
        self.modules.start()

    def tearDown(self):
        self.modules.stop()

    def test_rejects_a_non_selfcheckgpt_detector(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            alignscore = AlignScoreDetector(checkpoint_path=checkpoint.name)
            with self.assertRaises(ValueError):
                ReductionRunner({"benchmark": {"output_dir": tempfile.mkdtemp()}}, alignscore)

    def test_scores_baseline_and_refined_answers_with_the_frozen_detector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "datasets": [{
                    "name": "synthetic", "enabled": True, "source": "synthetic", "max_samples": 1,
                }],
                "reduction": {"method": "self_refine_adapted", "max_iterations": 1},
                "benchmark": {"output_dir": temp_dir, "seed": 42},
            }
            datasets = DatasetLoader(config, seed=42).load_all()
            detector = SelfCheckGPTDetector(method="ngram", n_samples=1, threshold=3.0)
            model = ScriptedModel("model-a", [
                "Python was released in 1985.",              # baseline answer
                "The year 1985 is not supported by the context.",  # feedback (not NO_ISSUES)
                "Python was released in 1991.",              # refined answer
            ])
            frame = ReductionRunner(config, detector).run(datasets, generators=[model])
            self.assertEqual(len(frame), 1)
            row = frame.iloc[0]
            self.assertIsNone(row["error"])
            self.assertEqual(row["baseline_answer"], "Python was released in 1985.")
            self.assertEqual(row["refined_answer"], "Python was released in 1991.")
            self.assertEqual(row["baseline_score"], 4.0)
            self.assertEqual(row["refined_score"], 1.0)
            self.assertEqual(row["score_delta"], -3.0)
            self.assertEqual(row["iterations"], 1)
            self.assertEqual(row["stopped_reason"], "max_iterations")
            # The "not upstream" fact must be readable from the raw CSV alone.
            self.assertEqual(row["method"], "self_refine_adapted")
            self.assertEqual(row["reproduction_status"], "local_inspired_baseline_NOT_an_upstream_reproduction")
            self.assertTrue((Path(temp_dir) / "reduction_comparison.csv").exists())

    def test_rejects_an_unqualified_method_name(self):
        """Config must say 'self_refine_adapted', never bare 'self_refine'."""
        config = {
            "reduction": {"method": "self_refine"},
            "benchmark": {"output_dir": tempfile.mkdtemp()},
        }
        detector = SelfCheckGPTDetector(method="ngram")
        with self.assertRaises(ValueError):
            ReductionRunner(config, detector)


if __name__ == "__main__":
    unittest.main()
