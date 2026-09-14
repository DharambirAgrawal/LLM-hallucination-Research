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

import yaml

from benchmark.detector_validation import DetectorValidator
from benchmark.runner import BenchmarkRunner
from data.datasets import DatasetLoader
from detectors.alignscore_detector import AlignScoreDetector
from detectors.minicheck_detector import MiniCheckDetector
from detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from detectors.summac_detector import SummaCDetector
from models.openai_compatible_model import OpenAICompatibleModel
from models.replay_model import ReplayModel
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


class RemoteAdapterTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
