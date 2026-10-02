"""AlignScore import failures must expose the incompatible dependency."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class AlignScoreImportTests(unittest.TestCase):
    def test_dependency_import_error_is_not_reported_as_missing_alignscore(self):
        path = ROOT / "detectors" / "alignscore_detector.py"
        spec = importlib.util.spec_from_file_location("alignscore_adapter_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)

            class BrokenAlignScore(types.ModuleType):
                def __getattr__(self, name):
                    if name == "AlignScore":
                        raise ImportError("cannot import name 'AdamW' from 'transformers'")
                    raise AttributeError(name)

            with tempfile.NamedTemporaryFile() as checkpoint:
                detector = module.AlignScoreDetector(checkpoint_path=checkpoint.name)
                with patch.dict(sys.modules, {"alignscore": BrokenAlignScore("alignscore")}):
                    with self.assertRaisesRegex(RuntimeError, "AdamW.*requirements/alignscore.txt"):
                        detector.detect("context", "answer")


if __name__ == "__main__":
    unittest.main()
