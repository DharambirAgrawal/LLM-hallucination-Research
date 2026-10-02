"""No-download run-plan checks; stdlib only."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.offline_plan import local_datasets


class OfflinePlanTests(unittest.TestCase):
    def test_missing_inputs_are_never_passed_to_loaders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            present = Path(temp_dir) / "present.json"
            present.write_text("[]", encoding="utf-8")
            config = {"datasets": [
                {"name": "json", "source": "json", "path": str(present)},
                {"name": "ragtruth", "source": "ragtruth", "responses_path": str(present),
                 "sources_path": str(Path(temp_dir) / "missing.jsonl")},
                {"name": "halubench", "source": "halubench", "path": str(Path(temp_dir) / "missing.parquet")},
                {"name": "remote", "source": "hf", "hf_path": "example/dataset"},
                {"name": "disabled", "enabled": False, "source": "synthetic"},
                {"name": "synthetic", "source": "synthetic"},
            ]}
            original = list(config["datasets"])
            ready, unavailable = local_datasets(config)
            self.assertEqual([d["name"] for d in ready["datasets"]], ["json", "synthetic"])
            self.assertEqual(unavailable, ["ragtruth", "halubench", "remote"])
            self.assertEqual(config["datasets"], original)


if __name__ == "__main__":
    unittest.main()
