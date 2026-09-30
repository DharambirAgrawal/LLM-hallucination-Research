"""Small run-plan checks that need no model or detector packages."""
import unittest

from utils.smoke import configure_two_question_smoke


class TwoQuestionSmokeTests(unittest.TestCase):
    def test_uses_two_questions_total_and_all_detectors(self):
        config = {
            "datasets": [
                {"name": "disabled", "enabled": False, "max_samples": 20},
                {"name": "qa", "enabled": True, "max_samples": 20},
                {"name": "dialogue", "enabled": True, "max_samples": 20},
            ],
            "run": {"detectors": ["selfcheckgpt"], "reduce": False},
            "reduction": {"methods": ["closed_book", "greedy"]},
        }
        name = configure_two_question_smoke(config, ("selfcheckgpt", "alignscore"))
        self.assertEqual(name, "qa")
        self.assertEqual(config["datasets"], [{"name": "qa", "enabled": True, "max_samples": 2}])
        self.assertEqual(config["run"]["detectors"], ["selfcheckgpt", "alignscore"])
        self.assertTrue(config["run"]["reduce"])
        self.assertEqual(config["reduction"]["methods"], ["closed_book", "greedy"])

    def test_requires_enabled_dataset(self):
        with self.assertRaisesRegex(ValueError, "at least one enabled dataset"):
            configure_two_question_smoke({"datasets": [{"name": "off", "enabled": False}]}, ())


if __name__ == "__main__":
    unittest.main()
