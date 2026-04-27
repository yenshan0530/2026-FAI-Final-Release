import sys
import unittest

from src.game_utils import _compiled_module_version_mismatch_message, load_players


class GameUtilsTests(unittest.TestCase):
    def test_compiled_module_version_mismatch_message_reports_public_baseline_requirement(self):
        message = _compiled_module_version_mismatch_message("src.players.TA.public_baselines1")

        if sys.version_info[:2] == (3, 13):
            self.assertIsNone(message)
            return

        self.assertIsNotNone(message)
        self.assertIn("src.players.TA.public_baselines1", message)
        self.assertIn("public_baselines1.cpython-313-x86_64-linux-gnu.so", message)
        self.assertIn("CPython 3.13", message)
        self.assertIn(sys.version.split()[0], message)

    @unittest.skipIf(sys.version_info[:2] == (3, 13), "Mismatch is only expected off Python 3.13")
    def test_load_players_raises_clear_error_for_version_mismatched_baseline(self):
        config = {
            "players": [
                {"path": "src.players.TA.public_baselines1", "class": "Baseline1"},
            ]
        }

        with self.assertRaises(ImportError) as ctx:
            load_players(config)

        message = str(ctx.exception)
        self.assertIn("matching Python interpreter", message)
        self.assertNotIn("No module named", message)


if __name__ == "__main__":
    unittest.main()
