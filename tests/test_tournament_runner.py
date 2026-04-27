import unittest

from src.tournament_runner import RandomPartitionTournamentRunner


class RandomPartitionTournamentRunnerTests(unittest.TestCase):
    def test_runner_pads_short_player_list_before_running(self):
        config = {
            "players": [
                {"path": "src.players.TA.random_player", "class": "RandomPlayer"},
            ],
            "engine": {
                "n_players": 4,
                "n_rounds": 10,
            },
            "tournament": {
                "type": "random_partition",
                "duplication_mode": "none",
                "num_games_per_player": 1,
                "num_workers": 1,
            },
        }

        runner = RandomPartitionTournamentRunner(config)

        self.assertEqual(runner.original_num_players, 1)
        self.assertEqual(len(runner.player_classes), 4)
        self.assertEqual(len(runner.player_configs), 4)
        self.assertTrue(all(player["class"] == "RandomPlayer" for player in runner.player_configs))
        self.assertEqual(sum(player.get("label") == "(PAD)" for player in runner.player_configs), 3)


if __name__ == "__main__":
    unittest.main()
