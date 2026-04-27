import unittest
from unittest.mock import patch

from src.engine import Engine
from src.players.TA.random_player import RandomPlayer
from src.players.cfr.full_cfr_player import FullCFRPlayer


class FullCFRPlayerTests(unittest.TestCase):
    def test_action_reaches_cfr_solver_with_undealt_cards_in_unseen_pool(self):
        player = FullCFRPlayer(0, max_cfr_hand_size=10)
        players = [player, RandomPlayer(1), RandomPlayer(2), RandomPlayer(3)]
        engine = Engine({"n_players": 4, "n_rounds": 10, "seed": 7}, players)

        hand = engine.hands[0].copy()
        history = {
            "board": [row.copy() for row in engine.board],
            "scores": engine.scores.copy(),
            "round": engine.round,
            "history_matrix": [row.copy() for row in engine.history_matrix],
            "board_history": [[row.copy() for row in board] for board in engine.board_history],
            "score_history": [scores.copy() for scores in engine.score_history],
        }
        unseen_cards = player._collect_unseen_cards(hand, history)
        expected_hidden_cards = (len(history["scores"]) - 1) * len(hand)
        preferred_card = hand[-1]
        strategy = {card: 1.0 if card == preferred_card else 0.0 for card in hand}

        self.assertGreater(len(unseen_cards), expected_hidden_cards)

        with patch.object(player, "_sample_budget", return_value=1), patch.object(
            player, "_solve_sampled_round", return_value=strategy
        ) as solve_sampled_round:
            chosen_card = player.action(hand, history)

        solve_sampled_round.assert_called_once()
        self.assertEqual(chosen_card, preferred_card)

    def test_action_skips_cfr_solver_when_hand_exceeds_default_limit(self):
        player = FullCFRPlayer(0)
        players = [player, RandomPlayer(1), RandomPlayer(2), RandomPlayer(3)]
        engine = Engine({"n_players": 4, "n_rounds": 10, "seed": 11}, players)

        hand = engine.hands[0].copy()
        history = {
            "board": [row.copy() for row in engine.board],
            "scores": engine.scores.copy(),
            "round": engine.round,
            "history_matrix": [row.copy() for row in engine.history_matrix],
            "board_history": [[row.copy() for row in board] for board in engine.board_history],
            "score_history": [scores.copy() for scores in engine.score_history],
        }
        preferred_card = hand[-1]
        heuristic_scores = {card: 0.0 for card in hand}
        heuristic_scores[preferred_card] = 1.0

        self.assertEqual(player._resolve_search_budget(len(hand), len(history["scores"])), (0, 0))

        with patch.object(player, "_heuristic_scores", return_value=heuristic_scores), patch.object(
            player, "_solve_sampled_round"
        ) as solve_sampled_round:
            chosen_card = player.action(hand, history)

        solve_sampled_round.assert_not_called()
        self.assertEqual(chosen_card, preferred_card)
if __name__ == "__main__":
    unittest.main()
