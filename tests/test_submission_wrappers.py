import unittest

from best_player1 import BestPlayer1
from best_player2 import BestPlayer2


class SubmissionWrapperTests(unittest.TestCase):
    def test_wrappers_return_legal_actions(self):
        history = {
            "board": [[12], [27], [44], [70]],
            "scores": [0, 3, 6, 9],
            "round": 0,
            "history_matrix": [],
            "board_history": [],
        }
        hand = [13, 31, 46, 55, 68, 74, 81, 88, 96, 103]

        for cls in (BestPlayer1, BestPlayer2):
            player = cls(player_idx=0)
            action = player.action(hand[:], history)
            self.assertIn(action, hand)


if __name__ == "__main__":
    unittest.main()
