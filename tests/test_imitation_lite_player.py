from src.engine import Engine
from src.game_utils import load_players
from src.players.TA.random_player import RandomPlayer
from src.players.imitation_lite import ImitationLitePlayer, ImitationPlayer


def _history(board, scores=None, round_idx=0):
    return {
        "board": [row[:] for row in board],
        "scores": list(scores or [0, 0, 0, 0]),
        "round": round_idx,
        "history_matrix": [],
        "board_history": [],
        "score_history": [],
    }


def test_load_players_imports_imitation_lite_package():
    config = {
        "players": [
            {"path": "src.players.imitation_lite", "class": "ImitationLitePlayer"},
        ]
    }

    classes = load_players(config)

    assert [cls.__name__ for cls in classes] == ["ImitationLitePlayer"]


def test_imitation_player_alias_points_to_main_class():
    assert ImitationPlayer is ImitationLitePlayer


def test_imitation_lite_keeps_aggressive_opening_bias():
    player = ImitationLitePlayer(player_idx=0)
    hand = [6, 12, 61, 87]
    history = _history([[10], [22, 33, 44], [59], [84]], scores=[0, 0, 0, 0], round_idx=0)

    assert player.action(hand, history) == 87


def test_imitation_lite_switches_when_trailing():
    player = ImitationLitePlayer(player_idx=0)
    hand = [18, 55, 84]
    board = [[17], [44, 45, 46, 54], [80], [95]]
    history = _history(board, scores=[15, 2, 3, 4])

    assert player.action(hand, history) == 55


def test_imitation_lite_allows_clear_cheap_reset_override():
    player = ImitationLitePlayer(player_idx=0)
    hand = [9, 10, 11, 12]
    history = _history([[13], [44], [57], [90]], scores=[0, 0, 0, 0])

    assert player.action(hand, history) == 9


def test_imitation_lite_chooses_valid_card_from_engine_state():
    player = ImitationLitePlayer(player_idx=0)
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

    chosen_card = player.action(hand, history)

    assert chosen_card in hand
