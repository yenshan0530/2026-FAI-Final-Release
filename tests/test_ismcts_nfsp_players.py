from src.engine import Engine
from src.game_utils import load_players
from src.players.ISMCTS import ISMCTSPlayer
from src.players.ISMCTS.ismcts_player import _DeterminizedState
from src.players.NFSP import NFSPPlayer
from src.players.TA.random_player import RandomPlayer


def _history(board, scores=None, round_idx=0, history_matrix=None):
    return {
        "board": [row[:] for row in board],
        "scores": list(scores or [0, 0, 0, 0]),
        "round": round_idx,
        "history_matrix": [row[:] for row in (history_matrix or [])],
        "board_history": [],
        "score_history": [],
    }


def test_load_players_imports_ismcts_and_nfsp_packages():
    config = {
        "players": [
            {"path": "src.players.ISMCTS", "class": "ISMCTSPlayer"},
            {"path": "src.players.NFSP", "class": "NFSPPlayer"},
        ]
    }

    classes = load_players(config)

    assert [cls.__name__ for cls in classes] == ["ISMCTSPlayer", "NFSPPlayer"]


def test_ismcts_uses_exact_three_card_endgame_values():
    player = ISMCTSPlayer(player_idx=0)
    sampled_hands = (
        (7, 53, 79),
        (14, 37, 43),
        (13, 17, 69),
        (10, 25, 63),
    )
    history = {
        "board": [[1, 91, 94], [3], [2, 67], [89]],
        "scores": [1, 1, 8, 7],
        "round": 7,
        "history_matrix": [],
        "board_history": [],
        "score_history": [],
    }
    root_state = _DeterminizedState(
        board=tuple(tuple(row) for row in history["board"]),
        scores=tuple(history["scores"]),
        hands=sampled_hands,
        round_idx=history["round"],
    )

    exact_values = player._exact_action_values(root_state)
    best_card = max(sampled_hands[0], key=lambda card: exact_values[card])

    player._sample_hidden_hands = (
        lambda hand, unseen_cards, n_players, rng: [list(other_hand) for other_hand in sampled_hands]
    )
    player._state_seed = lambda hand, history: 0

    assert exact_values[best_card] > min(exact_values.values())
    assert player.action(list(sampled_hands[0]), history) == best_card


def test_nfsp_shifts_action_under_score_pressure():
    player = NFSPPlayer(player_idx=0)
    hand = [47, 93, 101]
    board = [[2, 31, 61, 70], [12, 64, 83, 86], [15], [23]]

    leading_history = _history(
        board,
        scores=[1, 5, 6, 7],
        round_idx=1,
        history_matrix=[[47, 30, 70, 100]],
    )
    trailing_history = _history(
        board,
        scores=[15, 2, 3, 4],
        round_idx=1,
        history_matrix=[[47, 30, 70, 100]],
    )

    assert player.action(hand, leading_history) == 93
    assert player.action(hand, trailing_history) == 47


def test_new_players_choose_valid_cards_from_engine_state():
    player_classes = [ISMCTSPlayer, NFSPPlayer]

    for player_class in player_classes:
        player = player_class(player_idx=0)
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
