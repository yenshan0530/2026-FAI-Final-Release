from src.players.alpha_zero_lite.alpha_zero_lite_player import (
    AlphaZeroLitePlayer,
    _SearchState,
)


def test_alpha_zero_lite_uses_exact_three_card_endgame_values():
    player = AlphaZeroLitePlayer(player_idx=0)
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
    root_state = _SearchState(
        board=tuple(tuple(row) for row in history["board"]),
        scores=tuple(history["scores"]),
        hands=sampled_hands,
        round_idx=history["round"],
    )

    exact_values = player._exact_action_values(root_state)

    player._sample_hidden_hands = (
        lambda hand, unseen_cards, n_players, rng: [list(other_hand) for other_hand in sampled_hands]
    )
    player._state_seed = lambda hand, history: 0

    assert exact_values[79] > exact_values[7]
    assert exact_values[79] > exact_values[53]
    assert player.action(list(sampled_hands[0]), history) == 79
