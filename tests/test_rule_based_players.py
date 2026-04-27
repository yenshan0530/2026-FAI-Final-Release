from src.engine import Engine
from src.game_utils import load_players
from src.players.TA.random_player import RandomPlayer
from src.players.rule_closest_fit_conservative import ClosestFitConservativePlayer
from src.players.rule_controlled_reset_sacrifice import ControlledResetSacrificePlayer
from src.players.rule_high_first_blocker import HighFirstBlockerPlayer
from src.players.rule_human_strategy_portfolio import HumanStrategyPortfolioPlayer
from src.players.rule_leader_punish_rank_aware import LeaderPunishRankAwarePlayer
from src.players.rule_low_first_safe_shed import LowFirstSafeShedPlayer


def _history(board, scores=None, round_idx=0):
    return {
        "board": [row[:] for row in board],
        "scores": list(scores or [0, 0, 0, 0]),
        "round": round_idx,
        "history_matrix": [],
        "board_history": [],
        "score_history": [],
    }


def test_load_players_imports_all_rule_based_packages():
    config = {
        "players": [
            {"path": "src.players.rule_low_first_safe_shed", "class": "LowFirstSafeShedPlayer"},
            {"path": "src.players.rule_closest_fit_conservative", "class": "ClosestFitConservativePlayer"},
            {"path": "src.players.rule_high_first_blocker", "class": "HighFirstBlockerPlayer"},
            {"path": "src.players.rule_controlled_reset_sacrifice", "class": "ControlledResetSacrificePlayer"},
            {"path": "src.players.rule_leader_punish_rank_aware", "class": "LeaderPunishRankAwarePlayer"},
            {"path": "src.players.rule_human_strategy_portfolio", "class": "HumanStrategyPortfolioPlayer"},
        ]
    }

    classes = load_players(config)

    assert [cls.__name__ for cls in classes] == [
        "LowFirstSafeShedPlayer",
        "ClosestFitConservativePlayer",
        "HighFirstBlockerPlayer",
        "ControlledResetSacrificePlayer",
        "LeaderPunishRankAwarePlayer",
        "HumanStrategyPortfolioPlayer",
    ]


def test_low_first_safe_shed_prefers_safe_small_card():
    player = LowFirstSafeShedPlayer(player_idx=0)
    hand = [4, 18, 41, 80]
    history = _history([[3], [17], [40], [75]])

    assert player.action(hand, history) == 4


def test_closest_fit_conservative_prefers_short_close_row():
    player = ClosestFitConservativePlayer(player_idx=0)
    hand = [18, 45, 82, 90]
    history = _history([[3, 4, 5], [17], [40, 44], [78, 79, 80]])

    assert player.action(hand, history) == 18


def test_high_first_blocker_uses_high_card_early():
    player = HighFirstBlockerPlayer(player_idx=0)
    hand = [6, 12, 61, 87]
    history = _history([[10], [22, 33, 44], [59], [84]], round_idx=0)

    assert player.action(hand, history) == 87


def test_controlled_reset_sacrifice_takes_cheap_reset_that_unlocks_future_cards():
    player = ControlledResetSacrificePlayer(player_idx=0)
    hand = [9, 10, 11, 12]
    history = _history([[13], [44], [57], [90]])

    assert player.action(hand, history) == 9


def test_leader_punish_rank_aware_changes_with_score_pressure():
    player = LeaderPunishRankAwarePlayer(player_idx=0)
    hand = [18, 55, 84]
    board = [[17], [44, 45, 46, 54], [80], [95]]

    leading_history = _history(board, scores=[1, 5, 6, 7])
    trailing_history = _history(board, scores=[15, 2, 3, 4])

    assert player.action(hand, leading_history) == 18
    assert player.action(hand, trailing_history) == 55


def test_human_strategy_portfolio_keeps_aggressive_opening_bias():
    player = HumanStrategyPortfolioPlayer(player_idx=0)
    hand = [6, 12, 61, 87]
    history = _history([[10], [22, 33, 44], [59], [84]], scores=[0, 0, 0, 0], round_idx=0)

    assert player.action(hand, history) == 87


def test_human_strategy_portfolio_switches_to_lpra_when_trailing():
    player = HumanStrategyPortfolioPlayer(player_idx=0)
    hand = [18, 55, 84]
    board = [[17], [44, 45, 46, 54], [80], [95]]
    history = _history(board, scores=[15, 2, 3, 4])

    assert player.action(hand, history) == 55


def test_human_strategy_portfolio_allows_clear_cheap_reset_override():
    player = HumanStrategyPortfolioPlayer(player_idx=0)
    hand = [9, 10, 11, 12]
    history = _history([[13], [44], [57], [90]], scores=[0, 0, 0, 0])

    assert player.action(hand, history) == 9


def test_rule_based_players_choose_valid_cards_from_engine_state():
    player_classes = [
        LowFirstSafeShedPlayer,
        ClosestFitConservativePlayer,
        HighFirstBlockerPlayer,
        ControlledResetSacrificePlayer,
        LeaderPunishRankAwarePlayer,
        HumanStrategyPortfolioPlayer,
    ]

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
