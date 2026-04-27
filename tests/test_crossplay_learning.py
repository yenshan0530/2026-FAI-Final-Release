from src.engine import Engine
from src.game_utils import load_players
from src.players.TA.random_player import RandomPlayer
from src.players.crossplay_imitation import CrossPlayImitationPlayer, CrossPlayPlayer
from src.training.crossplay_learning import (
    DEFAULT_TEACHER_LABELS,
    MODEL_FEATURE_NAMES,
    build_crossplay_lineups,
    generate_crossplay_examples,
    score_feature_values,
    train_linear_policy,
)


def _empty_feature_values():
    return {name: 0.0 for name in MODEL_FEATURE_NAMES}


def _candidate(card, **overrides):
    feature_values = _empty_feature_values()
    feature_values["bias"] = 1.0
    feature_values.update(overrides)
    return {"card": card, "feature_values": feature_values}


def test_load_players_imports_crossplay_imitation_package():
    config = {
        "players": [
            {
                "path": "src.players.crossplay_imitation",
                "class": "CrossPlayImitationPlayer",
            }
        ]
    }

    classes = load_players(config)

    assert [cls.__name__ for cls in classes] == ["CrossPlayImitationPlayer"]


def test_crossplay_player_alias_points_to_main_class():
    assert CrossPlayPlayer is CrossPlayImitationPlayer


def test_build_crossplay_lineups_covers_all_requested_teachers():
    lineups = build_crossplay_lineups(include_rotations=False)

    assert len(lineups) == 5
    assert set().union(*map(set, lineups)) == set(DEFAULT_TEACHER_LABELS)


def test_build_crossplay_lineups_can_cap_lineups_deterministically():
    labels = ("CFR", "HSP", "HFB", "AZL", "LPRA", "LFS")

    lineups_a = build_crossplay_lineups(
        labels=labels,
        include_rotations=False,
        max_lineups=3,
        lineup_seed=17,
    )
    lineups_b = build_crossplay_lineups(
        labels=labels,
        include_rotations=False,
        max_lineups=3,
        lineup_seed=17,
    )

    assert len(lineups_a) == 3
    assert lineups_a == lineups_b


def test_generate_crossplay_examples_records_teacher_pool():
    examples = generate_crossplay_examples(
        games_per_lineup=1,
        base_seed=3,
        include_rotations=False,
        engine_cfg={"n_players": 4, "n_rounds": 2},
        teacher_args={
            "CFR": {
                "search_time_limit": 0.01,
                "max_samples": 1,
                "base_iterations": 2,
                "endgame_iterations": 4,
            },
            "AZL": {
                "search_time_limit": 0.01,
                "max_samples": 1,
                "simulations_per_sample": 4,
                "max_depth": 1,
            },
        },
    )

    assert examples
    assert {example["teacher"] for example in examples} == set(DEFAULT_TEACHER_LABELS)

    first = examples[0]
    assert first["chosen_card"] in first["hand"]
    assert any(candidate["card"] == first["chosen_card"] for candidate in first["candidates"])
    assert "final_rank" in first
    assert "final_score" in first


def test_generate_crossplay_examples_records_teacher_weights():
    examples = generate_crossplay_examples(
        games_per_lineup=1,
        base_seed=5,
        include_rotations=False,
        teacher_weights={"LPRA": 0.25, "CFR": 1.0},
        engine_cfg={"n_players": 4, "n_rounds": 2},
        teacher_args={
            "CFR": {
                "search_time_limit": 0.01,
                "max_samples": 1,
                "base_iterations": 2,
                "endgame_iterations": 4,
            },
            "AZL": {
                "search_time_limit": 0.01,
                "max_samples": 1,
                "simulations_per_sample": 4,
                "max_depth": 1,
            },
        },
    )

    example_weights = {example["teacher"]: example["example_weight"] for example in examples}

    assert example_weights["CFR"] == 1.0
    assert example_weights["LPRA"] == 0.25


def test_train_linear_policy_fits_simple_preferences():
    examples = [
        {
            "teacher": "HFB",
            "chosen_card": 41,
            "candidates": [
                _candidate(41, safe_gain=3.0, immediate_penalty=0.0),
                _candidate(55, safe_gain=0.0, immediate_penalty=5.0),
            ],
        },
        {
            "teacher": "AZL",
            "chosen_card": 73,
            "candidates": [
                _candidate(18, safe_gain=0.0, immediate_penalty=4.0),
                _candidate(73, safe_gain=2.0, immediate_penalty=0.0),
            ],
        },
        {
            "teacher": "LPRA",
            "chosen_card": 62,
            "candidates": [
                _candidate(62, safe_gain=1.5, immediate_penalty=0.0),
                _candidate(88, safe_gain=-1.0, immediate_penalty=6.0),
            ],
        },
    ]

    model_data = train_linear_policy(examples, epochs=120, learning_rate=0.1)
    summary = model_data["training_summary"]

    assert summary["num_decisions"] == 3
    assert summary["train_accuracy"] >= 0.99
    assert set(model_data["weights_by_phase"]) == {"early", "mid", "late"}
    assert len(model_data["weights_by_phase"]["late"]) == len(MODEL_FEATURE_NAMES)
    assert summary["phase_summaries"]["late"]["num_decisions"] == 3


def test_train_linear_policy_learns_phase_specific_preferences():
    safe_card_early = _candidate(
        41,
        safe_gain=3.0,
        immediate_penalty=0.0,
        phase_early=1.0,
    )
    risky_card_early = _candidate(
        55,
        safe_gain=0.0,
        immediate_penalty=5.0,
        phase_early=1.0,
    )
    safe_card_late = _candidate(
        41,
        safe_gain=3.0,
        immediate_penalty=0.0,
        phase_late=1.0,
    )
    risky_card_late = _candidate(
        55,
        safe_gain=0.0,
        immediate_penalty=5.0,
        phase_late=1.0,
    )
    examples = [
        {
            "teacher": "HFB",
            "chosen_card": 41,
            "candidates": [safe_card_early, risky_card_early],
        },
        {
            "teacher": "LPRA",
            "chosen_card": 55,
            "candidates": [safe_card_late, risky_card_late],
        },
    ]

    model_data = train_linear_policy(examples, epochs=200, learning_rate=0.1)

    early_safe_score = score_feature_values(model_data, safe_card_early["feature_values"])
    early_risky_score = score_feature_values(model_data, risky_card_early["feature_values"])
    late_safe_score = score_feature_values(model_data, safe_card_late["feature_values"])
    late_risky_score = score_feature_values(model_data, risky_card_late["feature_values"])

    assert early_safe_score > early_risky_score
    assert late_risky_score > late_safe_score
    assert model_data["training_summary"]["phase_summaries"]["early"]["num_decisions"] == 1
    assert model_data["training_summary"]["phase_summaries"]["late"]["num_decisions"] == 1


def test_train_linear_policy_honors_example_weights():
    left = _candidate(41, safe_gain=3.0, immediate_penalty=0.0)
    right = _candidate(55, safe_gain=0.0, immediate_penalty=5.0)
    examples = [
        {
            "teacher": "HFB",
            "chosen_card": 41,
            "example_weight": 0.25,
            "final_rank": 1.0,
            "candidates": [left, right],
        },
        {
            "teacher": "LFS",
            "chosen_card": 55,
            "example_weight": 4.0,
            "final_rank": 1.0,
            "candidates": [left, right],
        },
    ]

    model_data = train_linear_policy(examples, epochs=120, learning_rate=0.1)

    left_score = score_feature_values(model_data, left["feature_values"])
    right_score = score_feature_values(model_data, right["feature_values"])

    assert right_score > left_score


def test_crossplay_imitation_player_chooses_valid_card_from_engine_state():
    player = CrossPlayImitationPlayer(player_idx=0)
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
