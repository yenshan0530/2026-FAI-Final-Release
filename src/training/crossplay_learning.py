from __future__ import annotations

import importlib
import itertools
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.engine import Engine
from src.players.rule_based_player_base import RuleBasedPlayerBase


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "results" / "training" / "crossplay_examples.jsonl"
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "src" / "players" / "crossplay_imitation" / "crossplay_model.json"
)
DEFAULT_TEACHER_LABELS = ("CFR", "HSP", "HFB", "AZL", "LPRA")
DEFAULT_TRAILING_THRESHOLD = 0.55
MODEL_FEATURE_NAMES = (
    "bias",
    "card_value_pct",
    "target_row_exists",
    "target_row_end_pct",
    "row_score",
    "row_len",
    "gap",
    "open_slots_before_take",
    "intervening_count",
    "low_reset_cost",
    "played_card_score",
    "dangerous_row",
    "hand_rank_pct",
    "run_middle_flag",
    "outlier_flag",
    "score_pressure",
    "immediate_penalty",
    "take_risk",
    "safe_after",
    "safe_gain",
    "danger_after",
    "danger_drop",
    "phase_early",
    "phase_mid",
    "phase_late",
    "trailing_flag",
    "opening_round_flag",
    "baseline_safe_future_count",
    "baseline_board_danger",
)
PHASE_KEYS = ("early", "mid", "late")

PLAYER_SPECS = {
    "CFR": {
        "path": "src.players.cfr",
        "class": "FullCFRPlayer",
        "args": {
            "search_time_limit": 0.12,
            "max_samples": 2,
            "base_iterations": 4,
            "endgame_iterations": 10,
        },
    },
    "HSP": {
        "path": "src.players.rule_human_strategy_portfolio",
        "class": "HumanStrategyPortfolioPlayer",
        "args": {},
    },
    "HFB": {
        "path": "src.players.rule_high_first_blocker",
        "class": "HighFirstBlockerPlayer",
        "args": {},
    },
    "AZL": {
        "path": "src.players.alpha_zero_lite",
        "class": "AlphaZeroLitePlayer",
        "args": {
            "search_time_limit": 0.12,
            "max_samples": 1,
            "simulations_per_sample": 8,
            "max_depth": 1,
        },
    },
    "LPRA": {
        "path": "src.players.rule_leader_punish_rank_aware",
        "class": "LeaderPunishRankAwarePlayer",
        "args": {},
    },
    "LFS": {
        "path": "src.players.rule_low_first_safe_shed",
        "class": "LowFirstSafeShedPlayer",
        "args": {},
    },
    "CFC": {
        "path": "src.players.rule_closest_fit_conservative",
        "class": "ClosestFitConservativePlayer",
        "args": {},
    },
    "CRS": {
        "path": "src.players.rule_controlled_reset_sacrifice",
        "class": "ControlledResetSacrificePlayer",
        "args": {},
    },
    "IMIT": {
        "path": "src.players.imitation_lite",
        "class": "ImitationLitePlayer",
        "args": {},
    },
    "RND": {
        "path": "src.players.TA.random_player",
        "class": "RandomPlayer",
        "args": {},
    },
    "B4": {
        "path": "src.players.TA.public_baselines1",
        "class": "Baseline4",
        "args": {},
    },
    "B5": {
        "path": "src.players.TA.public_baselines1",
        "class": "Baseline5",
        "args": {},
    },
}
AVAILABLE_CROSSPLAY_LABELS = tuple(sorted(PLAYER_SPECS))


class CrossPlayFeatureEncoder(RuleBasedPlayerBase):
    """Encodes live decisions into shared candidate features for offline learning."""

    def _score_candidate(self, features, hand, history, context):
        raise NotImplementedError("CrossPlayFeatureEncoder only provides feature encoding.")

    def encode_decision(self, hand, history):
        context = self._build_context(hand, history)
        return {
            "phase": context["phase"],
            "baseline_safe_future_count": context["baseline_safe_future_count"],
            "baseline_board_danger": context["baseline_board_danger"],
            "candidates": [
                self._encode_candidate(card, hand, history, context) for card in hand
            ],
        }

    def _encode_candidate(self, card, hand, history, context):
        features = self._candidate_features(
            card=card,
            hand=hand,
            board=context["board"],
            unseen_set=context["unseen_set"],
            scores=context["scores"],
        )
        board_after, penalty = self._simulate_our_placement(context["board"], features.card)
        remaining_hand = [other_card for other_card in hand if other_card != features.card]
        safe_after = self._future_safe_count(
            board_after,
            remaining_hand,
            context["unseen_set"],
            context["scores"],
        )
        danger_after = self._board_danger_score(board_after)

        feature_values = {
            "bias": 1.0,
            "card_value_pct": features.card / float(self.n_cards),
            "target_row_exists": 0.0 if features.target_row_idx == -1 else 1.0,
            "target_row_end_pct": (
                0.0
                if features.target_row_end < 0
                else features.target_row_end / float(self.n_cards)
            ),
            "row_score": float(features.row_score),
            "row_len": float(features.row_len),
            "gap": float(features.gap),
            "open_slots_before_take": float(features.open_slots_before_take),
            "intervening_count": float(features.intervening_count),
            "low_reset_cost": float(features.low_reset_cost),
            "played_card_score": float(features.played_card_score),
            "dangerous_row": float(features.dangerous_row),
            "hand_rank_pct": float(features.hand_rank_pct),
            "run_middle_flag": float(features.run_middle_flag),
            "outlier_flag": float(features.outlier_flag),
            "score_pressure": float(features.score_pressure),
            "immediate_penalty": float(features.immediate_penalty),
            "take_risk": float(features.take_risk),
            "safe_after": float(safe_after),
            "safe_gain": float(safe_after - context["baseline_safe_future_count"]),
            "danger_after": float(danger_after),
            "danger_drop": float(context["baseline_board_danger"] - danger_after),
            "phase_early": 1.0 if context["phase"] == "early" else 0.0,
            "phase_mid": 1.0 if context["phase"] == "mid" else 0.0,
            "phase_late": 1.0 if context["phase"] == "late" else 0.0,
            "trailing_flag": (
                1.0 if features.score_pressure >= DEFAULT_TRAILING_THRESHOLD else 0.0
            ),
            "opening_round_flag": 1.0 if history.get("round", 0) <= 1 else 0.0,
            "baseline_safe_future_count": float(context["baseline_safe_future_count"]),
            "baseline_board_danger": float(context["baseline_board_danger"]),
        }

        return {
            "card": features.card,
            "feature_values": feature_values,
            "immediate_penalty": float(penalty),
            "target_row_idx": features.target_row_idx,
        }


class _RecordingPlayer:
    def __init__(
        self,
        label,
        inner_player,
        encoder,
        player_idx,
        lineup,
        seed,
        sink,
        example_weight,
    ):
        self.label = label
        self._inner_player = inner_player
        self._encoder = encoder
        self._player_idx = player_idx
        self._lineup = tuple(lineup)
        self._seed = seed
        self._sink = sink
        self._example_weight = example_weight

    def action(self, hand, history):
        hand_snapshot = list(hand)
        history_snapshot = deepcopy(history)
        chosen_card = self._inner_player.action(hand, history)

        if len(hand_snapshot) > 1 and chosen_card in hand_snapshot:
            encoded = self._encoder.encode_decision(hand_snapshot, history_snapshot)
            self._sink.append(
                {
                    "teacher": self.label,
                    "lineup": list(self._lineup),
                    "seed": self._seed,
                    "player_idx": self._player_idx,
                    "round": history_snapshot.get("round", 0),
                    "hand": hand_snapshot,
                    "board": [list(row) for row in history_snapshot["board"]],
                    "scores": list(history_snapshot["scores"]),
                    "chosen_card": chosen_card,
                    "candidates": encoded["candidates"],
                    "example_weight": self._example_weight,
                }
            )

        return chosen_card


def build_crossplay_lineups(
    labels=DEFAULT_TEACHER_LABELS,
    n_players=4,
    include_rotations=False,
    max_lineups=None,
    lineup_seed=0,
):
    labels = tuple(labels)
    if len(labels) < n_players:
        raise ValueError(
            f"Need at least {n_players} teacher labels, received {len(labels)}."
        )
    if max_lineups is not None and max_lineups <= 0:
        raise ValueError("max_lineups must be positive when provided.")

    lineups = [tuple(lineup) for lineup in itertools.combinations(labels, n_players)]
    if not include_rotations:
        selected_lineups = lineups
    else:
        rotated_lineups = []
        for lineup in lineups:
            for shift in range(n_players):
                rotated_lineups.append(
                    tuple(
                        lineup[(offset + shift) % n_players]
                        for offset in range(n_players)
                    )
                )
        selected_lineups = rotated_lineups

    if max_lineups is None or len(selected_lineups) <= max_lineups:
        return selected_lineups

    shuffled = list(selected_lineups)
    random.Random(lineup_seed).shuffle(shuffled)
    return shuffled[:max_lineups]


def _instantiate_teacher(label, player_idx, teacher_args=None):
    if label not in PLAYER_SPECS:
        raise KeyError(f"Unknown teacher label: {label}")

    spec = PLAYER_SPECS[label]
    module = importlib.import_module(spec["path"])
    cls = getattr(module, spec["class"])
    kwargs = dict(spec["args"])
    if teacher_args and label in teacher_args:
        kwargs.update(teacher_args[label])
    return cls(player_idx=player_idx, **kwargs)


def _average_ranks(scores):
    indexed_scores = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    cursor = 0

    while cursor < len(indexed_scores):
        next_cursor = cursor + 1
        while (
            next_cursor < len(indexed_scores)
            and indexed_scores[next_cursor][1] == indexed_scores[cursor][1]
        ):
            next_cursor += 1

        avg_rank = (cursor + 1 + next_cursor) / 2.0
        for idx in range(cursor, next_cursor):
            player_idx, _ = indexed_scores[idx]
            ranks[player_idx] = avg_rank
        cursor = next_cursor

    return ranks


def generate_crossplay_examples(
    games_per_lineup=1,
    base_seed=11,
    include_rotations=False,
    teacher_labels=DEFAULT_TEACHER_LABELS,
    teacher_weights=None,
    teacher_args=None,
    engine_cfg=None,
    max_lineups=None,
    lineup_seed=None,
):
    engine_cfg = dict(engine_cfg or {})
    n_players = engine_cfg.get("n_players", 4)
    if games_per_lineup <= 0:
        raise ValueError("games_per_lineup must be positive.")
    teacher_weights = dict(teacher_weights or {})

    lineups = build_crossplay_lineups(
        labels=teacher_labels,
        n_players=n_players,
        include_rotations=include_rotations,
        max_lineups=max_lineups,
        lineup_seed=base_seed if lineup_seed is None else lineup_seed,
    )
    examples = []

    for lineup_idx, lineup in enumerate(lineups):
        for game_offset in range(games_per_lineup):
            seed = base_seed + lineup_idx * games_per_lineup + game_offset
            game_examples = []
            players = []
            for player_idx, label in enumerate(lineup):
                inner_player = _instantiate_teacher(label, player_idx, teacher_args=teacher_args)
                encoder = CrossPlayFeatureEncoder(
                    player_idx=player_idx,
                    n_cards=engine_cfg.get("n_cards", 104),
                    board_size_x=engine_cfg.get("board_size_x", 5),
                )
                players.append(
                    _RecordingPlayer(
                        label=label,
                        inner_player=inner_player,
                        encoder=encoder,
                        player_idx=player_idx,
                        lineup=lineup,
                        seed=seed,
                        sink=game_examples,
                        example_weight=float(teacher_weights.get(label, 1.0)),
                    )
                )

            cfg = {
                "n_players": n_players,
                "n_rounds": engine_cfg.get("n_rounds", 10),
                "board_size_x": engine_cfg.get("board_size_x", 5),
                "board_size_y": engine_cfg.get("board_size_y", 4),
                "timeout": engine_cfg.get("timeout"),
                "timeout_buffer": engine_cfg.get("timeout_buffer", 0.5),
                "verbose": engine_cfg.get("verbose", False),
            }
            cfg.update(engine_cfg)
            cfg["seed"] = seed

            engine = Engine(cfg, players)
            scores, full_history = engine.play_game()
            ranks = _average_ranks(scores)

            for example in game_examples:
                player_idx = example["player_idx"]
                forced_action_count = sum(
                    1
                    for round_flags in full_history.get("flags_matrix", [])
                    if round_flags[player_idx]
                )
                example["final_score"] = scores[player_idx]
                example["final_rank"] = ranks[player_idx]
                example["forced_action_count"] = forced_action_count

            examples.extend(game_examples)

    return examples


def save_crossplay_examples(examples, output_path=DEFAULT_DATASET_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True))
            handle.write("\n")


def load_crossplay_examples(dataset_path=DEFAULT_DATASET_PATH):
    dataset_path = Path(dataset_path)
    examples = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                examples.append(json.loads(stripped))
    return examples


def _candidate_vector(candidate):
    feature_values = candidate["feature_values"]
    return np.array(
        [float(feature_values.get(name, 0.0)) for name in MODEL_FEATURE_NAMES],
        dtype=np.float64,
    )


def _phase_key_from_feature_values(feature_values):
    if float(feature_values.get("phase_early", 0.0)) >= 0.5:
        return "early"
    if float(feature_values.get("phase_mid", 0.0)) >= 0.5:
        return "mid"
    return "late"


def _prepare_training_examples(examples):
    prepared = []
    for example in examples:
        candidates = example.get("candidates", [])
        if len(candidates) <= 1:
            continue

        chosen_card = example.get("chosen_card")
        cards = [candidate["card"] for candidate in candidates]
        if chosen_card not in cards:
            continue

        sample_weight = 1.0 / float(example.get("final_rank", 1.0))
        sample_weight *= float(example.get("example_weight", 1.0))
        if sample_weight <= 0.0:
            continue
        matrix = np.vstack([_candidate_vector(candidate) for candidate in candidates])
        phase_key = _phase_key_from_feature_values(candidates[0]["feature_values"])
        prepared.append(
            {
                "matrix": matrix,
                "target": cards.index(chosen_card),
                "weight": sample_weight,
                "phase_key": phase_key,
            }
        )
    return prepared


def _phase_weight_vector(model_data, feature_values):
    weights_by_phase = model_data.get("weights_by_phase")
    if not weights_by_phase:
        return model_data["weights"]

    phase_key = _phase_key_from_feature_values(feature_values)
    if phase_key in weights_by_phase:
        return weights_by_phase[phase_key]
    if "late" in weights_by_phase:
        return weights_by_phase["late"]
    return next(iter(weights_by_phase.values()))


def _softmax(logits):
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def train_linear_policy(
    examples,
    epochs=180,
    learning_rate=0.05,
    l2=1e-4,
):
    prepared = _prepare_training_examples(examples)
    if not prepared:
        raise ValueError("No non-trivial training examples were available.")

    all_candidates = np.vstack([item["matrix"] for item in prepared])
    feature_means = all_candidates.mean(axis=0)
    feature_scales = all_candidates.std(axis=0)
    feature_scales[feature_scales < 1e-6] = 1.0

    normalized = [
        {
            "matrix": (item["matrix"] - feature_means) / feature_scales,
            "target": item["target"],
            "weight": item["weight"],
            "phase_key": item["phase_key"],
        }
        for item in prepared
    ]
    phase_counts = {
        phase_key: sum(1 for item in normalized if item["phase_key"] == phase_key)
        for phase_key in PHASE_KEYS
    }

    weights_by_phase = {
        phase_key: np.zeros(len(MODEL_FEATURE_NAMES), dtype=np.float64)
        for phase_key in PHASE_KEYS
    }
    best_snapshot = None

    total_weight = sum(item["weight"] for item in normalized)
    for epoch_idx in range(epochs):
        gradients = {
            phase_key: np.zeros(len(MODEL_FEATURE_NAMES), dtype=np.float64)
            for phase_key in PHASE_KEYS
        }
        weighted_loss = 0.0
        weighted_correct = 0.0
        phase_weights = {phase_key: 0.0 for phase_key in PHASE_KEYS}
        phase_correct = {phase_key: 0.0 for phase_key in PHASE_KEYS}
        phase_loss = {phase_key: 0.0 for phase_key in PHASE_KEYS}

        for item in normalized:
            phase_key = item["phase_key"]
            weights = weights_by_phase[phase_key]
            logits = item["matrix"] @ weights
            probabilities = _softmax(logits)
            diff = probabilities.copy()
            diff[item["target"]] -= 1.0
            gradients[phase_key] += item["weight"] * (item["matrix"].T @ diff)
            loss_term = item["weight"] * math.log(
                float(probabilities[item["target"]]) + 1e-12
            )
            weighted_loss -= loss_term
            phase_loss[phase_key] -= loss_term
            phase_weights[phase_key] += item["weight"]
            if int(np.argmax(logits)) == item["target"]:
                weighted_correct += item["weight"]
                phase_correct[phase_key] += item["weight"]

        regularization = 0.0
        for phase_key in PHASE_KEYS:
            phase_total_weight = phase_weights[phase_key]
            weights = weights_by_phase[phase_key]
            if phase_total_weight > 0.0:
                gradients[phase_key] /= phase_total_weight
            gradients[phase_key] += l2 * weights
            weights_by_phase[phase_key] -= learning_rate * gradients[phase_key]
            regularization += 0.5 * l2 * float(np.dot(weights_by_phase[phase_key], weights_by_phase[phase_key]))

        avg_loss = (weighted_loss / total_weight) + regularization
        avg_accuracy = weighted_correct / total_weight
        phase_summaries = {}
        for phase_key in PHASE_KEYS:
            phase_total_weight = phase_weights[phase_key]
            if phase_total_weight <= 0.0:
                phase_summaries[phase_key] = {
                    "num_decisions": 0,
                    "total_example_weight": 0.0,
                    "train_accuracy": None,
                    "train_loss": None,
                }
                continue
            phase_summaries[phase_key] = {
                "num_decisions": int(phase_counts[phase_key]),
                "total_example_weight": phase_total_weight,
                "train_accuracy": phase_correct[phase_key] / phase_total_weight,
                "train_loss": phase_loss[phase_key] / phase_total_weight,
            }
        snapshot = {
            "epoch": epoch_idx + 1,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "weights_by_phase": {
                phase_key: weights_by_phase[phase_key].copy()
                for phase_key in PHASE_KEYS
            },
            "phase_summaries": phase_summaries,
        }
        if best_snapshot is None or (
            snapshot["accuracy"],
            -snapshot["loss"],
        ) > (
            best_snapshot["accuracy"],
            -best_snapshot["loss"],
        ):
            best_snapshot = snapshot

    teacher_labels = sorted(
        {example["teacher"] for example in examples if "teacher" in example}
    )
    return {
        "model_type": "phase_linear_policy",
        "phase_keys": list(PHASE_KEYS),
        "feature_names": list(MODEL_FEATURE_NAMES),
        "feature_means": feature_means.tolist(),
        "feature_scales": feature_scales.tolist(),
        "weights_by_phase": {
            phase_key: best_snapshot["weights_by_phase"][phase_key].tolist()
            for phase_key in PHASE_KEYS
        },
        "training_summary": {
            "epochs": epochs,
            "best_epoch": best_snapshot["epoch"],
            "learning_rate": learning_rate,
            "l2": l2,
            "num_decisions": len(prepared),
            "num_candidates": int(sum(item["matrix"].shape[0] for item in prepared)),
            "total_example_weight": float(sum(item["weight"] for item in normalized)),
            "train_accuracy": best_snapshot["accuracy"],
            "train_loss": best_snapshot["loss"],
            "teacher_labels": teacher_labels,
            "phase_summaries": best_snapshot["phase_summaries"],
        },
    }


def save_model(model_data, output_path=DEFAULT_MODEL_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(model_data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_model(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    with model_path.open("r", encoding="utf-8") as handle:
        model_data = json.load(handle)

    required_keys = {"feature_names", "feature_means", "feature_scales"}
    missing = required_keys.difference(model_data)
    if missing:
        raise ValueError(f"Model file is missing required keys: {sorted(missing)}")
    if "weights" not in model_data and "weights_by_phase" not in model_data:
        raise ValueError("Model file must contain 'weights' or 'weights_by_phase'.")
    if "weights_by_phase" in model_data and not isinstance(model_data["weights_by_phase"], dict):
        raise ValueError("'weights_by_phase' must be a mapping when provided.")
    return model_data


def score_feature_values(model_data, feature_values):
    feature_names = model_data["feature_names"]
    means = model_data["feature_means"]
    scales = model_data["feature_scales"]
    weights = _phase_weight_vector(model_data, feature_values)
    total = 0.0

    for name, mean, scale, weight in zip(feature_names, means, scales, weights):
        raw_value = float(feature_values.get(name, 0.0))
        effective_scale = scale if abs(scale) > 1e-9 else 1.0
        total += float(weight) * ((raw_value - float(mean)) / float(effective_scale))

    return total
