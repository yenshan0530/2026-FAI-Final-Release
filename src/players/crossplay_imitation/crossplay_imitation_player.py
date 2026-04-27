from __future__ import annotations

from pathlib import Path

from src.players.imitation_lite import ImitationLitePlayer
from src.training.crossplay_learning import (
    CrossPlayFeatureEncoder,
    load_model,
    score_feature_values,
)


class CrossPlayImitationPlayer:
    """Offline-trained phase-aware policy over shared cross-play candidate features."""

    DEFAULT_MODEL_PATH = Path(__file__).with_name("crossplay_model.json")

    def __init__(
        self,
        player_idx,
        model_path=None,
        model_data=None,
        n_cards=104,
        board_size_x=5,
    ):
        self.player_idx = player_idx
        self._encoder = CrossPlayFeatureEncoder(
            player_idx=player_idx,
            n_cards=n_cards,
            board_size_x=board_size_x,
        )
        self._fallback_player = ImitationLitePlayer(
            player_idx=player_idx,
            n_cards=n_cards,
            board_size_x=board_size_x,
        )
        self._model = model_data
        if self._model is None:
            try:
                self._model = load_model(model_path or self.DEFAULT_MODEL_PATH)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                self._model = None

    def action(self, hand, history):
        if len(hand) <= 1:
            return hand[0]
        if self._model is None:
            return self._fallback_player.action(hand, history)

        encoded = self._encoder.encode_decision(hand, history)
        best_card = hand[0]
        best_key = None

        for candidate in encoded["candidates"]:
            feature_values = candidate["feature_values"]
            learned_score = score_feature_values(self._model, feature_values)
            key = (
                learned_score,
                feature_values["safe_gain"],
                -feature_values["immediate_penalty"],
                feature_values["danger_drop"],
                feature_values["hand_rank_pct"],
                candidate["card"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_card = candidate["card"]

        return best_card


CrossPlayPlayer = CrossPlayImitationPlayer
