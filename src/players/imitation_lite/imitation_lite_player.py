from src.players.rule_based_player_base import RuleBasedPlayerBase


class ImitationLitePlayer(RuleBasedPlayerBase):
    """Lightweight teacher-distilled policy over the strongest local agents."""

    def __init__(
        self,
        player_idx,
        n_cards=104,
        board_size_x=5,
        trailing_pressure=0.54,
        cheap_reset_lead_threshold=2,
        cheap_reset_trail_threshold=4,
        required_safe_gain=2,
    ):
        super().__init__(player_idx, n_cards=n_cards, board_size_x=board_size_x)
        self.trailing_pressure = trailing_pressure
        self.cheap_reset_lead_threshold = cheap_reset_lead_threshold
        self.cheap_reset_trail_threshold = cheap_reset_trail_threshold
        self.required_safe_gain = required_safe_gain

    def _score_candidate(self, features, hand, history, context):
        trailing = features.score_pressure >= self.trailing_pressure
        board_after, penalty = self._simulate_our_placement(context["board"], features.card)
        remaining_hand = [card for card in hand if card != features.card]
        safe_after = self._future_safe_count(
            board_after,
            remaining_hand,
            context["unseen_set"],
            context["scores"],
        )
        safe_gain = safe_after - context["baseline_safe_future_count"]
        danger_after = self._board_danger_score(board_after)
        danger_drop = context["baseline_board_danger"] - danger_after

        hfb_value = self._hfb_value(features, context["phase"])
        lpra_value = self._lpra_value(features, danger_after)
        azl_value = self._azl_style_value(features)
        rank_value = self._rank_buffer_value(features, trailing)

        if trailing:
            value = 0.45 * lpra_value
            value += 0.25 * hfb_value
            value += 0.20 * azl_value
            value += 0.10 * rank_value
            value += 0.55 * safe_gain
            value += 0.05 * danger_after
            tie_break = (danger_after, features.hand_rank_pct, features.card)
        else:
            opening_like = context["phase"] == "early" or history.get("round", 0) <= 1
            value = 0.45 * hfb_value
            value += 0.20 * lpra_value
            value += 0.20 * azl_value
            value += 0.15 * rank_value
            value += 0.40 * safe_gain
            value += 0.12 * danger_drop
            if opening_like and features.target_row_idx != -1:
                value += 1.80 * features.hand_rank_pct
                value += 0.60 * min(3, features.intervening_count)
            if opening_like:
                tie_break = (safe_after, features.hand_rank_pct, features.card)
            elif context["phase"] == "mid":
                tie_break = (-features.gap, safe_after, -features.card)
            else:
                tie_break = (features.hand_rank_pct, danger_drop, features.card)

        cheap_reset_threshold = (
            self.cheap_reset_trail_threshold
            if trailing
            else self.cheap_reset_lead_threshold
        )
        override_active = (
            penalty > 0
            and penalty <= cheap_reset_threshold
            and (
                safe_gain >= self.required_safe_gain
                or (trailing and danger_drop >= 6.0)
            )
        )
        if override_active:
            value = max(
                value,
                self._cheap_reset_value(features, safe_gain, danger_drop),
            )

        return (value, 1 if override_active else 0, trailing, safe_gain, *tie_break)

    def _hfb_value(self, features, phase):
        value = self._conservative_value(features)

        if phase == "early":
            value += 6.0 * features.hand_rank_pct
        elif phase == "mid":
            value += 2.5 * features.hand_rank_pct
        else:
            value += 0.4 * features.hand_rank_pct

        if features.target_row_idx != -1 and features.immediate_penalty == 0:
            value += 0.75 * features.row_score
            if features.row_len <= 2:
                value += 0.6
        if features.outlier_flag:
            value += 1.5

        if (
            phase == "early"
            and features.hand_rank_pct < 0.35
            and features.target_row_idx == -1
            and features.low_reset_cost <= 3
        ):
            value -= 3.5

        value += 0.42 * features.played_card_score
        return value

    def _lpra_value(self, features, board_danger_after):
        value = self._conservative_value(features)
        value += 0.25 * board_danger_after
        value += 0.85 * features.played_card_score
        value += 1.35 * features.hand_rank_pct
        if features.target_row_idx != -1 and features.immediate_penalty == 0:
            value += 1.75 if features.dangerous_row else 0.0
        return value

    def _azl_style_value(self, features):
        score_pressure = features.score_pressure
        risk_weight = 1.65 - 0.35 * score_pressure
        dump_weight = 0.16 + 0.10 * score_pressure
        low_card_weight = 0.70 - 0.20 * score_pressure

        value = -9.5 * features.immediate_penalty
        value -= risk_weight * features.take_risk * max(1, features.row_score)
        value -= 0.035 * max(0, features.gap)
        value -= 0.42 * max(0, features.row_len - 3)
        value += dump_weight * features.played_card_score
        value -= 0.001 * features.card

        if features.target_row_idx == -1:
            value -= low_card_weight * max(1, features.low_reset_cost)

        return value

    def _rank_buffer_value(self, features, trailing):
        value = self._conservative_value(features)
        if trailing:
            value += 0.55 * features.hand_rank_pct
            value += 0.20 * features.played_card_score
        else:
            value += 2.25 * (1.0 - features.take_risk)
            if features.target_row_idx != -1 and features.row_len <= 2 and features.immediate_penalty == 0:
                value += 1.0
        return value

    def _cheap_reset_value(self, features, safe_gain, danger_drop):
        value = self._conservative_value(features)
        value += 8.0
        value += 4.0 * safe_gain
        value += 0.8 * danger_drop
        return value


ImitationPlayer = ImitationLitePlayer
