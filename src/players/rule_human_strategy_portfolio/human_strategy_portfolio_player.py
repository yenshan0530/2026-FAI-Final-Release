from src.players.rule_based_player_base import RuleBasedPlayerBase


class HumanStrategyPortfolioPlayer(RuleBasedPlayerBase):
    """Focused portfolio over the strongest rule-based modes from eval."""

    def __init__(
        self,
        player_idx,
        n_cards=104,
        board_size_x=5,
        trailing_pressure=0.55,
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

        if trailing:
            value = 0.65 * self._lpra_trailing_value(features, danger_after)
            value += 0.35 * self._hfb_value(features, context["phase"])
            value += 0.60 * safe_gain
            tie_break = (danger_after, features.hand_rank_pct, features.card)
        else:
            value = self._hfb_value(features, context["phase"])
            value += 0.35 * safe_gain
            value += 0.12 * danger_drop
            if context["phase"] == "early":
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
                self._crs_override_value(features, safe_gain, danger_drop),
            )

        return (value, 1 if override_active else 0, trailing, safe_gain, *tie_break)

    def _hfb_value(self, features, phase):
        value = self._conservative_value(features)

        if phase == "early":
            value += 7.0 * features.hand_rank_pct
        elif phase == "mid":
            value += 3.0 * features.hand_rank_pct
        else:
            value += 0.5 * features.hand_rank_pct

        if features.target_row_idx != -1 and features.immediate_penalty == 0:
            value += 0.8 * features.row_score
        if features.outlier_flag:
            value += 2.0

        if (
            phase == "early"
            and features.hand_rank_pct < 0.35
            and features.target_row_idx == -1
            and features.low_reset_cost <= 3
        ):
            value -= 4.0

        value += 0.45 * features.played_card_score
        return value

    def _lpra_trailing_value(self, features, board_danger_after):
        value = self._conservative_value(features)
        value += 0.3 * board_danger_after
        value += 0.9 * features.played_card_score
        value += 1.5 * features.hand_rank_pct
        if features.target_row_idx != -1 and features.immediate_penalty == 0:
            value += 2.0 if features.dangerous_row else 0.0
        return value

    def _crs_override_value(self, features, safe_gain, danger_drop):
        value = self._conservative_value(features)
        value += 8.0
        value += 4.0 * safe_gain
        value += 0.8 * danger_drop
        return value
