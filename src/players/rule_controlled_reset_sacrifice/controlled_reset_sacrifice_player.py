from src.players.rule_based_player_base import RuleBasedPlayerBase


class ControlledResetSacrificePlayer(RuleBasedPlayerBase):
    def _score_candidate(self, features, hand, history, context):
        board_after, penalty = self._simulate_our_placement(context["board"], features.card)
        remaining_hand = [card for card in hand if card != features.card]
        safe_after = self._future_safe_count(
            board_after,
            remaining_hand,
            context["unseen_set"],
            context["scores"],
        )
        improvement = safe_after - context["baseline_safe_future_count"]
        danger_drop = context["baseline_board_danger"] - self._board_danger_score(board_after)
        cheap_reset_threshold = 3 if features.score_pressure < 0.5 else 5

        value = self._conservative_value(features)
        if penalty > 0 and penalty <= cheap_reset_threshold:
            value += 8.0
            value += 4.0 * improvement
            value += 0.8 * danger_drop
        elif penalty > 0:
            value -= 5.0
        else:
            value += 1.5 * improvement

        return (value, safe_after, -features.card)
