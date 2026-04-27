from src.players.rule_based_player_base import RuleBasedPlayerBase


class ClosestFitConservativePlayer(RuleBasedPlayerBase):
    def _score_candidate(self, features, hand, history, context):
        value = self._conservative_value(features)
        value -= float(features.gap)
        value -= 4.0 * max(0, features.row_len - 2)
        value -= 3.0 * features.intervening_count
        value -= 2.0 * features.row_score
        value += 6.0 if features.target_row_idx != -1 and features.row_len <= 2 else 0.0
        value -= 3.0 if features.dangerous_row else 0.0
        if features.target_row_idx == -1 and features.low_reset_cost <= 2:
            value += 1.0

        return (value, -features.gap, -features.card)
