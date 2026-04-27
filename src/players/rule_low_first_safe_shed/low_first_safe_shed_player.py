from src.players.rule_based_player_base import RuleBasedPlayerBase


class LowFirstSafeShedPlayer(RuleBasedPlayerBase):
    def _score_candidate(self, features, hand, history, context):
        phase = context["phase"]
        safe_enough = (
            features.target_row_idx != -1
            and features.row_len <= 3
            and features.intervening_count < max(1, features.open_slots_before_take)
        ) or (features.target_row_idx == -1 and features.low_reset_cost <= 3)

        value = self._conservative_value(features)
        if phase == "early":
            value += 5.0 * (1.0 - features.hand_rank_pct)
        elif phase == "mid":
            value += 2.5 * (1.0 - features.hand_rank_pct)
        else:
            value += 0.5 * (1.0 - features.hand_rank_pct)

        value += 8.0 if safe_enough else -4.0
        value += 1.5 if features.run_middle_flag else 0.0
        value += 0.4 * features.played_card_score

        return (value, features.run_middle_flag, -features.card)
