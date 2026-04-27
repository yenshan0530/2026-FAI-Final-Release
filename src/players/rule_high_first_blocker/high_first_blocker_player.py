from src.players.rule_based_player_base import RuleBasedPlayerBase


class HighFirstBlockerPlayer(RuleBasedPlayerBase):
    def _score_candidate(self, features, hand, history, context):
        phase = context["phase"]
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

        return (value, features.card)
