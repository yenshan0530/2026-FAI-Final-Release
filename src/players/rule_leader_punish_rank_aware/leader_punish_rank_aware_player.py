from src.players.rule_based_player_base import RuleBasedPlayerBase


class LeaderPunishRankAwarePlayer(RuleBasedPlayerBase):
    def _score_candidate(self, features, hand, history, context):
        board_after, _ = self._simulate_our_placement(context["board"], features.card)
        board_danger_after = self._board_danger_score(board_after)
        trailing = features.score_pressure >= 0.55

        if trailing:
            value = self._conservative_value(features)
            value += 0.3 * board_danger_after
            value += 0.9 * features.played_card_score
            value += 1.5 * features.hand_rank_pct
            if features.target_row_idx != -1 and features.immediate_penalty == 0:
                value += 2.0 if features.dangerous_row else 0.0
            return (value, board_danger_after, features.card)

        value = self._conservative_value(features)
        value += 3.0 * (1.0 - features.hand_rank_pct)
        value -= 0.5 * features.played_card_score
        if features.target_row_idx != -1 and features.row_len <= 2 and features.immediate_penalty == 0:
            value += 2.0
        return (value, -features.gap, -features.card)
