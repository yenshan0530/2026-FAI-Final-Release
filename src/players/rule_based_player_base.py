from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateFeatures:
    card: int
    target_row_idx: int
    target_row_end: int
    row_score: int
    row_len: int
    gap: int
    open_slots_before_take: int
    intervening_count: int
    low_reset_cost: int
    played_card_score: int
    dangerous_row: bool
    hand_rank_pct: float
    run_middle_flag: bool
    outlier_flag: bool
    score_pressure: float
    immediate_penalty: int
    take_risk: float


class RuleBasedPlayerBase:
    """Shared helpers for lightweight human-strategy-inspired rule agents."""

    def __init__(self, player_idx, n_cards=104, board_size_x=5):
        self.player_idx = player_idx
        self.n_cards = n_cards
        self.board_size_x = board_size_x
        self._card_scores = (0,) + tuple(
            self._score_card_value(card) for card in range(1, self.n_cards + 1)
        )

    def action(self, hand, history):
        if len(hand) <= 1:
            return hand[0]

        context = self._build_context(hand, history)
        best_card = hand[0]
        best_key = None

        for card in hand:
            features = self._candidate_features(
                card=card,
                hand=hand,
                board=context["board"],
                unseen_set=context["unseen_set"],
                scores=context["scores"],
            )
            key = self._score_candidate(features, hand, history, context)
            if best_key is None or key > best_key:
                best_key = key
                best_card = card

        return best_card

    def _build_context(self, hand, history):
        board = [list(row) for row in history["board"]]
        scores = list(history["scores"])
        unseen_set = frozenset(self._collect_unseen_cards(hand, history))
        return {
            "board": board,
            "scores": scores,
            "unseen_set": unseen_set,
            "phase": self._phase(len(hand)),
            "baseline_safe_future_count": self._future_safe_count(
                board, hand, unseen_set, scores
            ),
            "baseline_board_danger": self._board_danger_score(board),
        }

    def _phase(self, hand_size):
        if hand_size >= 7:
            return "early"
        if hand_size >= 4:
            return "mid"
        return "late"

    def _score_candidate(self, features, hand, history, context):
        raise NotImplementedError

    def _candidate_features(self, card, hand, board, unseen_set, scores):
        target_row_idx, target_row_end = self._best_row_for_card(board, card)
        hand_idx = hand.index(card)
        hand_rank_pct = hand_idx / max(1, len(hand) - 1)
        run_middle_flag, outlier_flag = self._hand_shape_flags(hand, hand_idx)
        score_pressure = self._score_pressure(scores)
        played_card_score = self._card_score(card)

        if target_row_idx == -1:
            _, chosen_row, row_score = self._lowest_penalty_row(board)
            row_len = len(chosen_row)
            return CandidateFeatures(
                card=card,
                target_row_idx=-1,
                target_row_end=-1,
                row_score=row_score,
                row_len=row_len,
                gap=0,
                open_slots_before_take=0,
                intervening_count=0,
                low_reset_cost=row_score,
                played_card_score=played_card_score,
                dangerous_row=self._row_is_dangerous(chosen_row),
                hand_rank_pct=hand_rank_pct,
                run_middle_flag=run_middle_flag,
                outlier_flag=outlier_flag,
                score_pressure=score_pressure,
                immediate_penalty=row_score,
                take_risk=1.0,
            )

        chosen_row = board[target_row_idx]
        row_score = self._row_score(chosen_row)
        row_len = len(chosen_row)
        gap = card - target_row_end
        open_slots_before_take = max(0, self.board_size_x - row_len)

        if row_len >= self.board_size_x:
            immediate_penalty = row_score
            take_risk = 1.0
            intervening_count = 0
        else:
            intervening_count = sum(
                1 for unseen_card in unseen_set if target_row_end < unseen_card < card
            )
            n_opponents = max(1, len(scores) - 1)
            if open_slots_before_take <= 0:
                take_risk = 1.0
            else:
                effective_interveners = min(n_opponents, intervening_count)
                take_risk = min(
                    1.0,
                    effective_interveners
                    / float(open_slots_before_take * n_opponents),
                )
            immediate_penalty = 0

        return CandidateFeatures(
            card=card,
            target_row_idx=target_row_idx,
            target_row_end=target_row_end,
            row_score=row_score,
            row_len=row_len,
            gap=gap,
            open_slots_before_take=open_slots_before_take,
            intervening_count=intervening_count,
            low_reset_cost=0,
            played_card_score=played_card_score,
            dangerous_row=self._row_is_dangerous(chosen_row),
            hand_rank_pct=hand_rank_pct,
            run_middle_flag=run_middle_flag,
            outlier_flag=outlier_flag,
            score_pressure=score_pressure,
            immediate_penalty=immediate_penalty,
            take_risk=take_risk,
        )

    def _hand_shape_flags(self, hand, hand_idx):
        left_gap = hand[hand_idx] - hand[hand_idx - 1] if hand_idx > 0 else 999
        right_gap = (
            hand[hand_idx + 1] - hand[hand_idx]
            if hand_idx < len(hand) - 1
            else 999
        )
        run_middle_flag = (
            0 < hand_idx < len(hand) - 1 and left_gap <= 6 and right_gap <= 6
        )
        outlier_flag = left_gap >= 12 and right_gap >= 12
        return run_middle_flag, outlier_flag

    def _score_pressure(self, scores):
        low_score = min(scores)
        high_score = max(scores)
        if high_score <= low_score:
            return 0.0
        return (scores[self.player_idx] - low_score) / float(high_score - low_score)

    def _conservative_value(self, features):
        value = -10.0 * features.immediate_penalty
        value -= 2.4 * features.take_risk * max(1, features.row_score)
        value -= 0.04 * features.gap
        value -= 0.8 * max(0, features.row_len - 2)
        value -= 0.75 * features.intervening_count
        if features.dangerous_row:
            value -= 2.5
        return value

    def _simulate_our_placement(self, board, card):
        next_board = [row[:] for row in board]
        penalty = self._place_card(next_board, card)
        return next_board, penalty

    def _future_safe_count(self, board, hand, unseen_set, scores):
        safe_count = 0
        for card in hand:
            features = self._candidate_features(card, hand, board, unseen_set, scores)
            if (
                features.immediate_penalty == 0
                and features.take_risk <= 0.6
                and features.row_len <= 3
            ):
                safe_count += 1
        return safe_count

    def _board_danger_score(self, board):
        total = 0.0
        for row in board:
            row_score = self._row_score(row)
            total += row_score
            total += 2.0 * max(0, len(row) - 3)
            if any(card == 55 for card in row):
                total += 6.0
            if any(card % 11 == 0 for card in row):
                total += 3.0
        return total

    def _row_is_dangerous(self, row):
        return self._row_score(row) >= 5 or any(
            card == 55 or card % 11 == 0 for card in row
        )

    def _collect_unseen_cards(self, hand, history):
        visible = set(hand)

        for row in history["board"]:
            visible.update(row)

        for actions in history.get("history_matrix", []):
            visible.update(actions)

        for board_snapshot in history.get("board_history", []):
            for row in board_snapshot:
                visible.update(row)

        return [card for card in range(1, self.n_cards + 1) if card not in visible]

    def _best_row_for_card(self, board, card):
        best_row_idx = -1
        best_row_end = -1

        for row_idx, row in enumerate(board):
            row_end = row[-1]
            if row_end < card and row_end > best_row_end:
                best_row_end = row_end
                best_row_idx = row_idx

        return best_row_idx, best_row_end

    def _lowest_penalty_row(self, board):
        chosen_row_key = None
        chosen_row_idx = -1
        chosen_row = None
        chosen_row_score = 0

        for row_idx, row in enumerate(board):
            row_score = self._row_score(row)
            row_key = (row_score, len(row), row_idx)
            if chosen_row_key is None or row_key < chosen_row_key:
                chosen_row_key = row_key
                chosen_row_idx = row_idx
                chosen_row = row
                chosen_row_score = row_score

        return chosen_row_idx, chosen_row, chosen_row_score

    def _place_card(self, board, card):
        row_idx, _ = self._best_row_for_card(board, card)

        if row_idx == -1:
            row_idx, _, row_score = self._lowest_penalty_row(board)
            board[row_idx] = [card]
            return row_score

        row = board[row_idx]
        if len(row) >= self.board_size_x:
            row_score = self._row_score(row)
            board[row_idx] = [card]
            return row_score

        row.append(card)
        return 0

    def _row_score(self, row):
        return sum(self._card_score(card) for card in row)

    @staticmethod
    def _score_card_value(card):
        if card % 55 == 0:
            return 7
        if card % 11 == 0:
            return 5
        if card % 10 == 0:
            return 3
        if card % 5 == 0:
            return 2
        return 1

    def _card_score(self, card):
        if 0 < card <= self.n_cards:
            return self._card_scores[card]
        return self._score_card_value(card)
