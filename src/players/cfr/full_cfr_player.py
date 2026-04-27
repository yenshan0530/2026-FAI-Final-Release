from __future__ import annotations

import hashlib
import itertools
import random
import time


class FullCFRPlayer:
    """Local-round CFR player for the 6 Nimmt assignment engine."""

    def __init__(
        self,
        player_idx,
        search_time_limit=0.82,
        max_samples=6,
        base_iterations=8,
        endgame_iterations=28,
        max_profile_work=50000,
        min_iterations=4,
        max_cfr_hand_size=6,
        n_cards=104,
        board_size_x=5,
        rank_weight=0.35,
        dump_weight=0.15,
    ):
        self.player_idx = player_idx
        self.search_time_limit = min(search_time_limit, 0.95)
        self.max_samples = max_samples
        self.base_iterations = base_iterations
        self.endgame_iterations = max(endgame_iterations, base_iterations)
        self.max_profile_work = max_profile_work
        self.min_iterations = min(min_iterations, self.base_iterations)
        self.max_cfr_hand_size = max(1, max_cfr_hand_size)
        self.n_cards = n_cards
        self.board_size_x = board_size_x
        self.rank_weight = rank_weight
        self.dump_weight = dump_weight
        self._card_scores = (0,) + tuple(
            self._score_card_value(card) for card in range(1, self.n_cards + 1)
        )

    def action(self, hand, history):
        if len(hand) == 1:
            return hand[0]

        board = [row[:] for row in history["board"]]
        scores = list(history["scores"])
        n_players = len(scores)
        unseen_cards = self._collect_unseen_cards(hand, history)
        heuristic_scores = self._heuristic_scores(hand, board, unseen_cards, n_players)

        expected_hidden_cards = (n_players - 1) * len(hand)
        # The engine leaves most of the 104-card deck undealt, so the unseen pool
        # is larger than the opponents' live hidden hands. CFR only needs enough
        # unseen candidates to sample those remaining hidden cards.
        if len(unseen_cards) < expected_hidden_cards:
            return max(hand, key=lambda card: (heuristic_scores[card], -card))

        sample_budget, iterations = self._resolve_search_budget(len(hand), n_players)
        if sample_budget <= 0 or iterations <= 0:
            return max(hand, key=lambda card: (heuristic_scores[card], -card))

        deadline = time.perf_counter() + self.search_time_limit
        rng = random.Random(self._state_seed(hand, history))
        aggregate_strategy = {card: 0.0 for card in hand}
        completed_samples = 0

        while completed_samples < sample_budget and time.perf_counter() < deadline:
            sampled_hands = self._sample_hidden_hands(hand, unseen_cards, n_players, rng)
            strategy = self._solve_sampled_round(board, scores, sampled_hands, iterations, deadline)
            if strategy is None:
                break

            for card, probability in strategy.items():
                aggregate_strategy[card] += probability
            completed_samples += 1

        if completed_samples == 0:
            return max(hand, key=lambda card: (heuristic_scores[card], -card))

        averaged_strategy = {
            card: aggregate_strategy[card] / completed_samples for card in hand
        }
        return max(
            hand,
            key=lambda card: (averaged_strategy[card], heuristic_scores[card], -card),
        )

    def _sample_budget(self, hand_size):
        if hand_size >= 8:
            return min(self.max_samples, 2)
        if hand_size >= 5:
            return min(self.max_samples, 4)
        return self.max_samples

    def _iteration_budget(self, hand_size):
        if hand_size <= 1:
            return self.endgame_iterations
        fraction_complete = (10 - hand_size) / 9.0
        extra_iters = int((self.endgame_iterations - self.base_iterations) * fraction_complete)
        return self.base_iterations + extra_iters

    def _resolve_search_budget(self, hand_size, n_players):
        if hand_size > self.max_cfr_hand_size:
            return 0, 0

        sample_budget = self._sample_budget(hand_size)
        iterations = self._iteration_budget(hand_size)
        profile_count = hand_size ** n_players

        # Bound total joint-profile work so large early-round hands stay reliable
        # under the engine's wall-clock timeout in multi-worker tournaments.
        while (
            sample_budget > 1
            and profile_count * sample_budget * iterations > self.max_profile_work
        ):
            sample_budget -= 1

        while (
            iterations > self.min_iterations
            and profile_count * sample_budget * iterations > self.max_profile_work
        ):
            iterations -= 1

        return sample_budget, iterations

    def _state_seed(self, hand, history):
        payload = repr(
            (
                self.player_idx,
                history["round"],
                tuple(hand),
                tuple(tuple(row) for row in history["board"]),
                tuple(history["scores"]),
                tuple(tuple(actions) for actions in history.get("history_matrix", [])),
            )
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big")

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

    def _sample_hidden_hands(self, hand, unseen_cards, n_players, rng):
        shuffled = unseen_cards[:]
        rng.shuffle(shuffled)
        hand_size = len(hand)
        sampled_hands = []
        offset = 0

        for player_idx in range(n_players):
            if player_idx == self.player_idx:
                sampled_hands.append(list(hand))
                continue

            next_hand = sorted(shuffled[offset : offset + hand_size])
            sampled_hands.append(next_hand)
            offset += hand_size

        return sampled_hands

    def _solve_sampled_round(self, board, scores, sampled_hands, iterations, deadline):
        action_lists = [tuple(player_hand) for player_hand in sampled_hands]
        profile_data = []
        action_ranges = [range(len(actions)) for actions in action_lists]

        for profile_indices in itertools.product(*action_ranges):
            if time.perf_counter() >= deadline:
                return None

            chosen_cards = [action_lists[player_idx][action_idx] for player_idx, action_idx in enumerate(profile_indices)]
            utilities = self._profile_utilities(board, scores, chosen_cards)
            profile_data.append((profile_indices, utilities))

        regrets = [[0.0] * len(actions) for actions in action_lists]
        strategy_sums = [[0.0] * len(actions) for actions in action_lists]

        for _ in range(iterations):
            if time.perf_counter() >= deadline:
                break

            strategies = []
            for player_idx, actions in enumerate(action_lists):
                strategy = self._regret_matching_strategy(regrets[player_idx], len(actions))
                strategies.append(strategy)
                for action_idx, probability in enumerate(strategy):
                    strategy_sums[player_idx][action_idx] += probability

            expected_utilities = [0.0] * len(action_lists)
            action_utilities = [[0.0] * len(actions) for actions in action_lists]

            for profile_indices, utilities in profile_data:
                probability = 1.0
                for player_idx, action_idx in enumerate(profile_indices):
                    probability *= strategies[player_idx][action_idx]
                    if probability == 0.0:
                        break

                if probability == 0.0:
                    continue

                for player_idx, utility in enumerate(utilities):
                    expected_utilities[player_idx] += probability * utility

                for player_idx, action_idx in enumerate(profile_indices):
                    action_probability = strategies[player_idx][action_idx]
                    if action_probability > 0.0:
                        action_utilities[player_idx][action_idx] += (probability / action_probability) * utilities[player_idx]

            for player_idx, player_regrets in enumerate(regrets):
                baseline = expected_utilities[player_idx]
                for action_idx in range(len(player_regrets)):
                    player_regrets[action_idx] += action_utilities[player_idx][action_idx] - baseline

        total = sum(strategy_sums[self.player_idx])
        if total <= 0.0:
            uniform_probability = 1.0 / len(action_lists[self.player_idx])
            return {
                card: uniform_probability for card in action_lists[self.player_idx]
            }

        return {
            action_lists[self.player_idx][action_idx]: strategy_sums[self.player_idx][action_idx] / total
            for action_idx in range(len(action_lists[self.player_idx]))
        }

    def _regret_matching_strategy(self, regrets, action_count):
        positive_regrets = [max(0.0, regret) for regret in regrets]
        normalizer = sum(positive_regrets)

        if normalizer <= 0.0:
            uniform_probability = 1.0 / action_count
            return [uniform_probability] * action_count

        return [regret / normalizer for regret in positive_regrets]

    def _profile_utilities(self, board, scores, chosen_cards):
        next_board, penalties = self._simulate_round(board, chosen_cards)
        future_scores = [scores[player_idx] + penalties[player_idx] for player_idx in range(len(scores))]
        round_ranks = self._fractional_ranks(future_scores)

        utilities = []
        for player_idx, card in enumerate(chosen_cards):
            utility = -penalties[player_idx]
            utility -= self.rank_weight * round_ranks[player_idx]
            utility += self.dump_weight * self._card_score(card)
            if self._board_is_reset_card(next_board, card):
                utility += 0.1
            utilities.append(utility)
        return utilities

    def _fractional_ranks(self, scores):
        ranks = []
        for score in scores:
            better_count = sum(1 for other_score in scores if other_score < score)
            same_count = sum(1 for other_score in scores if other_score == score)
            ranks.append((2 * better_count + same_count + 1) / 2.0)
        return ranks

    def _simulate_round(self, board, chosen_cards):
        next_board = [row[:] for row in board]
        penalties = [0] * len(chosen_cards)

        for card, player_idx in sorted((card, idx) for idx, card in enumerate(chosen_cards)):
            penalties[player_idx] += self._place_card(next_board, card)

        return next_board, penalties

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
        """Return the engine tie-break choice for a card lower than every row."""
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

    def _board_is_reset_card(self, board, card):
        return any(len(row) == 1 and row[0] == card for row in board)

    def _heuristic_scores(self, hand, board, unseen_cards, n_players):
        unseen_set = set(unseen_cards)
        heuristic_scores = {}

        for card in hand:
            card_penalty, row_score, row_gap, row_length, take_risk, is_low_card = self._card_risk_snapshot(
                board, card, unseen_set, n_players - 1
            )

            utility = -12.0 * card_penalty
            utility -= 1.75 * take_risk * row_score
            utility -= 0.04 * row_gap
            utility -= 0.5 * max(0, row_length - 3)
            utility += 0.2 * self._card_score(card)
            utility -= 0.001 * card

            if is_low_card:
                utility -= 0.75 * row_score

            heuristic_scores[card] = utility

        return heuristic_scores

    def _card_risk_snapshot(self, board, card, unseen_set, n_opponents):
        row_idx, row_end = self._best_row_for_card(board, card)

        if row_idx == -1:
            _, chosen_row, row_score = self._lowest_penalty_row(board)
            row_length = len(chosen_row)
            return row_score, row_score, 0, row_length, 1.0, True

        chosen_row = board[row_idx]
        row_score = self._row_score(chosen_row)
        row_length = len(chosen_row)
        gap = card - row_end

        if row_length >= self.board_size_x:
            return row_score, row_score, gap, row_length, 1.0, False

        intervening_cards = sum(1 for unseen_card in unseen_set if row_end < unseen_card < card)
        effective_interveners = min(n_opponents, intervening_cards)
        take_threshold = self.board_size_x - row_length
        if take_threshold <= 0:
            take_risk = 1.0
        else:
            take_risk = min(1.0, effective_interveners / float(take_threshold * max(1, n_opponents)))

        return 0, row_score, gap + intervening_cards, row_length, take_risk, False

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
