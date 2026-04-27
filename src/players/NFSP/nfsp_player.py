from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

from src.players.rule_based_player_base import RuleBasedPlayerBase


@dataclass(frozen=True)
class _ResponseState:
    board: tuple[tuple[int, ...], ...]
    scores: tuple[int, ...]
    hands: tuple[tuple[int, ...], ...]
    round_idx: int


class NFSPPlayer(RuleBasedPlayerBase):
    """Lightweight NFSP-style player with average-policy and best-response mixing."""

    def __init__(
        self,
        player_idx,
        anticipatory_eta=0.25,
        best_response_samples=3,
        best_response_depth=1,
        n_cards=104,
        n_rounds=10,
        board_size_x=5,
        policy_temperature=0.7,
        exact_endgame_hand_size=4,
    ):
        super().__init__(player_idx, n_cards=n_cards, board_size_x=board_size_x)
        self.anticipatory_eta = anticipatory_eta
        self.best_response_samples = max(1, best_response_samples)
        self.best_response_depth = max(1, best_response_depth)
        self.n_rounds = n_rounds
        self.policy_temperature = max(0.25, policy_temperature)
        self.exact_endgame_hand_size = max(3, exact_endgame_hand_size)
        self._history_matrix = ()
        self._average_utility_cache = {}

    def action(self, hand, history):
        if len(hand) <= 1:
            return hand[0]

        self._history_matrix = tuple(
            tuple(actions) for actions in history.get("history_matrix", [])
        )
        self._average_utility_cache = {}
        board = tuple(tuple(row) for row in history["board"])
        scores = tuple(history["scores"])
        n_players = len(scores)
        unseen_cards = self._collect_unseen_cards(hand, history)
        aggression_target = self._historical_aggression(hand, self.player_idx)
        average_utilities = self._average_policy_utilities(
            tuple(hand),
            board,
            scores,
            frozenset(unseen_cards),
            self.player_idx,
            aggression_target,
        )
        average_policy = self._softmax(average_utilities)

        expected_hidden_cards = (n_players - 1) * len(hand)
        if len(unseen_cards) >= expected_hidden_cards:
            best_response_values = self._best_response_values(hand, history, unseen_cards)
        else:
            best_response_values = {card: 0.0 for card in hand}

        eta = self._effective_eta(scores)
        return max(
            hand,
            key=lambda card: (
                (1.0 - eta) * average_utilities[card] + eta * 6.0 * best_response_values[card],
                best_response_values[card],
                average_policy[card],
                -card,
            ),
        )

    def _best_response_values(self, hand, history, unseen_cards):
        board = tuple(tuple(row) for row in history["board"])
        scores = tuple(history["scores"])
        n_players = len(scores)
        rng = random.Random(self._state_seed(hand, history))
        aggregate_values = {card: 0.0 for card in hand}
        completed_samples = 0

        while completed_samples < self.best_response_samples:
            sampled_hands = self._sample_hidden_hands(hand, unseen_cards, n_players, rng)
            root_state = _ResponseState(
                board=board,
                scores=scores,
                hands=tuple(tuple(player_hand) for player_hand in sampled_hands),
                round_idx=history["round"],
            )

            if len(hand) <= self.exact_endgame_hand_size:
                sample_values = self._exact_action_values(root_state)
            else:
                sample_values = {
                    action: self._best_response_state_value(
                        self._step_state(root_state, action),
                        depth=self.best_response_depth,
                    )
                    for action in hand
                }

            for card, value in sample_values.items():
                aggregate_values[card] += value

            completed_samples += 1

        return {
            card: aggregate_values[card] / float(completed_samples)
            for card in hand
        }

    def _best_response_state_value(self, state, depth):
        if self._is_terminal(state):
            return self._terminal_value(state.scores)
        if depth <= 0:
            return self._evaluate_state(state)
        if len(state.hands[self.player_idx]) <= self.exact_endgame_hand_size:
            return self._exact_state_value(state, {})

        return max(
            self._best_response_state_value(self._step_state(state, action), depth - 1)
            for action in state.hands[self.player_idx]
        )

    def _exact_action_values(self, state):
        cache = {}
        values = {}

        for action in state.hands[self.player_idx]:
            next_state = self._step_state(state, action)
            values[action] = self._exact_state_value(next_state, cache)

        return values

    def _exact_state_value(self, state, cache):
        cached_value = cache.get(state)
        if cached_value is not None:
            return cached_value

        if self._is_terminal(state):
            value = self._terminal_value(state.scores)
        else:
            value = max(
                self._exact_state_value(self._step_state(state, action), cache)
                for action in state.hands[self.player_idx]
            )

        cache[state] = value
        return value

    def _step_state(self, state, action):
        chosen_cards = []
        next_hands = []

        for player_idx, hand in enumerate(state.hands):
            if player_idx == self.player_idx:
                chosen_card = action
            else:
                chosen_card = self._average_policy_action(state, player_idx)

            chosen_cards.append(chosen_card)
            next_hands.append(tuple(card for card in hand if card != chosen_card))

        next_board, penalties = self._simulate_round(state.board, chosen_cards)
        next_scores = tuple(
            state.scores[player_idx] + penalties[player_idx]
            for player_idx in range(len(state.scores))
        )

        return _ResponseState(
            board=tuple(tuple(row) for row in next_board),
            scores=next_scores,
            hands=tuple(next_hands),
            round_idx=state.round_idx + 1,
        )

    def _average_policy_action(self, state, player_idx):
        hand = state.hands[player_idx]
        aggression_target = self._historical_aggression(hand, player_idx)
        opponent_cards = frozenset(
            card
            for idx, other_hand in enumerate(state.hands)
            if idx != player_idx
            for card in other_hand
        )
        utilities = self._average_policy_utilities(
            hand,
            state.board,
            state.scores,
            opponent_cards,
            player_idx,
            aggression_target,
        )
        return max(
            hand,
            key=lambda card: (utilities[card], -card),
        )

    def _average_policy_utilities(
        self,
        hand,
        board,
        scores,
        unseen_set,
        player_idx,
        aggression_target,
    ):
        key = (
            player_idx,
            tuple(hand),
            tuple(tuple(row) for row in board),
            tuple(scores),
            unseen_set,
            round(aggression_target, 6),
        )
        cached = self._average_utility_cache.get(key)
        if cached is not None:
            return cached

        utilities = {}
        score_pressure = self._score_pressure_for_player(scores, player_idx)

        for card in hand:
            features = self._candidate_features(card, hand, board, unseen_set, scores)
            utility = self._conservative_value(features)
            utility -= 2.6 * abs(features.hand_rank_pct - aggression_target)
            utility += 0.35 * float(features.run_middle_flag)
            utility += 0.5 * float(features.outlier_flag)
            utility += 0.18 * score_pressure * features.played_card_score
            utility += 0.08 * max(0, 10 - features.gap)

            if features.target_row_idx == -1:
                utility -= 0.55 * features.low_reset_cost
                if features.low_reset_cost <= 3:
                    utility += 0.9 * (1.0 - score_pressure)

            utilities[card] = utility

        self._average_utility_cache[key] = utilities
        return utilities

    def _historical_aggression(self, current_hand, player_idx):
        past_actions = [
            actions[player_idx]
            for actions in self._history_matrix
            if player_idx < len(actions)
        ]
        if not past_actions:
            return 0.5

        reconstructed_hand = sorted(tuple(current_hand) + tuple(past_actions))
        turn_hand = list(reconstructed_hand)
        aggression_scores = []

        for action in past_actions:
            action_idx = turn_hand.index(action)
            aggression_scores.append(action_idx / max(1, len(turn_hand) - 1))
            turn_hand.pop(action_idx)

        return sum(aggression_scores) / len(aggression_scores)

    def _evaluate_state(self, state):
        if self._is_terminal(state):
            return self._terminal_value(state.scores)

        hand = state.hands[self.player_idx]
        aggression_target = self._historical_aggression(hand, self.player_idx)
        opponent_cards = frozenset(
            card
            for idx, other_hand in enumerate(state.hands)
            if idx != self.player_idx
            for card in other_hand
        )
        utilities = self._average_policy_utilities(
            hand,
            state.board,
            state.scores,
            opponent_cards,
            self.player_idx,
            aggression_target,
        )
        best_utility = max(utilities.values()) if utilities else 0.0
        ranks = self._fractional_ranks(state.scores)
        rank_term = self._rank_value(ranks[self.player_idx], len(state.scores))
        score_term = -state.scores[self.player_idx] / 20.0
        return math.tanh(0.14 * best_utility + 0.8 * rank_term + 0.25 * score_term)

    def _terminal_value(self, scores):
        ranks = self._fractional_ranks(scores)
        rank_term = self._rank_value(ranks[self.player_idx], len(scores))
        score_term = -scores[self.player_idx] / 20.0
        return math.tanh(rank_term + score_term)

    def _effective_eta(self, scores):
        return min(0.8, max(0.1, self.anticipatory_eta + 0.25 * self._score_pressure_for_player(scores, self.player_idx)))

    def _is_terminal(self, state):
        return not state.hands[self.player_idx] or state.round_idx >= self.n_rounds

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

    def _simulate_round(self, board, chosen_cards):
        next_board = [list(row) for row in board]
        penalties = [0] * len(chosen_cards)

        for card, player_idx in sorted((card, idx) for idx, card in enumerate(chosen_cards)):
            penalties[player_idx] += self._place_card(next_board, card)

        return next_board, penalties

    def _state_seed(self, hand, history):
        payload = repr(
            (
                "nfsp",
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

    def _softmax(self, utilities):
        if not utilities:
            return {}

        max_utility = max(utilities.values())
        weights = {
            card: math.exp((utility - max_utility) / self.policy_temperature)
            for card, utility in utilities.items()
        }
        normalizer = sum(weights.values())
        if normalizer <= 0.0:
            uniform = 1.0 / len(utilities)
            return {card: uniform for card in utilities}
        return {card: weight / normalizer for card, weight in weights.items()}

    def _score_pressure_for_player(self, scores, player_idx):
        low_score = min(scores)
        high_score = max(scores)
        if high_score <= low_score:
            return 0.0
        return (scores[player_idx] - low_score) / float(high_score - low_score)

    def _rank_value(self, rank, n_players):
        if n_players <= 1:
            return 0.0
        return (n_players + 1 - 2 * rank) / float(n_players - 1)

    def _fractional_ranks(self, scores):
        ranks = []
        for score in scores:
            better_count = sum(1 for other_score in scores if other_score < score)
            same_count = sum(1 for other_score in scores if other_score == score)
            ranks.append((2 * better_count + same_count + 1) / 2.0)
        return ranks
