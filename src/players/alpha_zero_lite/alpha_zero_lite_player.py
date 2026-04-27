from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import time


@dataclass(frozen=True)
class _SearchState:
    board: tuple[tuple[int, ...], ...]
    scores: tuple[int, ...]
    hands: tuple[tuple[int, ...], ...]
    round_idx: int


class _Node:
    __slots__ = ("state", "prior", "depth", "children", "visits", "value_sum")

    def __init__(self, state, prior, depth):
        self.state = state
        self.prior = prior
        self.depth = depth
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def average_value(self):
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


class AlphaZeroLitePlayer:
    """AlphaZero-lite player with heuristic policy/value heads and bounded PUCT search."""

    def __init__(
        self,
        player_idx,
        search_time_limit=0.82,
        max_samples=3,
        simulations_per_sample=20,
        max_depth=2,
        c_puct=1.35,
        n_cards=104,
        n_rounds=10,
        board_size_x=5,
        policy_temperature=0.75,
    ):
        self.player_idx = player_idx
        self.search_time_limit = min(search_time_limit, 0.95)
        self.max_samples = max(1, max_samples)
        self.simulations_per_sample = max(4, simulations_per_sample)
        self.max_depth = max(1, max_depth)
        self.c_puct = c_puct
        self.n_cards = n_cards
        self.n_rounds = n_rounds
        self.board_size_x = board_size_x
        self.policy_temperature = max(0.25, policy_temperature)
        self._card_scores = (0,) + tuple(
            self._score_card_value(card) for card in range(1, self.n_cards + 1)
        )

    def action(self, hand, history):
        if len(hand) <= 1:
            return hand[0]

        board = tuple(tuple(row) for row in history["board"])
        scores = tuple(history["scores"])
        n_players = len(scores)
        unseen_cards = self._collect_unseen_cards(hand, history)
        fallback_priors, _, fallback_utilities = self._policy_value(
            tuple(hand),
            board,
            scores,
            frozenset(unseen_cards),
            n_players,
            self.player_idx,
        )
        fallback_card = max(
            hand,
            key=lambda card: (fallback_priors[card], fallback_utilities[card], -card),
        )

        expected_hidden_cards = (n_players - 1) * len(hand)
        if len(unseen_cards) < expected_hidden_cards:
            return fallback_card

        deadline = time.perf_counter() + self.search_time_limit
        rng = random.Random(self._state_seed(hand, history))
        aggregate_visits = {card: 0 for card in hand}
        aggregate_value_sums = {card: 0.0 for card in hand}
        depth_limit = self._depth_budget(len(hand))
        simulation_budget = self._simulation_budget(len(hand))
        completed_samples = 0

        while completed_samples < self._sample_budget(len(hand)) and time.perf_counter() < deadline:
            sampled_hands = self._sample_hidden_hands(hand, unseen_cards, n_players, rng)
            root_state = _SearchState(
                board=board,
                scores=scores,
                hands=tuple(tuple(player_hand) for player_hand in sampled_hands),
                round_idx=history["round"],
            )
            if len(hand) <= 3:
                exact_values = self._exact_action_values(root_state)
                for card, value in exact_values.items():
                    aggregate_visits[card] += 1
                    aggregate_value_sums[card] += value
                completed_samples += 1
                continue

            root = self._run_mcts(root_state, deadline, depth_limit, simulation_budget)
            if not root.children:
                break

            for card, child in root.children.items():
                aggregate_visits[card] += child.visits
                aggregate_value_sums[card] += child.value_sum
            completed_samples += 1

        if completed_samples == 0 or not any(aggregate_visits.values()):
            return fallback_card

        return max(
            hand,
            key=lambda card: (
                aggregate_visits[card],
                aggregate_value_sums[card] / max(1, aggregate_visits[card]),
                fallback_priors[card],
                fallback_utilities[card],
                -card,
            ),
        )

    def _run_mcts(self, root_state, deadline, depth_limit, simulation_budget):
        root = _Node(root_state, prior=1.0, depth=0)
        for _ in range(simulation_budget):
            if time.perf_counter() >= deadline:
                break
            self._search(root, deadline, depth_limit)
        return root

    def _exact_action_values(self, state):
        cache = {}
        action_values = {}

        for action in state.hands[self.player_idx]:
            next_state = self._step_state(state, action)
            action_values[action] = self._exact_state_value(next_state, cache)

        return action_values

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

    def _search(self, node, deadline, depth_limit):
        if time.perf_counter() >= deadline:
            value = self._evaluate_state(node.state)
        elif self._is_terminal(node.state) or node.depth >= depth_limit:
            value = self._evaluate_state(node.state)
        elif not node.children:
            priors, value, _ = self._policy_value_for_state(node.state, self.player_idx)
            if priors:
                for action, prior in priors.items():
                    node.children[action] = _Node(None, prior=prior, depth=node.depth + 1)
        else:
            action, child = self._select_child(node)
            if child.state is None:
                child.state = self._step_state(node.state, action)
            value = self._search(child, deadline, depth_limit)

        node.visits += 1
        node.value_sum += value
        return value

    def _select_child(self, node):
        exploration_base = math.sqrt(max(1, node.visits))
        best_choice = None
        best_key = None

        for action, child in node.children.items():
            q_value = child.average_value()
            explore = self.c_puct * child.prior * exploration_base / (1 + child.visits)
            key = (q_value + explore, child.prior, -action)
            if best_key is None or key > best_key:
                best_choice = (action, child)
                best_key = key

        return best_choice

    def _policy_value_for_state(self, state, player_idx):
        hand = state.hands[player_idx]
        if not hand:
            return {}, self._terminal_value(state.scores), {}

        opponent_cards = frozenset(
            card
            for idx, other_hand in enumerate(state.hands)
            if idx != player_idx
            for card in other_hand
        )
        return self._policy_value(
            hand,
            state.board,
            state.scores,
            opponent_cards,
            len(state.scores),
            player_idx,
        )

    def _policy_value(self, hand, board, scores, opponent_cards, n_players, player_idx):
        utilities = {}
        score_pressure = self._score_pressure(scores, player_idx)
        risk_weight = 1.75 - 0.45 * score_pressure
        dump_weight = 0.18 + 0.12 * score_pressure
        low_card_weight = 0.75 - 0.25 * score_pressure

        for card in hand:
            (
                immediate_penalty,
                row_score,
                row_gap,
                row_length,
                take_risk,
                is_low_card,
            ) = self._card_risk_snapshot(board, card, opponent_cards, n_players - 1)

            utility = -10.5 * immediate_penalty
            utility -= risk_weight * take_risk * row_score
            utility -= 0.035 * row_gap
            utility -= 0.45 * max(0, row_length - 3)
            utility += dump_weight * self._card_score(card)
            utility -= 0.001 * card

            if is_low_card:
                utility -= low_card_weight * row_score

            utilities[card] = utility

        priors = self._softmax(utilities)
        expected_utility = sum(priors[card] * utilities[card] for card in hand)
        ranks = self._fractional_ranks(scores)
        rank_term = self._rank_value(ranks[player_idx], n_players)
        score_term = -scores[player_idx] / 20.0
        value = math.tanh(0.16 * expected_utility + 0.7 * rank_term + 0.25 * score_term)
        return priors, value, utilities

    def _step_state(self, state, action):
        chosen_cards = []
        next_hands = []

        for player_idx, hand in enumerate(state.hands):
            if player_idx == self.player_idx:
                chosen_card = action
            else:
                chosen_card = self._choose_opponent_card(state, player_idx)

            chosen_cards.append(chosen_card)
            next_hands.append(tuple(card for card in hand if card != chosen_card))

        next_board, penalties = self._simulate_round(state.board, chosen_cards)
        next_scores = tuple(
            state.scores[player_idx] + penalties[player_idx]
            for player_idx in range(len(state.scores))
        )

        return _SearchState(
            board=tuple(tuple(row) for row in next_board),
            scores=next_scores,
            hands=tuple(next_hands),
            round_idx=state.round_idx + 1,
        )

    def _choose_opponent_card(self, state, player_idx):
        priors, _, utilities = self._policy_value_for_state(state, player_idx)
        hand = state.hands[player_idx]
        return max(
            hand,
            key=lambda card: (priors[card], utilities[card], -card),
        )

    def _evaluate_state(self, state):
        if self._is_terminal(state):
            return self._terminal_value(state.scores)
        _, value, _ = self._policy_value_for_state(state, self.player_idx)
        return value

    def _terminal_value(self, scores):
        ranks = self._fractional_ranks(scores)
        rank_term = self._rank_value(ranks[self.player_idx], len(scores))
        score_term = -scores[self.player_idx] / 20.0
        return math.tanh(rank_term + score_term)

    def _is_terminal(self, state):
        return not state.hands[self.player_idx] or state.round_idx >= self.n_rounds

    def _sample_budget(self, hand_size):
        if hand_size >= 8:
            return 1
        if hand_size >= 5:
            return min(self.max_samples, 2)
        return self.max_samples

    def _simulation_budget(self, hand_size):
        if hand_size >= 8:
            return max(6, self.simulations_per_sample // 2)
        if hand_size >= 5:
            return max(8, (2 * self.simulations_per_sample) // 3)
        return self.simulations_per_sample

    def _depth_budget(self, hand_size):
        if hand_size <= 3:
            return self.max_depth + 1
        return self.max_depth

    def _state_seed(self, hand, history):
        payload = repr(
            (
                "alpha_zero_lite",
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

    def _simulate_round(self, board, chosen_cards):
        next_board = [list(row) for row in board]
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

    def _card_risk_snapshot(self, board, card, opponent_cards, n_opponents):
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

        intervening_cards = sum(
            1 for unseen_card in opponent_cards if row_end < unseen_card < card
        )
        effective_interveners = min(n_opponents, intervening_cards)
        take_threshold = self.board_size_x - row_length
        if take_threshold <= 0:
            take_risk = 1.0
        else:
            take_risk = min(
                1.0,
                effective_interveners / float(take_threshold * max(1, n_opponents)),
            )

        return 0, row_score, gap + intervening_cards, row_length, take_risk, False

    def _score_pressure(self, scores, player_idx):
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
            uniform_probability = 1.0 / len(utilities)
            return {card: uniform_probability for card in utilities}
        return {
            card: weight / normalizer for card, weight in weights.items()
        }

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


AlphaZeroPlayer = AlphaZeroLitePlayer
