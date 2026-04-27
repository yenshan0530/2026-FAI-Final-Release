from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import time

from src.players.rule_based_player_base import RuleBasedPlayerBase


@dataclass(frozen=True)
class _DeterminizedState:
    board: tuple[tuple[int, ...], ...]
    scores: tuple[int, ...]
    hands: tuple[tuple[int, ...], ...]
    round_idx: int


class ISMCTSPlayer(RuleBasedPlayerBase):
    """Root-focused ISMCTS redesign with sampled opponent actions and shallow rollouts."""

    def __init__(
        self,
        player_idx,
        search_time_limit=0.45,
        max_samples=4,
        simulations_per_sample=12,
        rollout_depth=2,
        exploration_weight=1.10,
        policy_temperature=0.75,
        exact_endgame_hand_size=4,
        n_cards=104,
        n_rounds=10,
        board_size_x=5,
    ):
        super().__init__(player_idx, n_cards=n_cards, board_size_x=board_size_x)
        self.search_time_limit = min(search_time_limit, 0.95)
        self.max_samples = max(1, max_samples)
        self.simulations_per_sample = max(4, simulations_per_sample)
        self.rollout_depth = max(1, rollout_depth)
        self.exploration_weight = exploration_weight
        self.policy_temperature = max(0.25, policy_temperature)
        self.exact_endgame_hand_size = max(3, exact_endgame_hand_size)
        self.n_rounds = n_rounds
        self._policy_utility_cache = {}

    def action(self, hand, history):
        if len(hand) <= 1:
            return hand[0]

        self._policy_utility_cache = {}
        board = tuple(tuple(row) for row in history["board"])
        scores = tuple(history["scores"])
        n_players = len(scores)
        unseen_cards = self._collect_unseen_cards(hand, history)
        fallback_utilities = self._policy_utilities(
            tuple(hand),
            board,
            scores,
            frozenset(unseen_cards),
            self.player_idx,
        )
        fallback_card = max(
            hand,
            key=lambda card: (fallback_utilities[card], -card),
        )

        expected_hidden_cards = (n_players - 1) * len(hand)
        if len(unseen_cards) < expected_hidden_cards:
            return fallback_card

        if not self._should_search(hand, board, scores, frozenset(unseen_cards), fallback_utilities):
            return fallback_card

        deadline = time.perf_counter() + self.search_time_limit
        rng = random.Random(self._state_seed(hand, history))
        aggregate_visits = {card: 0 for card in hand}
        aggregate_value_sums = {card: 0.0 for card in hand}
        completed_samples = 0

        while completed_samples < self._sample_budget(len(hand)) and time.perf_counter() < deadline:
            sampled_hands = self._sample_hidden_hands(hand, unseen_cards, n_players, rng)
            root_state = _DeterminizedState(
                board=board,
                scores=scores,
                hands=tuple(tuple(player_hand) for player_hand in sampled_hands),
                round_idx=history["round"],
            )

            if len(hand) <= self.exact_endgame_hand_size:
                exact_values = self._exact_action_values(root_state)
                for card, value in exact_values.items():
                    aggregate_visits[card] += 1
                    aggregate_value_sums[card] += value
                completed_samples += 1
                continue

            sample_stats = self._root_search(root_state, deadline, rng)
            if not any(sample_stats[card][0] for card in hand):
                break

            for card in hand:
                visits, value_sum = sample_stats[card]
                aggregate_visits[card] += visits
                aggregate_value_sums[card] += value_sum
            completed_samples += 1

        if completed_samples == 0 or not any(aggregate_visits.values()):
            return fallback_card

        return max(
            hand,
            key=lambda card: (
                aggregate_value_sums[card] / max(1, aggregate_visits[card]),
                aggregate_visits[card],
                fallback_utilities[card],
                -card,
            ),
        )

    def _should_search(self, hand, board, scores, unseen_set, fallback_utilities):
        if len(hand) <= self.exact_endgame_hand_size:
            return True

        ordered_utilities = sorted(fallback_utilities.values(), reverse=True)
        best_margin = (
            ordered_utilities[0] - ordered_utilities[1]
            if len(ordered_utilities) >= 2
            else ordered_utilities[0]
        )
        score_pressure = self._score_pressure(scores)
        board_danger = self._board_danger_score(board)
        max_take_risk = 0.0
        immediate_penalty_present = False

        for card in hand:
            features = self._candidate_features(card, hand, board, unseen_set, scores)
            max_take_risk = max(
                max_take_risk,
                features.take_risk * max(1, features.row_score),
            )
            if features.immediate_penalty > 0:
                immediate_penalty_present = True

        if len(hand) >= 7:
            return False
        if len(hand) == 6:
            return (
                score_pressure >= 0.65
                or immediate_penalty_present
                or max_take_risk >= 5.0
                or best_margin <= 0.90
            )
        if len(hand) == 5:
            return (
                score_pressure >= 0.55
                or board_danger >= 24.0
                or immediate_penalty_present
                or max_take_risk >= 4.5
                or best_margin <= 1.10
            )

        return (
            score_pressure >= 0.55
            or board_danger >= 22.0
            or immediate_penalty_present
            or max_take_risk >= 4.0
            or best_margin <= 1.5
        )

    def _root_search(self, root_state, deadline, rng):
        hand = root_state.hands[self.player_idx]
        priors = self._policy_priors(
            hand,
            root_state.board,
            root_state.scores,
            frozenset(
                card
                for idx, other_hand in enumerate(root_state.hands)
                if idx != self.player_idx
                for card in other_hand
            ),
            self.player_idx,
        )
        stats = {action: [0, 0.0] for action in hand}
        total_simulations = 0

        for action in hand:
            if time.perf_counter() >= deadline:
                return stats
            value = self._simulate_action(root_state, action, rng)
            stats[action][0] += 1
            stats[action][1] += value
            total_simulations += 1

        budget = self._simulation_budget(len(hand))
        while total_simulations < budget and time.perf_counter() < deadline:
            action = self._select_root_action(stats, priors, total_simulations)
            value = self._simulate_action(root_state, action, rng)
            stats[action][0] += 1
            stats[action][1] += value
            total_simulations += 1

        return stats

    def _select_root_action(self, stats, priors, total_simulations):
        best_action = next(iter(stats))
        best_key = None
        log_total = math.log(total_simulations + 1.0)

        for action, (visits, value_sum) in stats.items():
            mean_value = value_sum / max(1, visits)
            explore = self.exploration_weight * priors[action] * math.sqrt(log_total / visits)
            key = (mean_value + explore, mean_value, priors[action], -action)
            if best_key is None or key > best_key:
                best_action = action
                best_key = key

        return best_action

    def _simulate_action(self, state, action, rng):
        next_state = self._step_state(state, action, rng=rng, stochastic=True)
        return self._rollout_value(next_state, rng, depth=1)

    def _rollout_value(self, state, rng, depth):
        if self._is_terminal(state):
            return self._terminal_value(state.scores)
        if len(state.hands[self.player_idx]) <= self.exact_endgame_hand_size:
            return self._exact_state_value(state, {})

        current_state = state
        current_depth = depth
        depth_limit = self._rollout_depth_budget(len(state.hands[self.player_idx]))

        while not self._is_terminal(current_state) and current_depth < depth_limit:
            if len(current_state.hands[self.player_idx]) <= self.exact_endgame_hand_size:
                return self._exact_state_value(current_state, {})
            action = self._best_policy_action(current_state, self.player_idx)
            current_state = self._step_state(current_state, action, rng=rng, stochastic=True)
            current_depth += 1

        if len(current_state.hands[self.player_idx]) <= self.exact_endgame_hand_size:
            return self._exact_state_value(current_state, {})
        return self._evaluate_state(current_state)

    def _exact_action_values(self, state):
        cache = {}
        values = {}

        for action in state.hands[self.player_idx]:
            next_state = self._step_state(state, action, stochastic=False)
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
                self._exact_state_value(
                    self._step_state(state, action, stochastic=False),
                    cache,
                )
                for action in state.hands[self.player_idx]
            )

        cache[state] = value
        return value

    def _step_state(self, state, action, rng=None, stochastic=False):
        chosen_cards = []
        next_hands = []

        for player_idx, hand in enumerate(state.hands):
            if player_idx == self.player_idx:
                chosen_card = action
            elif stochastic:
                chosen_card = self._sample_policy_action(state, player_idx, rng)
            else:
                chosen_card = self._best_policy_action(state, player_idx)

            chosen_cards.append(chosen_card)
            next_hands.append(tuple(card for card in hand if card != chosen_card))

        next_board, penalties = self._simulate_round(state.board, chosen_cards)
        next_scores = tuple(
            state.scores[player_idx] + penalties[player_idx]
            for player_idx in range(len(state.scores))
        )

        return _DeterminizedState(
            board=tuple(tuple(row) for row in next_board),
            scores=next_scores,
            hands=tuple(next_hands),
            round_idx=state.round_idx + 1,
        )

    def _best_policy_action(self, state, player_idx):
        hand = state.hands[player_idx]
        opponent_cards = frozenset(
            card
            for idx, other_hand in enumerate(state.hands)
            if idx != player_idx
            for card in other_hand
        )
        utilities = self._policy_utilities(
            hand,
            state.board,
            state.scores,
            opponent_cards,
            player_idx,
        )
        return max(
            hand,
            key=lambda card: (utilities[card], -card),
        )

    def _sample_policy_action(self, state, player_idx, rng):
        hand = state.hands[player_idx]
        opponent_cards = frozenset(
            card
            for idx, other_hand in enumerate(state.hands)
            if idx != player_idx
            for card in other_hand
        )
        priors = self._policy_priors(
            hand,
            state.board,
            state.scores,
            opponent_cards,
            player_idx,
        )
        threshold = rng.random()
        cumulative = 0.0
        chosen_card = hand[-1]

        for card in sorted(hand):
            cumulative += priors[card]
            if cumulative >= threshold:
                chosen_card = card
                break

        return chosen_card

    def _policy_utilities(self, hand, board, scores, unseen_set, player_idx):
        key = (
            player_idx,
            tuple(hand),
            tuple(tuple(row) for row in board),
            tuple(scores),
            unseen_set,
        )
        cached = self._policy_utility_cache.get(key)
        if cached is not None:
            return cached

        utilities = {}
        phase = self._phase(len(hand))
        score_pressure = self._score_pressure_for_player(scores, player_idx)
        board_list = [list(row) for row in board]
        baseline_safe_count = self._future_safe_count(board_list, hand, unseen_set, scores)
        baseline_board_danger = self._board_danger_score(board_list)
        trailing = score_pressure >= 0.55

        for card in hand:
            features = self._candidate_features(card, hand, board, unseen_set, scores)
            board_after, penalty = self._simulate_our_placement(board_list, card)
            remaining_hand = [remaining for remaining in hand if remaining != card]
            safe_after = self._future_safe_count(
                board_after,
                remaining_hand,
                unseen_set,
                scores,
            )
            safe_gain = safe_after - baseline_safe_count
            danger_after = self._board_danger_score(board_after)
            danger_drop = baseline_board_danger - danger_after
            base_value = self._phase_base_value(features, phase)

            if trailing:
                utility = 0.65 * self._lpra_value(features, danger_after)
                utility += 0.35 * base_value
            else:
                utility = base_value

            utility += 0.40 * safe_gain
            utility += 0.10 * danger_drop

            cheap_reset_threshold = 5 if trailing else 3
            if penalty > 0 and penalty <= cheap_reset_threshold and safe_gain >= 2:
                utility = max(
                    utility,
                    self._crs_value(features, safe_gain, danger_drop),
                )

            utilities[card] = utility

        self._policy_utility_cache[key] = utilities
        return utilities

    def _policy_priors(self, hand, board, scores, unseen_set, player_idx):
        return self._softmax(
            self._policy_utilities(hand, board, scores, unseen_set, player_idx)
        )

    def _evaluate_state(self, state):
        if self._is_terminal(state):
            return self._terminal_value(state.scores)

        hand = state.hands[self.player_idx]
        opponent_cards = frozenset(
            card
            for idx, other_hand in enumerate(state.hands)
            if idx != self.player_idx
            for card in other_hand
        )
        utilities = self._policy_utilities(
            hand,
            state.board,
            state.scores,
            opponent_cards,
            self.player_idx,
        )
        best_utility = max(utilities.values()) if utilities else 0.0
        safe_count = sum(1 for value in utilities.values() if value >= -2.0)
        safe_ratio = safe_count / max(1, len(hand))
        ranks = self._fractional_ranks(state.scores)
        rank_term = self._rank_value(ranks[self.player_idx], len(state.scores))
        score_term = -state.scores[self.player_idx] / 20.0
        return math.tanh(0.16 * best_utility + 0.80 * rank_term + 0.25 * score_term + 0.30 * safe_ratio)

    def _terminal_value(self, scores):
        ranks = self._fractional_ranks(scores)
        rank_term = self._rank_value(ranks[self.player_idx], len(scores))
        score_term = -scores[self.player_idx] / 20.0
        return math.tanh(rank_term + score_term)

    def _is_terminal(self, state):
        return not state.hands[self.player_idx] or state.round_idx >= self.n_rounds

    def _sample_budget(self, hand_size):
        if hand_size >= 8:
            return min(self.max_samples, 2)
        if hand_size >= 5:
            return min(self.max_samples, 3)
        return self.max_samples

    def _simulation_budget(self, hand_size):
        if hand_size >= 8:
            return max(8, self.simulations_per_sample - 4)
        if hand_size >= 5:
            return self.simulations_per_sample
        return self.simulations_per_sample + 4

    def _rollout_depth_budget(self, hand_size):
        if hand_size >= 7:
            return self.rollout_depth
        return self.rollout_depth + 1

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

    def _phase_base_value(self, features, phase):
        low_first = self._lfs_value(features, phase)
        closest_fit = self._cfc_value(features)
        blocker = self._hfb_value(features, phase)

        if phase == "early":
            return max(low_first, blocker)
        if phase == "mid":
            return 0.70 * closest_fit + 0.30 * max(low_first, blocker)
        return 0.70 * blocker + 0.30 * closest_fit

    def _lfs_value(self, features, phase):
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
        return value

    def _cfc_value(self, features):
        value = self._conservative_value(features)
        value -= float(features.gap)
        value -= 4.0 * max(0, features.row_len - 2)
        value -= 3.0 * features.intervening_count
        value -= 2.0 * features.row_score
        value += 6.0 if features.target_row_idx != -1 and features.row_len <= 2 else 0.0
        value -= 3.0 if features.dangerous_row else 0.0
        if features.target_row_idx == -1 and features.low_reset_cost <= 2:
            value += 1.0
        return value

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

    def _lpra_value(self, features, board_danger_after):
        value = self._conservative_value(features)
        value += 0.3 * board_danger_after
        value += 0.9 * features.played_card_score
        value += 1.5 * features.hand_rank_pct
        if features.target_row_idx != -1 and features.immediate_penalty == 0:
            value += 2.0 if features.dangerous_row else 0.0
        return value

    def _crs_value(self, features, safe_gain, danger_drop):
        value = self._conservative_value(features)
        value += 8.0
        value += 4.0 * safe_gain
        value += 0.8 * danger_drop
        return value

    def _state_seed(self, hand, history):
        payload = repr(
            (
                "ismcts",
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
