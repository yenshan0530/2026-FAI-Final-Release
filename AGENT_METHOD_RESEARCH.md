# 6 Nimmt! Agent Method Research

This file is research/planning only. It does not contain implementation code.

## Assignment Constraints To Preserve

- Use the assignment game, not the standard physical-table rules where a low card can choose any row. In this engine, if a played card is lower than all row ends, the row taken is deterministic: least bullheads, then shortest row, then smallest row index.
- Each `action(hand, history)` call has a 1 second decision limit. Timeout fallback is the smallest card.
- Do not catch `BaseException` or use bare `except:` because the engine's `TimeoutException` inherits from `BaseException`.
- Each player must stay under the 1GB RAM limit.
- Do not use multiprocessing or threading inside the submitted agent. The tournament runner already parallelizes matchups externally.
- CPU-only final evaluation; no GPU.
- Python version is 3.13.11. Allowed packages are the pinned packages in `requirements.txt`.
- Final evaluation is closed-network, so any trained model or generated table must be submitted with the code; the agent cannot download models or data during evaluation.
- Final tournament format is random partitioning. Average rank is more important than raw penalty sum.
- We can submit two best players, so method diversity matters. One robust generalist plus one different counter-style is likely better than two near-identical variants.
- The engine passes a copied hand and deep-copied history. The useful fields visible during `action` are current board, scores, round, previous round actions, board history, and score history.
- Hands are already sorted. Values that depend on the current game should be recomputed from `history` rather than stored blindly, because timeouts/default moves can change the true game state.

## Game Implications

- The game is simultaneous-move and imperfect-information: we know our hand and public history, but not opponents' remaining hands.
- Cards are processed from smallest to largest after all players choose. A card that looks safe can become unsafe if lower cards fill its row before it resolves.
- Low-card behavior is automated in this project, so deliberately playing a very low card can sometimes be a controlled row reset if the lowest-penalty row is cheap.
- The objective is tournament rank, not only minimizing our expected bullheads. A strong method should consider both self-risk and relative score position.
- A practical agent should have a hard time budget. Any sampling/search method needs an early cutoff and a deterministic fallback.

## Related Paper Takeaways

- AlphaGo combines policy/value networks, self-play reinforcement learning, and Monte Carlo tree search. The main lesson for this project is the hybrid pattern: use learned or hand-designed value estimates to make search/rollouts much cheaper. Full AlphaGo-style training is too large for this assignment.
- AlphaGo Zero and AlphaZero show that self-play can learn strong game policies from rules alone. For 6 Nimmt, the transferable idea is offline self-play against diverse agents; the risky part is relying on a large neural net or heavy per-move MCTS under a 1-second CPU-only limit.
- MuZero learns a model for planning instead of assuming known rules. This is less necessary here because the assignment engine rules are known, but it supports the idea of learning compact value/policy approximations for planning.
- RLCard is directly relevant because it targets reinforcement learning for imperfect-information card games and discusses state/action/reward design. We should not import RLCard unless it is approved, but its design ideas transfer to a custom 6 Nimmt environment.
- CFR, MCCFR, NFSP, and Deep CFR are relevant to imperfect-information games such as poker. They are more principled than plain PPO for hidden information, but full-game 6 Nimmt abstraction would be a large research project; a local-round abstraction is more realistic.
- Pluribus-style multiplayer poker work is relevant because 6 Nimmt is multiplayer and not two-player zero-sum. The practical lesson is to validate against a mixture of opponents and avoid assuming a single exploitable opponent model.
- The most practical direction from the papers is not "copy AlphaZero"; it is "combine a small evaluator, self-play/tournament data, and shallow/capped search." That fits the 1-second, single-threaded, CPU-only constraints much better.

## Candidate Methods

| Method | Core idea | Pros | Cons / risks | Novelty | Feasibility |
| --- | --- | --- | --- | --- | --- |
| Pure rule-based heuristic | Handcraft priorities such as play safe low cards early, avoid sixth-card traps, avoid high-bullhead captures, and preserve flexible cards. | Fast, transparent, safe under 1 second, easy to explain in report. | Limited adaptation; likely weaker than methods that model opponents and future rounds. | Low. | Very high. Good baseline and fallback. |
| One-step tactical expected value | For each card, simulate placement under different opponent-card order assumptions and estimate immediate penalty plus risk. | Still fast; directly matches engine rules; strong first improvement over fixed heuristics. | Myopic; can sacrifice long-term hand quality. | Low-medium if feature design is original. | Very high. |
| Risk-aware greedy policy | Score each card with expected penalty, worst-case penalty, row-fill risk, high-bullhead exposure, and score-position pressure. | Robust in tournaments; useful when avoiding catastrophic rank losses matters. | Requires careful weight tuning; may become too conservative. | Medium. | Very high. |
| Deterministic low-card reset strategy | Intentionally play below all rows when the automated chosen row has small penalty or when resetting a dangerous row helps future cards. | Exploits an assignment-specific rule; can convert weak low cards into controlled penalties. | Can backfire if opponents' lower cards resolve first or if row choice changes before placement. | Medium-high because it uses project-specific deterministic row taking. | High. |
| Endgame exact search | In the final 1-3 rounds, enumerate our remaining cards and sample or enumerate plausible opponent plays. | Endgames have small branching factor; can be reliable within time. | Still imperfect-information; early/mid game not covered. | Medium. | High. |
| Probabilistic opponent-card sampling | Infer unseen deck from public cards and our hand, sample possible opponent hands, and evaluate our cards across those samples. | Handles hidden information without huge tables; easy to combine with heuristic evaluation. | Sampling quality depends on count/time budget; assumes opponents are random unless modeled. | Medium. | High if sample count is capped. |
| Rollout / Monte Carlo simulation | For each candidate card, sample hidden hands and simulate remaining rounds using simple policies for all players. | Captures future consequences and hand value; flexible. | Needs heavy optimization for 1 second; rollout policy bias matters. | Medium. | Medium-high with small samples and fallback. |
| Perfect-information Monte Carlo (PIMC) | Sample complete hidden hands, treat each as known, solve or roll out, then average action values. | Simple imperfect-information adaptation; often useful in card games. | Strategy fusion risk: decisions may rely on information we would not actually know. | Medium. | Medium. |
| Information Set MCTS (ISMCTS) | Build a search tree over information sets rather than exact hidden states, using determinization per iteration. | Literature-matched for hidden-information games; more principled than PIMC. | Complex to implement correctly for simultaneous moves and 1-second budget. | High. | Medium-low for first implementation, possible as later experiment. |
| UCT / MCTS over abstracted states | Use UCT selection with sampled opponent plays and heuristic rollouts on a simplified game model. | Can outperform plain rollouts by focusing samples on promising actions. | Tree overhead may be too high because each turn has up to 10 cards and simultaneous opponents. | Medium-high. | Medium. |
| Local simultaneous-game solver | For the current round, build a payoff matrix over our cards and sampled opponent card distributions, then choose a minimax/Nash/softmax robust action. | Directly addresses simultaneous reveal; can be much cheaper than full-game search. | Needs payoff approximation; multiplayer general-sum solving is not trivial. | High if adapted well to 6 Nimmt. | Medium-high. |
| Level-k / quantal-response modeling | Assume opponents are level-0 random, level-1 heuristic, level-2 responding to heuristics, etc.; choose best response to a mixture. | Good for mixed student/baseline pools; interpretable opponent assumptions. | Poor if assumptions are wrong; needs calibration. | Medium-high. | High. |
| Opponent style classifier | Use public action history to classify each opponent as random, greedy, low-first, high-first, risk-seeking, etc., then adapt samples/rollouts. | Helps against diverse tournaments; uses available history without illegal state. | Only 10 rounds, so little data per game; classifier must be simple. | Medium. | High. |
| Portfolio / meta-policy | Maintain several sub-policies and choose among them by game phase, score position, and opponent model. | Fits the two-player submission idea; robust against overfitting to one baseline. | More moving parts; needs careful testing to avoid unstable switches. | Medium-high. | High. |
| Offline weight optimization | Parameterize a heuristic/rollout evaluator, then tune weights via random search, simple genetic algorithms, or a small custom optimizer over tournaments. | Produces small fast agents; no model-download issue if weights are embedded. | Can overfit released baselines or seeds; needs many local simulations. Extra optimizer packages should not be imported by final submitted code unless approved. | Medium. | High. |
| Contextual bandit over heuristics | During a game or across offline training, learn which heuristic variant works for board/score contexts. | Lightweight alternative to full RL; can be converted into static learned weights. | Online learning within one 10-round game has too little feedback. | Medium. | Medium-high offline, low-medium online. |
| Tabular Q-learning on abstract features | Discretize board/hand/score features and learn action values from self-play simulations. | Simple, fast inference; explainable feature buckets. | State abstraction may lose too much information; table can grow if not controlled. | Medium. | Medium. |
| Supervised imitation / behavior cloning | Generate a dataset from a strong search or heuristic teacher, then train a small model to imitate it. | Fast inference; can compress expensive search into a small policy. | Teacher quality caps performance; model file must be submitted and fit memory/time constraints. | Medium. | Medium-high if model is tiny. |
| PPO / actor-critic self-play | Build a Gymnasium/PettingZoo-style environment and train a policy with PPO against self-play and baselines. | Allowed packages include Gymnasium, PettingZoo, Stable-Baselines3, SB3-Contrib, Torch, and CleanRL; PPO is a standard robust baseline. | Training environment work is nontrivial; pure self-play may learn brittle strategies; inference must be small and CPU-safe. | Medium-high. | Medium. |
| Recurrent PPO / LSTM policy | Use history-aware recurrent policy to infer opponent style and hidden state from public sequence data. | The hidden-information nature makes memory useful. | More risk of timeout/model-size issues; recurrent inference and state handling add implementation complexity. | High. | Medium-low unless kept very small. |
| DQN / masked action-value network | Encode state features and predict value for each card action, masking cards not in hand. | Direct discrete-action fit; fast once trained. | Action set changes by hand; simultaneous multiplayer and partial observability can destabilize learning. | Medium. | Medium. |
| Neural Fictitious Self-Play (NFSP) | Combine a best-response learner with an average-policy learner for imperfect-information self-play. | More theoretically aligned with imperfect-information games than plain DQN/PPO. | Larger implementation and training burden; likely overkill for 10-round 6 Nimmt under course constraints. | High. | Low-medium. |
| Counterfactual Regret Minimization (CFR/MCCFR) | Solve an abstracted extensive-form version via regret minimization, possibly with Monte Carlo sampling. | Strong foundation for imperfect-information games. | Four-player non-zero-sum scoring and huge card state require heavy abstraction; real-time use is hard. | High. | Low for full game, medium for a local-round abstraction. |
| AlphaZero-lite hybrid | Train a small policy/value model from self-play and use it to guide shallow rollouts or MCTS. | Potentially strong and report-worthy if successful. | Much more engineering than a heuristic/rollout agent; AlphaZero assumes perfect-information turn games, so adaptation is nontrivial. | High. | Low-medium. |
| Opening/endgame book from simulations | Precompute common first-turn and late-game patterns, then use a lookup or compressed table. | Very fast inference; can be combined with heuristics. | Many possible hands/boards; exact lookup will be sparse. | Medium if compressed by features. | Medium. |
| Rank-aware objective shaping | Optimize expected rank or probability of not finishing last instead of raw bullheads. | Aligns with grading tournament rank; can choose different risk levels when leading/trailing. | Rank requires opponent score modeling; can choose locally odd moves. | Medium-high. | High as an evaluator feature. |
| Adversarial/counter-baseline tuning | Train or tune specifically against released baselines and adversarial synthetic styles, while holding out seeds/opponents. | Practical for class tournament; improves robustness if validation is honest. | Overfitting risk; released baselines are only a subset and cannot be imported directly. | Medium. | High. |

## Human-Strategy-Derived Rule-Based Variants

The two human-strategy sources point to three broad families that are realistic under the assignment constraints:

- low-first play order
- high-first / blocker play order
- board-aware "play close to the table" behavior

The important adaptation is that the paper assumes standard 6 Nimmt low-card choice, while this assignment engine forces the lowest-penalty row on low cards. So these methods should not blindly copy human play; they should convert "take a cheap row on purpose" into an explicit deterministic-row calculation.

### Shared State Features For Any Rule-Based Player

Every variant below can reuse the same cheap features per candidate card:

- `target_row_idx`, `target_row_end`: the closest lower row end, or `-1` if the card is below every row.
- `row_score`: bullheads currently sitting in the target row.
- `row_len`: current number of cards in the target row.
- `gap`: `card - target_row_end` when the card fits somewhere.
- `open_slots_before_take`: `board_size_x - row_len`. In this engine, `0` means the next card takes the row.
- `intervening_count`: unseen cards strictly between `target_row_end` and `card`. This is a cheap proxy for "how many lower opponent cards can arrive before my card resolves."
- `low_reset_cost`: if `target_row_idx == -1`, the exact penalty of the deterministic row the engine would make us take.
- `played_card_score`: bullheads on the card itself. This matters for dump logic because once the card is on the table, someone else may later take it.
- `dangerous_row`: true if the row contains `55`, any multiple of `11`, or total row bullheads above a threshold such as `>= 5`.
- `hand_rank_pct`: percentile rank of the card inside our sorted hand.
- `run_middle_flag`: true when the card is part of a local run and has nearby hand neighbors on both sides.
- `outlier_flag`: true when the card is isolated far from the rest of our hand.
- `score_pressure`: normalized measure of whether we are leading, tied, or trailing on current bullheads.

These are all compatible with the current player API and match the existing helper style already used by [src/players/cfr/full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/cfr/full_cfr_player.py) and [src/players/alpha_zero_lite/alpha_zero_lite_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/alpha_zero_lite/alpha_zero_lite_player.py).

### Variant 1: Low-First Safe-Shed

Source inspiration:

- `strategy.txt`: get rid of small cards early; middle cards in runs are often safer.
- Ayala et al.: low-first players produced the best aggregate result in their small study.

How it should work:

1. Split the hand into early-, mid-, and late-game behavior by `len(hand)`.
2. In early game, heavily bias toward the lowest 30-40% of cards in hand, but only after a safety filter.
3. A card is "safe enough" if either:
   - it fits a row with `row_len <= 3` and `intervening_count < open_slots_before_take`, or
   - it is a low reset whose `low_reset_cost` is below a small threshold such as `<= 3`.
4. Among safe-enough cards, choose the smallest card, except:
   - prefer a `run_middle_flag` card over the absolute minimum if both are similarly safe, because that preserves flexibility on both sides of the run;
   - if two choices have similar safety, dump the one with larger `played_card_score`.
5. In late game, reduce the low-first bias and switch to pure minimum-risk selection, because forced low resets become more dangerous when only a few cards remain.

What makes it distinct:

- It is not "always play the smallest card." It is "shed low cards early only when the engine-specific reset cost and sixth-card risk are acceptable."

Likely strengths:

- Very fast.
- Directly grounded in the best-performing human archetype from the paper.
- Good baseline and fallback policy.

Likely weaknesses:

- Can become too passive when a slightly larger card would create a much safer board for the next two rounds.

### Variant 2: Closest-Fit Conservative

Source inspiration:

- Ayala et al.: the table-aware group tried to play close to current rows and often preferred shorter rows.
- `strategy.txt`: "safely nestle your cards into the middle of existing rows."

How it should work:

1. For each legal card, compute a board-fit score such as:
   `fit_score = gap + 4 * max(row_len - 2, 0) + 3 * intervening_count + 2 * row_score`
2. Reject or heavily penalize cards whose target row already has `row_len == 5`, unless they intentionally take a very cheap row.
3. Add a bonus when the target row has `row_len <= 2`, following the paper's "prefer short columns" idea.
4. Add a penalty when `dangerous_row` is true, especially for rows containing `55` or red cards.
5. If no fitting card looks acceptable, compare:
   - cheapest deterministic low reset
   - least-bad risky fit
   and take whichever has lower worst-case penalty.

What makes it distinct:

- This model cares more about local board geometry than play-order bias.

Likely strengths:

- Simple, interpretable, and directly tied to visible table structure.
- Strong candidate against random and weak heuristic opponents.

Likely weaknesses:

- Can be shortsighted about hand shape and future forced resets.

### Variant 3: High-First Blocker

Source inspiration:

- Ayala et al.: a smaller group played high cards early and low cards late with moderate success.
- `strategy.txt`: in the normal variant, use high cards early to close dangerous rows before getting trapped later.

How it should work:

1. In rounds 1-4, bias toward the top 30-40% of cards in hand.
2. Prefer high cards that do one of the following:
   - land safely on a high-bullhead row without taking it,
   - replace an outlier high card that will be awkward later,
   - create a board endpoint that makes opponents' low-mid cards harder to place.
3. Keep one or two low cards in reserve as emergency cheap resets, but do not preserve them if the current cheapest reset row is already expensive.
4. If several high cards are similarly risky, dump the one with the largest `played_card_score` first.
5. After the hand shrinks to about 5 cards, switch to a neutral risk scorer so the player does not blindly cling to the high-first plan.

What makes it distinct:

- It deliberately trades away early hand smoothness to avoid late-game row takes and to create more awkward board states for others.

Likely strengths:

- Different failure profile from low-first, which is useful if we eventually want two meaningfully different submission players.

Likely weaknesses:

- More volatile, and likely weaker than low-first unless the blocker rules are tuned carefully.

### Variant 4: Controlled Reset Sacrifice

Source inspiration:

- `strategy.txt`: taking a small penalty early can be better than taking a disaster later.
- The assignment engine's deterministic low-card rule makes cheap sacrifices exactly computable.

How it should work:

1. For every candidate card, simulate the immediate board after placement.
2. Mark a card as an intentional-sacrifice candidate if it either:
   - takes a deterministic low reset with `low_reset_cost <= cheap_reset_threshold`, or
   - takes a full row whose `row_score` is still cheap enough relative to current score pressure.
3. Only accept that sacrifice if the resulting board improves our future hand structure. A cheap proxy is:
   - count how many remaining hand cards become "safe enough" after the sacrifice;
   - require this count to increase by at least `2`, or require one very dangerous future card to disappear.
4. When leading, use a strict sacrifice threshold.
5. When trailing, allow a slightly larger sacrifice threshold to create volatility and avoid slow death by repeated medium penalties.

What makes it distinct:

- This is the best way to exploit the repo's deterministic low-card rule in a purely rule-based player.

Likely strengths:

- Can turn a forced future disaster into a controlled small loss.
- Strongly assignment-specific, which gives it novelty.

Likely weaknesses:

- Needs careful board-improvement heuristics or it may over-sacrifice.

### Variant 5: Leader-Punish Rank-Aware Blocker

Source inspiration:

- Ayala et al. conclude that low-first plus targeting strong players worked best.
- `strategy.txt` emphasizes score position, group psychology, and forcing disasters onto others.

How it should work:

1. Compute current rank pressure from `history["scores"]`.
2. If we are leading or near the lead:
   - play conservative safety-first rules;
   - avoid volatile blocker moves unless they are nearly free.
3. If we are trailing:
   - reward moves that leave behind dangerous rows for the table, for example rows with:
     - high row bullheads,
     - `row_len == 4` or `row_len == 5`,
     - awkward endpoints that split the visible board badly.
4. Add a "leader-punish" bonus to moves that worsen the board while the current leader still has many cards left. This is not player-specific targeting by hidden hand; it is table-level pressure based on score position.
5. Tie-break toward dumping high-bullhead cards into the board when we are behind, because increasing future pickup cost for others matters more than preserving our own clean rows.

What makes it distinct:

- Same raw board can receive different actions depending on tournament rank pressure.

Likely strengths:

- Better aligned with the course objective of average rank, not just expected bullheads.

Likely weaknesses:

- More aggressive behavior can backfire if the board-danger heuristic is not calibrated.

### Variant 6: Human-Strategy Portfolio Switcher

Source inspiration:

- The sources do not present one single universal rule. They suggest phase-dependent play: low-first, table-aware fit, tactical sacrifices, and score-aware aggression.

How it should work:

1. Use `Low-First Safe-Shed` in rounds 1-3.
2. Use `Closest-Fit Conservative` in rounds 4-6.
3. Overlay `Controlled Reset Sacrifice` whenever a cheap board reset improves at least two future cards.
4. Overlay `Leader-Punish Rank-Aware Blocker` only when trailing by a configured margin.
5. Fall back to a simple minimum-risk tie-break if no specialized rule clearly dominates.

What makes it distinct:

- This is the most likely "best practical rule-based player" because it combines the strongest human themes without needing search or training.

Likely strengths:

- Highest ceiling among the rule-only methods.
- Natural candidate for `BestPlayer1`.

Likely weaknesses:

- More moving parts, so debugging and ablation matter more.

## Best Rule-Based Implementation Order

If the next task is to implement rule-based agents only, the best order is:

1. `Low-First Safe-Shed`
2. `Closest-Fit Conservative`
3. `Controlled Reset Sacrifice`
4. `Human-Strategy Portfolio Switcher`
5. `High-First Blocker` as an intentionally different second player
6. `Leader-Punish Rank-Aware Blocker` after basic evaluation is working

## Recommended Research Direction For Later Coding

1. Build a safe heuristic evaluator first. This becomes the fallback for every future method and helps avoid timeouts.
2. Add project-specific tactical expected value: deterministic low-card row choice, sixth-card risk, bullhead values, and rank-aware score pressure.
3. Add capped opponent-card sampling and short rollouts. Use a hard time budget and fallback to the heuristic if sampling is cut short.
4. Tune evaluator weights offline with many tournaments against public baselines, random players, and synthetic strategy players. Keep a separate validation set of seeds/opponent mixtures.
5. For novelty, try one of these extensions after the robust agent works: local simultaneous-game solver, opponent-style mixture model, or a tiny neural imitation policy trained from the rollout agent.
6. If we use two submitted players later, make them meaningfully different. Example: `BestPlayer1` as robust expected-value rollout; `BestPlayer2` as rank-aware aggressive/opponent-model portfolio.

## Methods I Would Avoid As First Implementation

- Full ISMCTS, full CFR, NFSP, or AlphaZero-lite before a strong heuristic/rollout baseline exists. They are interesting but carry a high implementation/time-budget risk.
- Large Torch models. The spec warns about Torch timeouts and the final run is CPU-only.
- Any method needing downloads, web APIs, multiprocessing, threading, or large generated tables. These conflict with the final-evaluation environment or project restrictions.
- Bare `except:` wrappers around search/model code. If we need exception handling later, it must catch specific exceptions only.

## Validation Ideas For Later

- Evaluate average rank, not only average bullheads.
- Track timeout, exception, and disqualification counts as hard failure metrics.
- Compare against random, public baselines, low-first, high-first, greedy-safe, and intentionally adversarial synthetic agents.
- Use held-out random seeds and opponent mixtures to reduce overfitting.
- Report qualitative novelty through method design and quantitative novelty through ablations: heuristic only vs. sampling vs. opponent model vs. rank-aware objective.

## Sources Consulted

- Local: `Spec.txt`, `README.md`, `requirements.txt`, and read-only inspection of `src/engine.py`.
- Browne et al., "A survey of Monte Carlo tree search methods", IEEE Transactions on Computational Intelligence and AI in Games, 2012: https://research.monash.edu/en/publications/a-survey-of-monte-carlo-tree-search-methods/
- Cowling, Powley, and Whitehouse, "Information Set Monte Carlo Tree Search", IEEE Transactions on Computational Intelligence and AI in Games, 2012: https://pure.york.ac.uk/portal/en/publications/information-set-monte-carlo-tree-search/
- Zinkevich et al., "Regret Minimization in Games with Incomplete Information", NeurIPS 2007: https://papers.neurips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information
- Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive Games", NeurIPS 2009: https://papers.nips.cc/paper_files/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html
- Heinrich, Lanctot, and Silver, "Fictitious Self-Play in Extensive-Form Games", ICML 2015: https://proceedings.mlr.press/v37/heinrich15
- Heinrich and Silver, "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games", arXiv 2016: https://arxiv.org/abs/1603.01121
- Silver et al., "Mastering the game of Go with deep neural networks and tree search", Nature 2016: https://www.nature.com/articles/nature16961
- Silver et al., "Mastering the game of Go without human knowledge", Nature 2017: https://www.nature.com/articles/nature24270
- Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017: https://arxiv.org/abs/1707.06347
- Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", arXiv 2017: https://arxiv.org/abs/1712.01815
- Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play", Science 2018: https://pubmed.ncbi.nlm.nih.gov/30523106/
- Brown and Sandholm, "Superhuman AI for multiplayer poker", Science 2019: https://pubmed.ncbi.nlm.nih.gov/31296650/
- Brown et al., "Deep Counterfactual Regret Minimization", ICML 2019: https://proceedings.mlr.press/v97/brown19b
- Lanctot et al., "OpenSpiel: A Framework for Reinforcement Learning in Games", arXiv 2019: https://arxiv.org/abs/1908.09453
- Zha et al., "RLCard: A Platform for Reinforcement Learning in Card Games", IJCAI 2020: https://www.ijcai.org/Proceedings/2020/764
- Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model", Nature 2020: https://www.nature.com/articles/s41586-020-03051-4
- Stable-Baselines3 PPO documentation, version 2.7.1: https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html
- SB3-Contrib Recurrent PPO documentation, version 2.7.1: https://sb3-contrib.readthedocs.io/en/v2.7.1/modules/ppo_recurrent.html
- PettingZoo documentation: https://pettingzoo.farama.org/
- Ayala et al., "The Effect of Different Strategies on Winning 6 Nimmt!", Journal of Research in Progress Vol. 5. This is game-specific but small-sample/informal, so I treat it as weak evidence only: https://pressbooks.howardcc.edu/jrip5/chapter/the-effect-of-different-strategies-on-winning-6-nimmt/
