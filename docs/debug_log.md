# Debug Log
Append-only. Record observations, hypotheses, conclusions, and next steps.

## 2026-04-08 CFR branch debug

Bug symptom:
- `FullCFRPlayer.action()` never reached its CFR search under the default 4-player, 10-round engine setup and always fell back to the heuristic path.
- Reproduced round-1 state: `unseen_cards=90`, `expected_hidden_cards=30`.

Root cause:
- [src/players/cfr/full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/cfr/full_cfr_player.py) required `len(unseen_cards) == (n_players - 1) * len(hand)` before running CFR.
- `_collect_unseen_cards()` returns every not-yet-visible card from the full 104-card deck.
- The engine only activates `n_players * n_rounds + board_size_y = 44` cards per game, so the unseen pool always includes many undealt cards.
- That made the equality check impossible under normal engine settings, so the CFR branch was unreachable.

Exact fix:
- Changed the guard in [src/players/cfr/full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/cfr/full_cfr_player.py) from exact equality to a sufficiency check.
- Before: fallback unless `len(unseen_cards) == expected_hidden_cards`
- After: fallback only if `len(unseen_cards) < expected_hidden_cards`
- This is sufficient because the existing sampler already draws only the number of hidden opponent cards it needs from the larger unseen pool.

Files changed:
- [src/players/cfr/full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/cfr/full_cfr_player.py)
- [tests/test_full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_full_cfr_player.py)

Verification run:
- `python -m pytest -q tests/test_full_cfr_player.py` -> `1 passed in 0.05s`
- Direct runtime probe under `n_players=4`, `n_rounds=10` confirmed `_solve_sampled_round()` is now called.

## 2026-04-08 CFR row-helper refactor

Investigation target:
- The duplicated row-selection logic in `_place_card()` and `_card_risk_snapshot()` was a maintainability risk because both methods had to stay aligned with the engine's tie-break rules.

Observation:
- A first-pass shared helper that computed row scores for every row on every call preserved results but slowed a targeted `_heuristic_scores()`/`_profile_utilities()` benchmark by about `29%`.
- The hot path only needs row scores when resolving the low-card fallback or a full target row.

Conclusion / final change:
- Split the refactor in [src/players/cfr/full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/cfr/full_cfr_player.py) into `_best_row_for_card()` for normal placement search and `_lowest_penalty_row()` for the low-card tie-break.
- Added a precomputed `self._card_scores` table so repeated bullhead scoring in heuristics and simulation no longer reruns the modulus chain.
- Side-by-side randomized checks against the pre-change implementation matched exactly for `_place_card()`, `_card_risk_snapshot()`, `_heuristic_scores()`, and `_profile_utilities()` across `250` cases.

Why it matters:
- Future CFR work can reuse the shared helpers without duplicating engine row rules, but should avoid eagerly scoring every row in the common path.

## 2026-04-08 Baseline test import and exit-code debug

Bug symptom:
- `python run_tournament.py --config configs/tournament/example_2.json` failed before any games started.
- Under local `Python 3.10.12`, the baseline import path surfaced as `No module named 'src.players.TA.public_baselines1'`.
- The tournament CLI printed the traceback but still exited with status `0`, which could make automation treat the failed baseline test as a pass.

Root cause:
- The only shipped TA public baseline artifact is [src/players/TA/public_baselines1.cpython-313-x86_64-linux-gnu.so](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/TA/public_baselines1.cpython-313-x86_64-linux-gnu.so).
- Normal `importlib.import_module("src.players.TA.public_baselines1")` on `Python 3.10` cannot match a `cpython-313` extension filename, so the real environment mismatch was misreported as `ModuleNotFoundError`.
- The earlier low-level `undefined symbol: _PyIntrinsics_UnaryFunctions` observation is the same incompatibility seen through a forced binary load path.
- [run_tournament.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/run_tournament.py) caught startup exceptions and returned normally, leaving the shell exit code at `0`.

Exact fix:
- Added compiled-module version mismatch detection in [src/game_utils.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/game_utils.py) so the loader now raises an explicit `ImportError` naming the shipped baseline binary and the required CPython version.
- Updated [run_tournament.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/run_tournament.py) to return `1` on configuration, tournament, or save failures and `0` on success, with `sys.exit(run())` at the CLI entrypoint.
- Added [tests/test_game_utils.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_game_utils.py) to lock in the clearer mismatch message on non-3.13 interpreters.

Verification run:
- `python -m pytest -q tests/test_game_utils.py tests/test_full_cfr_player.py` -> `3 passed in 0.17s`
- Baseline smoke validation via subprocess:
  - Command: `python run_tournament.py --config configs/tournament/example_2.json`
  - Result: exit status `1`
  - Message now states that `src.players.TA.public_baselines1` is only available as `public_baselines1.cpython-313-x86_64-linux-gnu.so` for `CPython 3.13`, while the current interpreter is `CPython 3.10.12`.

Remaining risk:
- The local workspace still cannot run the public TA baselines until a matching `python3.13` interpreter is available.

## 2026-04-08 AlphaZero draft implementation

Task:
- Add a minimal `src/players/alpha_zero` player package that follows the repo's player-loading convention and stays within the requested scope.

Implementation:
- Added [src/players/alpha_zero/alpha_zero_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/alpha_zero/alpha_zero_player.py) with an `AlphaZeroPlayer` that combines:
  - a heuristic policy head that scores each legal card from row risk, likely take penalty, gap size, row length, and bullhead dump pressure
  - a heuristic value head that converts expected action quality plus current rank/score pressure into a bounded scalar
  - a depth-limited PUCT search that samples hidden opponent hands from the unseen-card pool and searches our own future actions under those determinizations
- Added lightweight course-style aliases in [src/players/alpha_zero/best_player1.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/alpha_zero/best_player1.py), [src/players/alpha_zero/best_player2.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/alpha_zero/best_player2.py), and [src/players/alpha_zero/__init__.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/alpha_zero/__init__.py).

Important implementation note:
- This is an AlphaZero-lite draft, not a trained neural AlphaZero reproduction. The "policy/value" pieces are bounded heuristic evaluators, which keeps the method CPU-only, single-threaded, and self-contained for the assignment environment.
- Search opponent actions are modeled by the same policy heuristic inside each sampled state. Future improvement work should tune those heuristics and budgets before expanding the tree depth.

Validation:
- `python -m py_compile src/players/alpha_zero/alpha_zero_player.py src/players/alpha_zero/best_player1.py src/players/alpha_zero/best_player2.py src/players/alpha_zero/__init__.py` passed.
- Seeded smoke game:
  - Engine config: `{"n_players": 4, "n_rounds": 10, "seed": 7}`
  - Lineup: `AlphaZeroPlayer` + `3` `RandomPlayer`s
  - Result: `scores=[8, 8, 25, 20]`, `timeouts={}`, `exceptions={}`, `dq=[]`

## 2026-04-08 AlphaZero-lite rename

Task:
- Rename the new player package and visible descriptions from `alpha_zero` to `alpha_zero_lite`.

Change:
- Moved the package from `src/players/alpha_zero` to [src/players/alpha_zero_lite](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/alpha_zero_lite).
- Renamed the core module to [src/players/alpha_zero_lite/alpha_zero_lite_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/alpha_zero_lite/alpha_zero_lite_player.py).
- Renamed the main class to `AlphaZeroLitePlayer`.
- Kept `AlphaZeroPlayer` as an in-package alias so direct class imports from the new package can still use the shorter name if needed.

Why:
- The implementation is a heuristic policy/value plus bounded search draft, so `alpha_zero_lite` is a more accurate label than `alpha_zero`.

Validation:
- `python -m py_compile src/players/alpha_zero_lite/alpha_zero_lite_player.py src/players/alpha_zero_lite/best_player1.py src/players/alpha_zero_lite/best_player2.py src/players/alpha_zero_lite/__init__.py` passed.
- Seeded smoke game using `from src.players.alpha_zero_lite import AlphaZeroLitePlayer` completed with `scores=[8, 4, 20, 16]`, `timeouts={}`, `exceptions={}`, `dq=[]`.

## 2026-04-08 AlphaZeroLite late-game search debug

Target:
- `src/players/alpha_zero_lite`

Symptom:
- In sampled late-game states with only `3` cards left, `AlphaZeroLitePlayer` could choose a move that was strictly worse than another move under its own deterministic sampled-game model.
- Reproduced default-parameter example:
  - board `((1, 91, 94), (3,), (2, 67), (89,))`
  - scores `(1, 1, 8, 7)`
  - sampled hands `((7, 53, 79), (14, 37, 43), (13, 17, 69), (10, 25, 63))`
  - exact action values `{7: 0.5488029060285571, 53: 0.22918899653509897, 79: 0.7162978701990245}`
  - old default action: `7`
  - exact best action: `79`

Reproduction:
- Constructed the above `history` / sampled-hand state directly and monkeypatched `_sample_hidden_hands()` to return the fixed determinization.
- Compared `player.action()` against an exact recursive solve that uses the same `_step_state()` / `_terminal_value()` model as the player.

Findings:
- The current implementation kept using bounded MCTS even when the sampled subtree was tiny enough to solve exactly.
- With only `20` default simulations at hand size `3`, the root search stayed heavily prior-dominated and often underexplored better actions.
- In the reproduced state, the root heuristic prior was strongest on `7`, so MCTS visited `7` far more often and returned it even though exact recursive evaluation favored `79`.

Root cause:
- Late sampled endgames were still using approximate MCTS instead of exact search, so tractable `3`-card deterministic subtrees could return a strictly suboptimal move under the player's own evaluation model.

Fix:
- Added exact recursive action evaluation for sampled states when our current hand size is `3` or less in `AlphaZeroLitePlayer.action()`.
- The new helper computes exact root action values with memoized recursion over `_step_state()` / `_terminal_value()` and aggregates those exact values across sampled determinizations instead of running MCTS in that late-game regime.

Validation:
- `python -m pytest -q tests/test_alpha_zero_lite_player.py` -> `1 passed in 0.02s`
- `python -m pytest -q tests/test_alpha_zero_lite_player.py tests/test_full_cfr_player.py tests/test_game_utils.py` -> `4 passed in 0.44s`
- Ad hoc strict-best sanity check on `20` random fixed determinized `3`-card states found `0` cases where the patched player chose an action outside the exact-best set.

Status:
- Fixed with targeted regression coverage.

## 2026-04-10 Rule-based human-strategy draft implementation

Task:
- Draft five separate rule-based player packages based on the human-strategy research pass, without introducing `BestPlayer1` / `BestPlayer2` aliases yet.

Implementation:
- Added a shared helper module at [src/players/rule_based_player_base.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_based_player_base.py) that centralizes:
  - bullhead scoring
  - engine-aligned row targeting
  - deterministic low-card row selection
  - per-card feature extraction
  - simple board-danger and future-safe-card heuristics
- Added these five standalone player packages under `src/players/`:
  - [src/players/rule_low_first_safe_shed](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_low_first_safe_shed)
  - [src/players/rule_closest_fit_conservative](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_closest_fit_conservative)
  - [src/players/rule_high_first_blocker](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_high_first_blocker)
  - [src/players/rule_controlled_reset_sacrifice](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_controlled_reset_sacrifice)
  - [src/players/rule_leader_punish_rank_aware](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_leader_punish_rank_aware)
- Added [tests/test_rule_based_players.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_rule_based_players.py) to lock in:
  - package loading through `load_players()`
  - one crafted decision test per new player
  - a small engine-state legality smoke check for all five players

Implementation note:
- This pass interprets the user's "five players" request as the first five standalone rule-based variants from the research document: low-first, closest-fit, high-first, controlled-reset, and leader-punish. The portfolio switcher variant was intentionally left out of this draft pass.
- The controlled-reset and rank-aware variants are still one-step heuristics. They simulate board changes from our own card placement only and do not yet sample opponent hidden hands or simultaneous-play outcomes.

Validation:
- `python -m pytest -q tests/test_rule_based_players.py` -> `7 passed in 0.32s`
- `python -m pytest -q tests/test_rule_based_players.py tests/test_game_utils.py` -> `9 passed in 0.32s`

## 2026-04-10 Focused human-strategy portfolio implementation

Task:
- Implement the eval-guided portfolio version of the human-strategy switcher, but keep it focused on the strongest rule-based modes instead of combining all five standalone variants equally.

Implementation:
- Added [src/players/rule_human_strategy_portfolio](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_human_strategy_portfolio) with `HumanStrategyPortfolioPlayer`.
- The portfolio uses:
  - `HighFirstBlocker`-style scoring as the default/base mode
  - `LeaderPunishRankAware`-style trailing logic when score pressure is high
  - a narrow `ControlledResetSacrifice`-style override only when a cheap self-penalty improves future hand safety by at least `2`
- Extended [tests/test_rule_based_players.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_rule_based_players.py) to cover:
  - package loading
  - HFB-like leading/default behavior
  - LPRA-like trailing behavior
  - the cheap-reset override path
  - engine-state legality for the new portfolio player

Implementation note:
- This is intentionally not the original five-way Variant 6 from the research note. The switcher is constrained to the modes that current evals actually supported: `HFB` and `LPRA`, plus a very small `CRS` override.
- Validation had to be run in the course conda env instead of the active base interpreter because the active base interpreter lacked both `pytest` and `joblib`.

Validation:
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python - <<'PY' ... import tests.test_rule_based_players ...` -> `10 rule-based tests passed via direct invocation in FAI_final`
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python -m unittest tests.test_game_utils` -> `OK (skipped=1)`

## 2026-04-10 ISMCTS and NFSP draft implementation

Task:
- Add minimal `src/players/ISMCTS` and `src/players/NFSP` player packages that fit the current repo contract and stay within the assignment-safe bounded-search scope described in `AGENT_METHOD_RESEARCH.md`.

Implementation:
- Added [src/players/ISMCTS](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/ISMCTS) with `ISMCTSPlayer`.
- [src/players/ISMCTS/ismcts_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/ISMCTS/ismcts_player.py) uses:
  - hidden-hand determinization from the unseen-card pool
  - a shallow information-set keyed UCT tree over our own legal actions
  - heuristic default policies for opponent/action rollout steps
  - exact recursive evaluation for sampled endgames when our current hand size is `3` or less
- Added [src/players/NFSP](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/NFSP) with `NFSPPlayer`.
- [src/players/NFSP/nfsp_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/NFSP/nfsp_player.py) mixes:
  - an average-policy scorer built from the shared engine-aligned candidate features plus a reconstructed aggression target from each player's observed `history_matrix` actions
  - a sampled best-response evaluator that assumes opponents follow that average policy and exact-solves sampled `3`-card endgames
- Added [tests/test_ismcts_nfsp_players.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_ismcts_nfsp_players.py) to lock in:
  - package loading through `load_players()`
  - ISMCTS exact sampled endgame selection
  - NFSP score-pressure-dependent action switching in a fixed public state
  - engine-state legality for both players

Implementation note:
- These are bounded ISMCTS-style / NFSP-style drafts, not full literature-faithful reproductions with training, large trees, or persistent learned models. That is intentional: `AGENT_METHOD_RESEARCH.md` explicitly calls full ISMCTS and full NFSP high-risk for this assignment and recommends smaller local-round abstractions first.
- Validation had to use direct test invocation because the available local Python interpreters in this workspace currently do not have `pytest` installed.

Validation:
- `python -m py_compile src/players/ISMCTS/ismcts_player.py src/players/ISMCTS/__init__.py src/players/NFSP/nfsp_player.py src/players/NFSP/__init__.py tests/test_ismcts_nfsp_players.py` passed.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python - <<'PY' ... import tests.test_ismcts_nfsp_players ...` passed all `4` targeted checks by direct invocation.
- Seeded smoke games:
  - `ISMCTSPlayer + 3 RandomPlayer`s with `Engine({"n_players": 4, "n_rounds": 10, "seed": 7})` -> `scores=[13, 19, 0, 11]`, `timeouts={}`, `exceptions={}`, `dq=[]`
  - `NFSPPlayer + 3 RandomPlayer`s with the same engine config -> `scores=[24, 3, 7, 10]`, `timeouts={}`, `exceptions={}`, `dq=[]`

## 2026-04-10 ISMCTS redesign and NFSP / portfolio improvement pass

Task:
- Improve `ISMCTSPlayer`, `NFSPPlayer`, and `HumanStrategyPortfolioPlayer`, with the main emphasis on redesigning MCTS while preserving the intended behavior of the other two players.

Implementation:
- Redesigned [src/players/ISMCTS/ismcts_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/ISMCTS/ismcts_player.py) into a root-focused sampled search:
  - heuristic policy utilities are now the default action selector and prior source
  - bounded search is only activated in late, risky, or ambiguous states via `_should_search()`
  - the search budget was reduced to `search_time_limit=0.45`, `max_samples=4`, and `simulations_per_sample=12`
  - sampled endgames now exact-solve when our remaining hand size is `4` or less
- Refined [src/players/NFSP/nfsp_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/NFSP/nfsp_player.py):
  - added `self._average_utility_cache` so repeated average-policy evaluations within one action call are reused
  - extended sampled exact endgame solving to hand size `4` or less for both root action values and deeper best-response states
  - kept the lightweight average-policy / best-response mixture after a heavier phase-aware attempt proved too slow
- Simplified [src/players/rule_human_strategy_portfolio/human_strategy_portfolio_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_human_strategy_portfolio/human_strategy_portfolio_player.py) back to the confirmed strong core:

## 2026-04-11 Imitation-lite draft implementation

Task:
- Add a minimal imitation-model player package under `src/players/` without changing existing player behavior outside that new package.

Implementation:
- Added [src/players/imitation_lite](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/imitation_lite) with:
  - [src/players/imitation_lite/imitation_lite_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/imitation_lite/imitation_lite_player.py)
  - [src/players/imitation_lite/__init__.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/imitation_lite/__init__.py)
  - [src/players/imitation_lite/best_player1.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/imitation_lite/best_player1.py)
  - [src/players/imitation_lite/best_player2.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/imitation_lite/best_player2.py)
- Added [tests/test_imitation_lite_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_imitation_lite_player.py) to cover:
  - package loading through `load_players()`
  - `ImitationPlayer` alias wiring
  - an aggressive opening-style crafted state
  - a trailing-pressure crafted state
  - a cheap-reset crafted state
  - engine-state legality

Important implementation note:
- This is a heuristic distillation draft over the confirmed teacher pool signals, not a trained imitation model with saved weights or an offline dataset.
- The implementation intentionally does not call `CFR`, `AZL`, `HSP`, or other teacher players inside `action()`. Future agents should keep that separation unless they have hard timing evidence, because stacking multiple full teacher policies inside one move would create avoidable timeout risk under the 1-second decision budget.
- The only scoring tweak needed after first-pass implementation was broadening the "aggressive opening" bonus to also trigger on early-round crafted states where `history["round"] <= 1`, because some targeted tests use short hands in round-0-style scenarios.

Validation:
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python -m py_compile src/players/imitation_lite/imitation_lite_player.py src/players/imitation_lite/__init__.py src/players/imitation_lite/best_player1.py src/players/imitation_lite/best_player2.py tests/test_imitation_lite_player.py` passed.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python - <<'PY' ... import tests.test_imitation_lite_player ...` passed all `6` targeted checks by direct invocation.
- Seeded smoke game:
  - `ImitationLitePlayer + 3 RandomPlayer`s with `Engine({"n_players": 4, "n_rounds": 10, "seed": 7})` -> `scores=[2, 10, 33, 7]`, `timeouts={}`, `exceptions={}`, `dq=[]`

## 2026-04-11 Cross-play training draft implementation

Task:
- Draft end-to-end cross-play data generation, offline training, and runnable-player code grounded in the confirmed top teacher pool `CFR`, `HSP`, `HFB`, `AZL`, and `LPRA`.

Implementation:
- Added [src/training/crossplay_learning.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/training/crossplay_learning.py) and [src/training/__init__.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/training/__init__.py).
- The new training module provides:
  - `build_crossplay_lineups()` for 4-player cross-play scheduling from the 5-teacher pool
  - `generate_crossplay_examples()` for logged teacher-vs-teacher game data
  - `CrossPlayFeatureEncoder`, which reuses the shared engine-aligned feature surface from [src/players/rule_based_player_base.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/rule_based_player_base.py)
  - JSONL dataset save/load helpers
  - a tiny rank-weighted linear policy trainer and model scorer
- Added runnable CLI entrypoints [scripts/generate_crossplay_data.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/scripts/generate_crossplay_data.py) and [scripts/train_crossplay_model.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/scripts/train_crossplay_model.py).
- Added [src/players/crossplay_imitation](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/crossplay_imitation) with:
  - [src/players/crossplay_imitation/crossplay_imitation_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/crossplay_imitation/crossplay_imitation_player.py)
  - [src/players/crossplay_imitation/__init__.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/crossplay_imitation/__init__.py)
  - [src/players/crossplay_imitation/best_player1.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/crossplay_imitation/best_player1.py)
  - [src/players/crossplay_imitation/best_player2.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/crossplay_imitation/best_player2.py)
  - [src/players/crossplay_imitation/crossplay_model.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/crossplay_imitation/crossplay_model.json)
- Added [tests/test_crossplay_learning.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_crossplay_learning.py) to cover:
  - player-package loading
  - alias wiring
  - teacher-lineup coverage
  - logged cross-play example generation
  - simple trainer fitting behavior
  - learned-player engine-state legality

Important implementation notes:
- The cross-play generator uses all `5 choose 4` mixed-teacher lineups because the engine is 4-player. Seat rotations are supported but left off by default to keep the draft data pass bounded.
- `CFR` and `AZL` use lighter default search budgets inside `TEACHER_SPECS` for this data-generation pipeline than in their evaluation-oriented player defaults. This keeps offline data generation practical without calling heavyweight searches at full budget on every logged move.
- `CrossPlayImitationPlayer` loads the saved JSON model once at init and falls back to `ImitationLitePlayer` if the model file is missing or invalid, so the package remains runnable even when the learned artifact has not been generated yet.
- Direct script execution from `scripts/` needed a small repo-root `sys.path` bootstrap because `python scripts/...` sets `sys.path[0]` to the script directory rather than the repository root.

Validation:
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python -m py_compile src/training/__init__.py src/training/crossplay_learning.py src/players/crossplay_imitation/__init__.py src/players/crossplay_imitation/crossplay_imitation_player.py src/players/crossplay_imitation/best_player1.py src/players/crossplay_imitation/best_player2.py scripts/generate_crossplay_data.py scripts/train_crossplay_model.py tests/test_crossplay_learning.py` passed.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python - <<'PY' ... import tests.test_crossplay_learning ...` passed all `6` targeted checks by direct invocation.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python scripts/generate_crossplay_data.py --output results/training/crossplay_examples.jsonl --games-per-lineup 1 --base-seed 11 --n-rounds 10` saved `180` examples from `5` teachers.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python scripts/train_crossplay_model.py --dataset results/training/crossplay_examples.jsonl --output src/players/crossplay_imitation/crossplay_model.json --epochs 180 --learning-rate 0.05 --l2 0.0001` trained a model with `180` decisions / `1080` candidates, best epoch `22`, train accuracy `0.7287058225273328`, and train loss `1.1308686064204114`.
- Seeded smoke game:
  - `CrossPlayImitationPlayer + 3 RandomPlayer`s with `Engine({"n_players": 4, "n_rounds": 10, "seed": 7})` -> `scores=[7, 7, 26, 14]`, `timeouts={}`, `exceptions={}`, `dq=[]`
  - default/base scoring is again pure `HighFirstBlocker`-style
  - trailing mode is again `LeaderPunishRankAware`-style
  - the cheap-reset override remains narrowly gated
  - removed the extra low-first / closest-fit phase switching that regressed the mini-benchmark
- Updated [tests/test_ismcts_nfsp_players.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_ismcts_nfsp_players.py) so the NFSP score-pressure regression test matches the final lightweight policy mix.

Benchmark:
- Reused the same small public-baseline config for before/after comparisons:
  - `random_partition`
  - `duplication_mode="cycle"`
  - `num_games_per_player=40`
  - lineup `candidate + Baseline4 + Baseline5 + RandomPlayer`, with baseline anchors `Baseline1`-`Baseline5`
- ISMCTS:
  - pre-change [results/tournament/2026-04-10_18-06-16_pre_ismcts_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-06-16_pre_ismcts_public_40.json) -> `avg_rank=2.35`, `avg_score=11.25625`, `est_elo=1528.82`
  - first ungated redesign [results/tournament/2026-04-10_18-18-31_pre_ismcts_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-18-31_pre_ismcts_public_40.json) regressed to `2.41875 / 11.175 / 1518.21`
  - final gated redesign [results/tournament/2026-04-10_18-22-44_pre_ismcts_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-22-44_pre_ismcts_public_40.json) improved to `2.240625 / 10.63125 / 1559.01`
- NFSP:
  - pre-change [results/tournament/2026-04-10_18-07-13_pre_nfsp_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-07-13_pre_nfsp_public_40.json) -> `avg_rank=2.21875`, `avg_score=10.09375`, `est_elo=1565.93`
  - heavier phase-aware attempt [results/tournament/2026-04-10_18-25-28_pre_nfsp_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-25-28_pre_nfsp_public_40.json) regressed to `2.29375 / 11.05625 / 1552.36` with `timeout_count=133`
  - final lighter version [results/tournament/2026-04-10_18-29-06_pre_nfsp_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-29-06_pre_nfsp_public_40.json) improved to `2.13125 / 9.8125 / 1582.26`
- HumanStrategyPortfolio:
  - pre-change [results/tournament/2026-04-10_18-06-35_pre_portfolio_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-06-35_pre_portfolio_public_40.json) -> `avg_rank=2.209375`, `avg_score=10.28125`, `est_elo=1559.98`
  - broader phase-aware variant [results/tournament/2026-04-10_18-28-42_pre_portfolio_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-28-42_pre_portfolio_public_40.json) regressed to `2.3 / 11.825 / 1552.77`
  - final simplified portfolio [results/tournament/2026-04-10_18-32-18_pre_portfolio_public_40.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-10_18-32-18_pre_portfolio_public_40.json) improved to `2.15625 / 9.8875 / 1577.09`

Implementation note:
- The key outcome from this pass is that stronger local results came from tighter activation and simpler policy mixing, not from broader phase logic. The final portfolio deliberately stays narrow, and the final ISMCTS deliberately searches less often.

Validation:
- `python -m py_compile src/players/ISMCTS/ismcts_player.py src/players/NFSP/nfsp_player.py src/players/rule_human_strategy_portfolio/human_strategy_portfolio_player.py tests/test_ismcts_nfsp_players.py tests/test_rule_based_players.py` passed.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python - <<'PY' ... import tests.test_ismcts_nfsp_players, tests.test_rule_based_players ...` passed all `14` targeted checks by direct invocation.
- Seeded smoke games with `Engine({"n_players": 4, "n_rounds": 10, "seed": 7})` completed with zero `TO/EXC/DQ`:
  - `ISMCTSPlayer + 3 RandomPlayer`s -> `scores=[17, 1, 32, 2]`
  - `NFSPPlayer + 3 RandomPlayer`s -> `scores=[7, 12, 9, 12]`
  - `HumanStrategyPortfolioPlayer + 3 RandomPlayer`s -> `scores=[3, 15, 2, 13]`

## 2026-04-11 Cross-play training draft implementation end note

Confirmed end-state:
- Added `src/training/crossplay_learning.py`, `scripts/generate_crossplay_data.py`, `scripts/train_crossplay_model.py`, `src/players/crossplay_imitation`, and `tests/test_crossplay_learning.py`.
- Generated `results/training/crossplay_examples.jsonl` with `180` logged teacher decisions from `CFR`, `HSP`, `HFB`, `AZL`, and `LPRA`.
- Trained `src/players/crossplay_imitation/crossplay_model.json` with `180` decisions / `1080` candidates; best epoch `22`, train accuracy `0.7287058225273328`, train loss `1.1308686064204114`.
- `CrossPlayImitationPlayer` smoke-tested cleanly with `scores=[7, 7, 26, 14]`, `timeouts={}`, `exceptions={}`, `dq=[]`.

## 2026-04-11 Weighted diversity cross-play update

Task:
- Extend the draft cross-play pipeline so future runs can use a much larger mixed-label dataset while down-weighting weaker diversity sources.

Implementation:
- Updated [src/training/crossplay_learning.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/training/crossplay_learning.py) to support:
  - optional extra player labels `LFS`, `CFC`, `CRS`, `IMIT`, `RND`, `B4`, and `B5`
  - label-specific `example_weight` values recorded into generated examples
  - bounded lineup sampling via `max_lineups` plus deterministic `lineup_seed`
  - training weights that now multiply `example_weight` into the existing `1 / final_rank` weighting
- Updated [scripts/generate_crossplay_data.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/scripts/generate_crossplay_data.py) with:
  - `--teacher-labels`
  - repeated `--teacher-weight LABEL=FLOAT`
  - `--max-lineups`
  - `--lineup-seed`
- Updated [scripts/train_crossplay_model.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/scripts/train_crossplay_model.py) so the default epoch count is now `600`.
- Extended [tests/test_crossplay_learning.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_crossplay_learning.py) to cover deterministic lineup capping, recorded teacher weights, and trainer behavior under conflicting examples with different example weights.

Important implementation notes:
- This weighting is intentionally simple: final training weight is now `example_weight / final_rank`, so stronger finishing policies still matter more even when diversity labels are mixed in.
- `max_lineups` applies after optional rotation expansion. If more labels are added later, this cap is the main guardrail against combinatorial blow-up.
- Baseline labels `B4` and `B5` are available for users in the working `FAI_final` environment, but they are optional and should not be treated as required for every local run.

Validation:
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python -m py_compile src/training/__init__.py src/training/crossplay_learning.py scripts/generate_crossplay_data.py scripts/train_crossplay_model.py tests/test_crossplay_learning.py` passed.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python - <<'PY' ... import tests.test_crossplay_learning ...` passed all `9` targeted checks by direct invocation.
- Weighted diversity generator smoke:
  - `/home/thoamslai/miniconda3/envs/FAI_final/bin/python scripts/generate_crossplay_data.py --output /tmp/crossplay_diverse_smoke.jsonl --teacher-labels CFR,HSP,HFB,AZL,LPRA,LFS,CFC,CRS,IMIT,RND --teacher-weight LFS=0.35 --teacher-weight CFC=0.35 --teacher-weight CRS=0.35 --teacher-weight IMIT=0.35 --teacher-weight RND=0.20 --games-per-lineup 1 --max-lineups 8 --base-seed 19 --n-rounds 2` saved `32` examples from `10` teachers with total example weight `22.30`.
- Weighted trainer smoke:
  - `/home/thoamslai/miniconda3/envs/FAI_final/bin/python scripts/train_crossplay_model.py --dataset /tmp/crossplay_diverse_smoke.jsonl --output /tmp/crossplay_diverse_smoke_model.json --epochs 40 --learning-rate 0.05 --l2 0.0001` trained with `32` decisions, `accuracy=0.8968`, and `loss=0.5094`.

## 2026-04-12 CrossPlayImitation large-run evaluation note

Observation:
- The user-trained large mixed-label run used `115200` logged decisions and `5000` epochs, yet the reported train accuracy only reached `0.6996`, which is very close to the earlier smaller-run `0.6969`.
- On the larger 40-partition mini public-baseline eval, `CrossPlayImitationPlayer` still beat `B4` and `B5` cleanly.
- On the 2000-partition shared-field run with `XPI` added to `example_3.json`, `CrossPlayImitationPlayer` finished `8th` at `avg_rank=2.414375`, essentially tied with the upper public-baseline cluster (`B4/B3/B5`) and clearly behind the top custom pack `CFR/HFB/HSP/AZL/LPRA`.

Important implementation risk:
- The current generator logs every included label as a training target. That means weaker diversity labels such as `LFS`, `CFC`, `CRS`, `IMIT`, and `RND` are not only present as opponents; they also directly provide target actions for the student.
- Given the large-run result, the next likely bottleneck is not simply lack of epochs. It is either target dilution from weaker labels, or limited expressive power in the current linear scorer over the shared handcrafted feature set.

Most practical next step:
- Run a size-matched top-five-only control dataset (`CFR,HSP,HFB,AZL,LPRA` only) before adding more epochs or more diversity labels. If that improves the shared-field result, the next code change should be teacher/opponent separation rather than a larger mixed-target dataset.

## 2026-04-12 Top-five-only control outcome

Observation:
- The top-five-only control dataset (`103680` decisions) and `5000`-epoch retrain reached train `accuracy=0.6994`, essentially unchanged from the larger mixed-label run (`0.6996`).
- On the matched 40-partition mini eval, `XPI_TOP5` regressed to `avg_rank=2.340625`, `avg_score=11.7125`, `est_elo=1528.50`, finishing behind `B4`.

Conclusion:
- The current evidence does not support the hypothesis that weaker mixed labels were the main reason the linear student plateaued.
- The more likely bottleneck is still student capacity or the limits of a single handcrafted feature surface with a shallow scorer.

Updated next step:
- Keep the mixed-label dataset as the better-performing data source.
- Improve the student architecture next, starting with a safe phase-specific scorer rather than another dataset-only retrain.

## 2026-04-12 Phase-specific cross-play student implementation

Task:
- Replace the single global linear cross-play scorer with a safer higher-capacity student while keeping the existing data format, CLI flow, and runtime player wiring stable.

Implementation:
- Updated [src/training/crossplay_learning.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/training/crossplay_learning.py) so `train_linear_policy()` now trains separate linear heads for `early`, `mid`, and `late` phases instead of one shared global weight vector.
- The saved model format now writes:
  - `model_type="phase_linear_policy"`
  - `phase_keys=["early", "mid", "late"]`
  - `weights_by_phase={...}`
  - per-phase metrics inside `training_summary["phase_summaries"]`
- The trainer still uses the existing normalized feature surface and rank/example weighting; the main change is that each logged decision is routed to a phase head using the already-logged `phase_early`, `phase_mid`, and `phase_late` features.
- Added a small `_phase_weight_vector()` compatibility helper so inference can score either the new `weights_by_phase` model format or the older legacy `weights` format.
- Updated [src/players/crossplay_imitation/crossplay_imitation_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/crossplay_imitation/crossplay_imitation_player.py) docstring wording only; the actual runtime player path did not need structural changes because it already delegates scoring through `score_feature_values()`.
- Updated [scripts/train_crossplay_model.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/scripts/train_crossplay_model.py) help text and [README.md](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/README.md) usage notes to describe the phase-aware training behavior without changing the CLI.
- Extended [tests/test_crossplay_learning.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_crossplay_learning.py) so it now checks:
  - the saved model exposes `weights_by_phase`
  - phase summaries are populated
  - conflicting early-vs-late synthetic preferences can be learned by different heads

Important implementation notes:
- CLI usage for `scripts/generate_crossplay_data.py` and `scripts/train_crossplay_model.py` is unchanged; retraining on the existing mixed dataset is enough to produce the new student.
- Backward compatibility is intentionally preserved at inference time: old single-head JSON models still load and score correctly.
- The main expected benefit is reducing policy averaging across the early, mid, and late parts of a game without jumping yet to a deeper neural model.

Validation:
- `python -m py_compile src/training/__init__.py src/training/crossplay_learning.py src/players/crossplay_imitation/crossplay_imitation_player.py scripts/train_crossplay_model.py tests/test_crossplay_learning.py` passed.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python - <<'PY' ... import tests.test_crossplay_learning ...` passed all `10` targeted checks by direct invocation.
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python scripts/train_crossplay_model.py --dataset results/training/crossplay_examples.jsonl --output /tmp/crossplay_phase_smoke_model.json --epochs 5 --learning-rate 0.05 --l2 0.0001` completed successfully and produced a `phase_linear_policy` model with the expected `early/mid/late` heads.

## 2026-04-13 CFR timeout-budget cap

Task:
- Reduce `FullCFRPlayer` timeout risk in the broad tournament setup without replacing its local-round CFR approach or changing the heuristic fallback path.

Investigation:
- The hot path is still `FullCFRPlayer.action()` -> `_solve_sampled_round()`, which enumerates every joint action profile for a sampled hidden-hand determinization.
- In 4-player round-start states, the raw profile count is `hand_size ** 4`, so the existing default search budgets were spending a lot of wall-clock on repeated determinizations in the early and mid game.
- Local pre-change timing probe under `/home/thoamslai/miniconda3/envs/FAI_final/bin/python` on `2` seeded `Engine({"n_players": 4, "n_rounds": 10})` games with lineup `FullCFRPlayer + 3 RandomPlayer`s showed these mean / max action times by hand size:
  - `10`: `739.35 ms / 755.17 ms`
  - `9`: `433.31 ms / 473.62 ms`
  - `8`: `535.06 ms / 774.81 ms`
  - `7`: `429.62 ms / 463.91 ms`
- Those local single-process timings are already close enough to the `1.0s` engine alarm that the shared-field `num_workers=40` tournament can plausibly push some CFR turns over the wall-clock limit even though `search_time_limit` is only `0.82s`.

Exact fix:
- Added `max_profile_work` and `min_iterations` parameters to [src/players/cfr/full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/cfr/full_cfr_player.py).
- Added `_resolve_search_budget(hand_size, n_players)` to cap total per-action CFR work by bounding `profile_count * sample_budget * iterations`.
- The new logic reduces sampled determinizations first and only then trims iteration count, which keeps the existing CFR solve shape but avoids the worst early-round work spikes.
- The default capped budgets for the 4-player game now resolve to:
  - `hand_size=10 -> (sample_budget=1, iterations=5)`
  - `9 -> (1, 7)`
  - `8 -> (1, 12)`
  - `7 -> (1, 14)`
  - `6 -> (2, 16)`
  - `5 -> (4, 19)`

Validation:
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python -m unittest tests.test_full_cfr_player` -> `Ran 2 tests in 0.002s`, `OK`
- Added a targeted regression in [tests/test_full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_full_cfr_player.py) that locks in the large-hand capped budget and verifies `action()` only launches one CFR solve on a round-1 `10`-card state.
- Post-change timing probe on the same `2` seeded games reduced mean / max action times to:
  - `10`: `279.71 ms / 302.04 ms`
  - `9`: `210.50 ms / 224.44 ms`
  - `8`: `182.76 ms / 221.75 ms`
  - `7`: `95.84 ms / 107.31 ms`

Remaining risk:
- This change is a robustness/performance tradeoff, not a proof that shared-field `TO` will fall to zero. It reduces CFR sampling depth on larger hands, so tournament-strength confirmation still needs a later multi-partition rerun in the user’s main evaluation setup.

## 2026-04-13 CFR late-phase-only gate after timeout regression

Task:
- Revisit the CFR timeout work after the user’s newer broad shared-field rerun showed the problem had become worse rather than better.

New evidence:
- The user-provided result [results/tournament/2026-04-13_13-37-09_example_3.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-13_13-37-09_example_3.json) showed `FullCFRPlayer` still finishing `1st` overall, but with `avg_rank=2.3125625`, `avg_score=11.39775`, `est_elo=1543.46`, `DQ=1`, and `TO=82`.
- That was worse on timeout robustness than the earlier saved shared-field result [results/tournament/2026-04-13_00-42-10_example_3.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-13_00-42-10_example_3.json), where `CFR` had `TO=29`.
- A matched pre-change mini rerun with the same broad field and `num_workers=40` at lower sample count, [results/tournament/2026-04-13_13-46-21_example_3.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-13_13-46-21_example_3.json), still recorded `CFR timeout_count=2` over `400` games.
- The tail-latency probe after the first budget cap still had its slowest actions at hand sizes `10` and `9`, with the worst sampled action at `329.24 ms`, so the remaining risk was still concentrated in the earliest rounds.

Updated diagnosis:
- The first fix reduced per-action work, but it still allowed CFR search on the `10`- to `7`-card rounds.
- Those early-round solves remained the slowest turns, and under the shared-field `40`-worker wall-clock contention they were still enough to trigger tournament timeouts.

Exact fix:
- Added `max_cfr_hand_size=6` to [src/players/cfr/full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/src/players/cfr/full_cfr_player.py).
- `_resolve_search_budget()` now returns `(0, 0)` when `hand_size > max_cfr_hand_size`.
- `action()` now treats a zero search budget as an explicit heuristic-only fallback, so rounds with `10`, `9`, `8`, or `7` cards use the existing heuristic policy instead of launching CFR.
- The earlier work cap remains in place for the remaining late-phase CFR rounds (`6` cards and below).

Why this is narrower than a full rewrite:
- The local-round CFR solver is kept intact for the lower-branching later rounds where it is much cheaper and more reliable.
- The change only removes the high-cost early phase that current evidence linked to the timeout tail.

Validation:
- `/home/thoamslai/miniconda3/envs/FAI_final/bin/python -m unittest tests.test_full_cfr_player` -> `Ran 2 tests in 0.002s`, `OK`
- Updated [tests/test_full_cfr_player.py](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/tests/test_full_cfr_player.py) so it now covers:
  - CFR still being reachable when `max_cfr_hand_size` is raised explicitly
  - the default early-round gate skipping `_solve_sampled_round()` on a round-1 `10`-card state
- Post-change 10-seed local latency probe found the slowest sampled action was `128.56 ms`, and every top-latency case was at hand sizes `6` or `5`; no `10`- through `7`-card state entered the slowest tail anymore.
- Matched post-change mini rerun on the same `100`-partition-per-player, `40`-worker broad field, [results/tournament/2026-04-13_13-49-45_example_3.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-13_13-49-45_example_3.json), recorded `CFR timeout_count=0`, `dq_count=0`, `avg_rank=2.2775`, `avg_score=11.02`, and `est_elo=1549.05`.

Remaining risk:
- This second fix clearly improved timeout robustness on the matched mini slice, but it also gave up some strength there versus the pre-change mini run (`avg_rank 2.21875 -> 2.2775`).
- The broad `2000`-partition field should be rerun later to confirm whether the `TO` reduction is worth the observed small-sample rank/score regression.

## 2026-04-13 Selective late-phase CFR experiments rejected

Task:
- Test whether a more selective late-phase CFR policy could improve both tournament strength and timeout robustness beyond the kept late-phase-only gate.

Variants tried:
- Risk-gated + top-`k` CFR:
  - kept `max_cfr_hand_size=6`
  - added heuristic/risk-based skipping within the `6`- and `5`-card phase
  - pruned our CFR action set to the top `4` heuristic cards
- Top-`k`-only CFR:
  - kept `max_cfr_hand_size=6`
  - removed the heuristic/risk skip
  - still pruned our CFR action set to the top `4` heuristic cards

Evidence:
- The current kept baseline for comparison was the late-phase-only gate on the matched broad-field mini rerun [results/tournament/2026-04-13_13-49-45_example_3.json](/home/thoamslai/projects/myproject/2026-FAI-Final-Release/results/tournament/2026-04-13_13-49-45_example_3.json): `CFR avg_rank=2.2775`, `avg_score=11.02`, `est_elo=1549.05`, `TO=0`, `DQ=0`.
- The risk-gated + top-`k` variant on the same `100`-partition-per-player, `40`-worker field produced [results/tournament/2026-04-13_15-02-03_example_3.json](/home/thoamslai/projects/myproject/2026-04-13_15-02-03_example_3.json): `CFR avg_rank=2.34`, `avg_score=11.50`, `est_elo=1535`, `TO=0`, `DQ=0`.
- The top-`k`-only variant on the same field produced [results/tournament/2026-04-13_15-06-24_example_3.json](/home/thoamslai/projects/myproject/2026-04-13_15-06-24_example_3.json): `CFR avg_rank=2.35`, `avg_score=11.82`, `est_elo=1535`, `TO=3`, `DQ=0`.
- A 10-seed local action probe for the top-`k`-only variant showed the intended phase structure (`10`-to-`7` card rounds heuristic-only; `6` and below CFR with at most `4` actions) and a local `max_elapsed_ms=195.48`, so the issue was not simple raw action slowness in isolation.

Conclusion:
- Both selective variants regressed the matched mini tournament enough that they were not kept.
- The likely issue is that pruning our own late-phase action set to the top `4` heuristic cards removes too much of the local-round CFR solver’s useful correction signal.
- Reverted the code to the better late-phase-only version with `max_cfr_hand_size=6` and no additional top-`k` pruning.
