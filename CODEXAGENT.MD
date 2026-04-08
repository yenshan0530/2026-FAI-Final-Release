# CODEXAGENT.MD

## Purpose
This repository contains the FAI 2026 final project framework for building and evaluating 6 Nimmt! game-playing agents.
Agents should make scoped, verifiable changes that match the existing architecture and strictly preserve the project rules in `Spec.txt` and `README.md`.

## Working style
- For non-trivial tasks, make a short plan before editing.
- Read relevant files before making changes.
- Prefer minimal diffs over broad rewrites.
- Do not change unrelated code.
- Be explicit about assumptions and uncertainty.
- For research tasks, prefer analysis and documentation over implementation unless coding is explicitly requested.
- When working in a dirty worktree, preserve unrelated user changes.

## Code standards
- Follow existing naming, structure, and style in the repository.
- Prefer simple, readable code over clever code.
- Preserve backward compatibility unless explicitly told otherwise.
- Add comments only where they clarify non-obvious logic.
- Avoid broad exception handling. Do not use bare `except:` and do not catch `BaseException`.
- Keep submitted player logic deterministic or carefully bounded when randomness is used.

## Project-specific rules from the spec
- Player actions have a 1 second decision limit. If `action()` times out, the engine defaults to the smallest card.
- The engine timeout exception inherits from `BaseException`; catching it or swallowing it can cause disqualification or severe scoring penalties.
- Each player has a 1GB RAM limit. Avoid huge lookup tables, unbounded caches, and large model batches.
- Do not use multiprocessing or threading inside an agent. The tournament runner already parallelizes matchups externally.
- Final evaluation is CPU-only. No GPU is provided.
- Use Python 3.13.11 and only the packages listed in `requirements.txt`, unless the course staff explicitly approves additional packages.
- Final evaluation runs in a closed network. Do not depend on downloading models, data, packages, or remote resources at evaluation time.
- Do not attempt to modify game history, change the random seed for unfair advantage, or alter engine behavior unless the user explicitly asks to work on the framework itself.
- Hands passed to `action(hand, history)` are sorted. Do not sort again unless there is a clear reason.
- Prefer computing game-dependent values from `history` each turn rather than storing state blindly, because timeouts/default moves can make cached assumptions stale.
- The assignment low-card rule is deterministic: if a card is lower than all row ends, the engine takes the row with least bullheads, then shortest length, then smallest row index.
- Cards chosen by players are processed from smallest to largest after all players choose.
- A row can hold 5 cards; placing the 6th card takes the existing row as penalty and starts the row with the played card.
- Tournament performance is based on average rank, not only raw bullhead count.
- The final tournament uses random partitioning. Avoid overfitting to only released baselines.
- Released baselines may be used for training/evaluation, but must not be submitted directly or imported into the final submitted player.
- The final code submission can include two best players: `BestPlayer1` in `best_player1.py` and `BestPlayer2` in `best_player2.py`.
- If using trained models, generated weights, or intermediate data, submit all required files and document how to reproduce them. Keep inference small enough for the 1 second/1GB limits.
- If LLMs are used, disclose that usage in the report; the submitted code and report remain the student's responsibility.

## Validation
Before finishing, run the most relevant checks you can.
Preferred order:
1. Targeted tests for changed code
2. Lint/format checks
3. Broader test suite if needed

For documentation-only changes, at minimum inspect the changed file and confirm `git status --short`.

## Commands
- Install: `pip install -r requirements.txt`
- Single game smoke test: `python run_single_game.py --config configs/game/example.json`
- Tournament smoke test: `python run_tournament.py --config configs/tournament/example.json`
- Lint: not currently configured in this repository
- Format: not currently configured in this repository
- Run app: use the single-game or tournament commands above

## File boundaries
- `src/engine.py`: game engine and timeout/disqualification behavior; do not modify for agent strategy unless explicitly asked.
- `src/game_utils.py`: player/config loading helpers.
- `src/tournament_runner.py`: tournament orchestration and scoring.
- `src/players/...`: player implementations.
- `configs/...`: game and tournament configuration examples.
- `results/...`: generated game/tournament outputs when produced.
- `docs/...` or root `*.md`: documentation and research notes.

## Task-specific expectations

### Research tasks
- Read before editing.
- Identify relevant files, architecture, constraints, and risks.
- Summarize likely causes, dependencies, or solution directions.
- Prefer analysis and planning over implementation.
- Recommend the next step: draft, debug, improve, or eval.
- Do not modify code unless explicitly requested.

### Draft tasks
- Implement the smallest complete version that satisfies the request.
- Do not expand scope without justification.
- For player agents, keep a safe fallback action and enforce bounded computation.

### Debug tasks
- Reproduce the bug if possible.
- Identify root cause before fixing.
- Prefer minimal corrective changes.
- Add a regression test when appropriate.
- Do not mask timeouts or memory errors with broad exception handlers.

### Improve tasks
- Preserve intended behavior.
- Focus on one improvement dimension at a time.
- Explain tradeoffs.
- Measure impact with tournaments or targeted simulations when practical.

### Eval tasks
- Report both aggregate results and representative failures.
- Separate observation from interpretation.
- Track average rank, average score, timeouts, exceptions, disqualifications, and OOM/ERR markers when available.
- Recommend next action: debug, improve, or redesign.

## Output format
When you finish:
- Summary
- Files changed
- Validation run
- Remaining risks / assumptions
