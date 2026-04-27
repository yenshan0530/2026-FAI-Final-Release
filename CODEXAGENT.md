# CODEXAGENT.MD

## Purpose
This repository uses multiple agent roles to work on the same 6 Nimmt! codebase with clear separation of responsibilities:
- research: understand the system, constraints, and likely next steps
- draft: produce a first complete implementation
- debug: identify root causes and fix correctness issues
- improve: make focused improvements without changing intended behavior
- eval: evaluate behavior, metrics, and regressions

Agents should make scoped, verifiable changes, leave clear written records for future agents, and strictly follow the project rules in `Spec.txt` and `README.md`.

## General Working Style
- Read relevant files before making changes.
- Prefer minimal diffs over broad rewrites.
- Do not change unrelated code.
- Be explicit about assumptions and uncertainty.
- For non-trivial tasks, make a short plan before editing.
- Prefer concrete evidence over speculation.
- If repository context is large, prioritize the most relevant files and recent log entries first.
- Preserve unrelated user changes in a dirty worktree.
- For research tasks, prefer analysis and planning over implementation unless code changes are explicitly requested.

## Project Constraints
- `action(hand, history)` has a 1 second decision limit. If a player times out, the engine defaults to the smallest card.
- The engine timeout exception inherits from `BaseException`. Do not use bare `except:` and do not catch `BaseException`, or you may swallow timeouts and trigger disqualification penalties.
- Each player has a 1GB RAM limit. Avoid huge caches, oversized models, or large rollout batches.
- Do not use `multiprocessing` or `threading` inside a submitted player. Tournament parallelism is handled externally.
- Final evaluation is CPU-only. No GPU is provided.
- Use Python 3.13.11 and only packages from `requirements.txt`, unless course staff explicitly approves more.
- Final evaluation runs in a closed network. Do not require downloads, remote APIs, or online model fetches at evaluation time.
- Hands passed to `action(hand, history)` are already sorted.
- Prefer recomputing state from `history` rather than relying on cached internal state, because timeouts and forced fallback actions can make cached assumptions stale.
- If a played card is lower than all row ends, the row taken is deterministic in this project: least bullheads, then shortest length, then smallest row index.
- Cards chosen in a round are resolved from smallest to largest.
- A row can hold 5 cards. The 6th card takes the existing row as penalty and starts the row with the played card.
- Final tournament performance is based on average rank in random-partition tournaments, not only raw bullhead count.
- Released baselines may be used for training or evaluation, but must not be submitted directly or imported into the final player.
- Final submission may include two best players: `BestPlayer1` in `best_player1.py` and `BestPlayer2` in `best_player2.py`.
- If trained weights or intermediate assets are required, submit them with the code and document reproduction in README-style docs.
- Do not try to modify game history, manipulate seeds for unfair advantage, or alter engine behavior unless the task explicitly asks for framework changes.
- If LLMs are used, that usage should be disclosed in the final report.

## Workflow
Read these before starting:
- `Spec.txt`
- `README.md`
- `CODEXAGENT.MD`
- `docs/current_findings.md`
- `docs/debug_log.md` for debug tasks
- `docs/eval_log.md` for eval tasks

Maintain these files during work:
- `docs/debug_log.md`: append-only investigation history
- `docs/eval_log.md`: append-only evaluation history
- `docs/current_findings.md`: short confirmed facts only

Rules:
- Do not rewrite or delete prior log entries; append updates.
- Update `docs/current_findings.md` only for confirmed facts, fixes, confirmed regressions, or important remaining risks.
- Do not put speculation in `docs/current_findings.md`.
- During eval, prioritize measurement and judgment.
- Do not modify code during eval unless a newly discovered correctness issue makes a minimal fix strictly necessary.
- If eval uncovers a correctness issue and a minimal fix is unavoidable, record the evidence, keep the fix tightly scoped, and recommend debug as the next step.
- Do not let eval expand into a general debugging pass.

## Logging Rules
- `docs/debug_log.md` is for:
  - bug symptoms
  - reproductions
  - findings
  - root causes
  - correctness fixes
  - important implementation risks discovered during investigation
- `docs/eval_log.md` is for:
  - evaluation target
  - setup
  - metrics
  - baseline comparison
  - interpretation
  - recommendation for next step
- `docs/current_findings.md` is for:
  - confirmed facts only
  - confirmed fixes
  - confirmed regressions
  - important remaining risks that future agents must know

## Role Boundaries
### Research
- Primarily read-only.
- Understand architecture, constraints, and likely solution directions.
- Recommend the next step: draft, debug, improve, or eval.
- Do not modify code unless explicitly requested.
- For this project, ground recommendations in `Spec.txt`, `README.md`, engine behavior, and existing player/eval artifacts.

### Draft
- Implement the smallest coherent solution that satisfies the task.
- Do not expand scope without a clear reason.
- Prefer clarity and maintainability over cleverness.
- For player agents, keep computation bounded and preserve a safe fallback decision path.

### Debug
- Reproduce the issue if possible.
- Identify root cause, not just symptom.
- Apply the minimal correct fix.
- Add a regression test when appropriate.
- Do not mask timeouts, memory errors, or engine exceptions with broad exception handlers.

### Improve
- Preserve intended behavior unless explicitly told otherwise.
- Focus on one improvement dimension at a time, such as:
  - readability
  - maintainability
  - performance
  - reliability
- Do not re-debug previously closed issues unless new evidence appears.
- For player improvements, measure the effect with a small controlled evaluation when practical.

### Eval
- Evaluate current behavior under a clearly stated setup.
- Separate raw results from interpretation.
- Identify representative failures, not just aggregate metrics.
- The primary responsibility during eval is measurement and judgment, not debugging or implementation.
- Do not modify code unless a newly discovered correctness issue makes a minimal fix strictly necessary.
- If a likely correctness issue is found:
  - record the evidence clearly
  - keep any code change minimal and tightly scoped
  - recommend debug as the next step
- Do not let eval expand into a general debugging pass.
- Prefer tournament-oriented metrics for this project: average rank first, then bullhead totals and failure markers.

## Validation
Before finishing, run the smallest relevant validation you can.
Preferred order:
1. Targeted test for changed behavior
2. Lint/format checks if relevant
3. Small benchmark or evaluation if relevant
4. Broader test suite only when needed

For documentation-only changes, inspect the changed file and confirm `git status --short`.

## Output Format
When finishing a task, report:
- target
- what you did
- files changed
- validation run
- key result
- remaining risks or assumptions
- exact log entries appended, if logs were updated

## Commands
- Install: `pip install -r requirements.txt`
- Run app: `python run_single_game.py --config configs/game/example.json`
- Run targeted test: `python run_single_game.py --config configs/game/example.json`
- Run full test suite: not currently configured; use a small tournament smoke test instead
- Lint: not currently configured in this repository
- Format: not currently configured in this repository
- Benchmark / eval: `python run_tournament.py --config configs/tournament/example.json`

## Important Paths
- Main source path: `src/`
- Main tests path: no dedicated `tests/` directory is currently present
- Eval / benchmark path: `run_tournament.py`, `configs/tournament/`, and generated `results/`
- Key docs path: `docs/`, `README.md`, `Spec.txt`, and `CODEXAGENT.MD`
