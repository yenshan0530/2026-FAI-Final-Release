# 2026 FAI Final - Building a 6 Nimmt! Agent 🐮

## Setup

Use CPython `3.13.x`. The bundled baseline extension file
`src/players/TA/public_baselines1.cpython-313-x86_64-linux-gnu.so` is compiled
for CPython 3.13, so Python 3.10/3.11/3.12 will not work for that module.

The standard local environment for this repo is the conda env `FAI_final`.
Current baseline-compatible local evals in this repo have been run from a
Python 3.13 conda environment rather than `./.venv`.

### Recommended Conda Setup

Create and activate a Python 3.13 conda env, then install the lightweight
runtime dependencies:

```bash
conda create -n FAI_final python=3.13 -y
conda activate FAI_final
python -m pip install -r requirements-core.txt
```

If you also want the larger RL/tooling stack kept in `requirements.txt`:

```bash
conda activate FAI_final
python -m pip install -r requirements.txt
```

### Optional `venv` Setup

If you still want an isolated `venv` for lighter local work, you can use:

```bash
python3.13 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements-core.txt
```

Note: the repo's recent public-baseline evaluations were validated in the
conda env, not the `venv`.

## Running Simulations and Tournaments

The framework provides two main scripts for evaluating players, both driven by JSON configuration files (examples located in `configs/game/` and `configs/tournament/`).

### 1. Single Game Simulation
Use `run_single_game.py` to run a single match. It supports detailed logging and captures initial hands and history outputs.
```bash
python run_single_game.py --config configs/game/example.json
```

### 2. Tournaments
Use `run_tournament.py` to run large-scale evaluations.
```bash
python run_tournament.py --config configs/tournament/example.json
```
*Optional config overrides*: You can mix-and-match configurations via the command line using `--player-cfg`, `--engine-cfg`, or `--tournament-cfg`.

## Cross-Play Dataset And Training

The repo now includes a lightweight offline learning pipeline under
`src/training/` plus two CLI helpers:

- `scripts/generate_crossplay_data.py`
- `scripts/train_crossplay_model.py`

The default top-teacher pool is `CFR,HSP,HFB,AZL,LPRA`. The generator also
supports broader diversity pools with optional lower per-label example weights.
The trainer now learns separate `early`, `mid`, and `late` linear heads from
the same candidate feature surface, so the CLI usage stays the same while the
student gets a safer capacity bump.
Currently supported labels are:

- `CFR`, `HSP`, `HFB`, `AZL`, `LPRA`
- `LFS`, `CFC`, `CRS`, `IMIT`, `RND`
- `B4`, `B5`

### Baseline Draft Dataset / Training

```bash
python scripts/generate_crossplay_data.py \
  --output results/training/crossplay_examples.jsonl \
  --games-per-lineup 1 \
  --base-seed 11 \
  --n-rounds 10

python scripts/train_crossplay_model.py \
  --dataset results/training/crossplay_examples.jsonl \
  --output src/players/crossplay_imitation/crossplay_model.json \
  --epochs 600 \
  --learning-rate 0.05 \
  --l2 0.0001
```

### Larger Diverse Dataset With Lower-Weight Auxiliary Models

This is the recommended next step if you want a much larger dataset while
keeping the strongest teacher pool dominant:

```bash
python scripts/generate_crossplay_data.py \
  --output results/training/crossplay_examples_diverse.jsonl \
  --teacher-labels CFR,HSP,HFB,AZL,LPRA,LFS,CFC,CRS,IMIT,RND \
  --teacher-weight LFS=0.35 \
  --teacher-weight CFC=0.35 \
  --teacher-weight CRS=0.35 \
  --teacher-weight IMIT=0.35 \
  --teacher-weight RND=0.20 \
  --games-per-lineup 3 \
  --include-rotations \
  --max-lineups 80 \
  --lineup-seed 11 \
  --base-seed 11 \
  --n-rounds 10

python scripts/train_crossplay_model.py \
  --dataset results/training/crossplay_examples_diverse.jsonl \
  --output src/players/crossplay_imitation/crossplay_model.json \
  --epochs 1200 \
  --learning-rate 0.03 \
  --l2 0.0001
```

Notes:

- `--teacher-weight LABEL=FLOAT` multiplies the logged-example weight for that
  label during training, on top of the existing rank-based weight.
- `--max-lineups` lets you include more labels without exploding into every
  possible `n choose 4` lineup.
- `CrossPlayImitationPlayer` loads
  `src/players/crossplay_imitation/crossplay_model.json` automatically.

### Quick Tournament Evaluation For The Learned Player

```bash
printf '%s\n' '[
  ["src.players.crossplay_imitation", "CrossPlayImitationPlayer", {}, "XPI"],
  ["src.players.TA.public_baselines1", "Baseline4", {}, "B4"],
  ["src.players.TA.public_baselines1", "Baseline5", {}, "B5"],
  ["src.players.TA.random_player", "RandomPlayer", {}, "RND"]
]' > /tmp/crossplay_eval_players.json

printf '%s\n' '{
  "type": "random_partition",
  "duplication_mode": "cycle",
  "num_games_per_player": 40,
  "num_workers": 4
}' > /tmp/crossplay_eval_tournament.json

python run_tournament.py \
  --config configs/tournament/example.json \
  --player-cfg /tmp/crossplay_eval_players.json \
  --tournament-cfg /tmp/crossplay_eval_tournament.json
```

## Config File Structure

Configuration files are structured as JSON objects with three main sections:

### 1. `players`
A list of players participating in the game or tournament. 
*   **Format**: Defined as either a dictionary `{"path": "...", "class": "...", "args": {...}, "label": "optional"}` or a compact list `["path", "class", {"args": "here"}, "optional_label"]`.
*   *Note: The compact list can be just length-2 `["path", "class"]` or 3 if no additional arguments or labels are needed. The label helps distinguish players using the same class.*

### 1b. `baselines`
An optional list with the same format as `players`. In `random_partition`-style tournaments, baseline entrants are scheduled exactly like normal players, but can also be used as score anchors for the final standings.

### 2. `engine`
Settings that control the inner game mechanics:
*   `n_players`: Number of players per game (default: 4).
*   `n_rounds`: Number of rounds played per game (default: 10).
*   `timeout` & `timeout_buffer`: Time limit (in seconds) allowed for a player's `action()`. If they exceed this, their card defaults. If they intentionally catch and swallow the alarm exception, they are disqualified.
*   `verbose`: Boolean to toggle detailed turn-by-turn print logs.

### 3. `tournament`
Settings specific to the `run_tournament.py` runner (ignored by `run_single_game.py`):
*   `type`: Type of tournament to run (`combination`, `random_partition`, or `grouped_random_partition`). **The `random_partition` tournament format will be used in final evaluations.**
*   `duplication_mode`: String (`"permutations"`, `"cycle"`, or `"none"`). Determines how hands are duplicated to reduce RNG variance. `"permutations"` plays $N!$ games with all seat assignments, `"cycle"` plays $N$ games shifting hands, and `"none"` plays 1 game. (Legacy boolean `use_permutations` is also supported).
*   `num_games_per_player` & `num_workers`: Used by `random_partition` to control the number of games played and parallel processing threads.
*   `scoring`: (Optional, used by `random_partition`). Adds a calibrated `Score` column in final standings by mapping `avg_rank` to a linear score scale defined by baseline percentiles. Supported keys are `baseline_upper_pct`, `baseline_lower_pct`, `score_at_upper_pct`, and `score_at_lower_pct`.
*   `num_groups`: (Optional, used by `grouped_random_partition`). The number of groups to split the players into for Stage 2 of the tournament based on their Stage 1 rank.
*   `max_memory_mb_per_matchup`: (Optional) Maximum memory limit in MB for a single matchup process to prevent memory leaks or out-of-memory issues from crashing the whole tournament.
*   `matchup_timeout_multiplier`: (Optional) Multiplier applied to the total expected matchup time to determine the hard timeout limit for the matchup subprocess.

## Player Disqualifications and Penalties
During games and tournaments, players may experience errors or consume excessive resources. These are marked in the final standings as:
*   **DQ (Disqualified)**: Player swallowed a timeout exception. Their subsequent moves defaults to their smallest available card.
*   **TO (Timeout)**: Player exceeded the `timeout` limit for a single turn. Their card defaults to their smallest available card.
*   **EXC (Exception)**: Player code raised an exception during their turn. Their card defaults to their smallest available card.
*   **OOM (Out of Memory)**: Player's matchup subprocess was killed due to exceeding the `max_memory_mb_per_matchup` limit. The entire matchup is aborted and players in that matchup receive no points or ranking.
*   **ERR (Generic Error)**: Player's matchup subprocess crashed for reasons other than OOM (e.g. fatal segfault, total process timeout). The entire matchup is aborted and players in that matchup receive no points or ranking.

## ⚠️ Important Warnings for Students
- **Do not use `multiprocessing` or `threading`!** Doing so will cause severe server instability and you will be penalized. The tournament orchestrator already runs games in parallel processes. Your agent must remain single-threaded.
- **Do NOT use bare `except:` or catch `BaseException` in your code.** You may accidentally catch system signals, timeout interrupts (`TimeoutException`), or out-of-memory errors (`MemoryError`), masking critical engine behaviors. Only catch specific exceptions you anticipate (e.g. `except ValueError:`).
- **Penalties for Errors:** High error rates or causing errors intentionally will result in score deductions.

## How to Add New Players
1.  Create a **subdirectory** under `src/players/` (e.g., `src/players/student_id(lowercase)/`).
2.  Add your player Python file(s) inside that directory.
3.  Your player class must implement the `action` method.

**Example:**
File: `src/players/student_id(lowercase)/best_player1.py`
```python
class BestPlayer1:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def action(self, hand, history):
        # hand: list of integers (your cards)
        # history: dict containing board state and past moves
        return hand[0] # Your logic here
```

## Game Engine Rules
*   **6th Card Rule**: If a row has 5 cards, the 6th card placed takes the row.
*   **Low Card Rule**: If a played card is lower than the last card of all rows, the player takes the row with the **lowest score**.
    *   Tie-breaking: Lowest card count -> Smallest row index.
