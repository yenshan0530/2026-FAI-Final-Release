import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.crossplay_learning import (
    AVAILABLE_CROSSPLAY_LABELS,
    DEFAULT_DATASET_PATH,
    DEFAULT_TEACHER_LABELS,
    generate_crossplay_examples,
    save_crossplay_examples,
)


def _parse_labels(raw_value):
    labels = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label must be provided.")
    unknown = sorted(set(labels).difference(AVAILABLE_CROSSPLAY_LABELS))
    if unknown:
        raise ValueError(
            f"Unknown label(s): {unknown}. Available labels: {list(AVAILABLE_CROSSPLAY_LABELS)}."
        )
    return labels


def _parse_weight_assignments(assignments):
    weights = {}
    for assignment in assignments:
        label, sep, value = assignment.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid --teacher-weight value '{assignment}'. Expected LABEL=FLOAT."
            )
        label = label.strip()
        if label not in AVAILABLE_CROSSPLAY_LABELS:
            raise ValueError(
                f"Unknown label in --teacher-weight: {label}. "
                f"Available labels: {list(AVAILABLE_CROSSPLAY_LABELS)}."
            )
        weights[label] = float(value)
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-play training data from the confirmed teacher pool."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DATASET_PATH),
        help="JSONL path for the generated decision dataset.",
    )
    parser.add_argument(
        "--games-per-lineup",
        type=int,
        default=1,
        help="Number of games to run for each 4-player teacher lineup.",
    )
    parser.add_argument(
        "--teacher-labels",
        default=",".join(DEFAULT_TEACHER_LABELS),
        help=(
            "Comma-separated labels to include in the cross-play pool. "
            f"Available labels: {', '.join(AVAILABLE_CROSSPLAY_LABELS)}."
        ),
    )
    parser.add_argument(
        "--teacher-weight",
        action="append",
        default=[],
        help=(
            "Optional per-label dataset weight written into each example as LABEL=FLOAT. "
            "Use this to down-weight diversity labels during training."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=11,
        help="Base seed used to derive per-game seeds.",
    )
    parser.add_argument(
        "--include-rotations",
        action="store_true",
        help="Also rotate seats for every teacher lineup.",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=10,
        help="Number of rounds per generated game.",
    )
    parser.add_argument(
        "--max-lineups",
        type=int,
        default=None,
        help="Optional cap on the number of sampled lineups after combination/rotation expansion.",
    )
    parser.add_argument(
        "--lineup-seed",
        type=int,
        default=None,
        help="Seed used when shuffling before applying --max-lineups. Defaults to --base-seed.",
    )
    args = parser.parse_args()

    teacher_labels = _parse_labels(args.teacher_labels)
    teacher_weights = _parse_weight_assignments(args.teacher_weight)

    examples = generate_crossplay_examples(
        games_per_lineup=args.games_per_lineup,
        base_seed=args.base_seed,
        include_rotations=args.include_rotations,
        teacher_labels=teacher_labels,
        teacher_weights=teacher_weights,
        engine_cfg={"n_players": 4, "n_rounds": args.n_rounds},
        max_lineups=args.max_lineups,
        lineup_seed=args.lineup_seed,
    )
    save_crossplay_examples(examples, args.output)

    teacher_labels = sorted({example["teacher"] for example in examples})
    total_weight = sum(float(example.get("example_weight", 1.0)) for example in examples)
    print(
        "saved "
        f"{len(examples)} examples from {len(teacher_labels)} teachers "
        f"(total example weight {total_weight:.2f}) to {args.output}"
    )


if __name__ == "__main__":
    main()
