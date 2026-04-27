import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.crossplay_learning import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    load_crossplay_examples,
    save_model,
    train_linear_policy,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a phase-aware linear cross-play imitation policy from JSONL data."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="JSONL path produced by scripts/generate_crossplay_data.py.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MODEL_PATH),
        help="Path for the trained JSON model artifact.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Number of full-batch gradient steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Gradient-descent step size.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-4,
        help="L2 regularization coefficient.",
    )
    args = parser.parse_args()

    examples = load_crossplay_examples(args.dataset)
    model_data = train_linear_policy(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    save_model(model_data, args.output)

    summary = model_data["training_summary"]
    print(
        "trained model with "
        f"{summary['num_decisions']} decisions, "
        f"accuracy={summary['train_accuracy']:.4f}, "
        f"loss={summary['train_loss']:.4f}, "
        f"saved to {args.output}"
    )


if __name__ == "__main__":
    main()
