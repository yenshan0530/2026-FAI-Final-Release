#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODE="${1:-core}"

case "$MODE" in
    core)
        REQUIREMENTS_FILE="$ROOT_DIR/requirements-core.txt"
        ;;
    full)
        REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
        ;;
    *)
        echo "Usage: $0 [core|full]" >&2
        exit 1
        ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: $PYTHON_BIN was not found. Install CPython 3.13 first." >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 13):
    raise SystemExit(
        f"error: expected CPython 3.13.x, found {sys.version.split()[0]}"
    )
PY

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m ensurepip --upgrade
"$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE"

echo
echo "Environment ready at $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Quick check: python -m unittest discover -s tests -q"
echo "Run example: python run_single_game.py --config configs/game/example.json"
