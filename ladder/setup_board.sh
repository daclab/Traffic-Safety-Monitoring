#!/usr/bin/env bash
# Called by setup_board.ps1, or manually on the UNO Q host.
set -euo pipefail
REPO=${1:?Usage: setup_board.sh /absolute/repo/path [--skip-install]}
SKIP=${2:-}
VENV=${VENV:-$HOME/venvs/yolo-litert}
cd "$REPO"
if [[ $(uname -m) != aarch64 ]]; then
    echo 'Expected an aarch64 UNO Q Linux system.' >&2; exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV" || {
        echo 'venv creation failed. Install python3-venv (sudo apt install python3-venv), then retry.' >&2
        exit 1
    }
fi
source "$VENV/bin/activate"
if [[ "$SKIP" != --skip-install ]]; then
    # Binary-only: fail clearly if the board Python lacks compatible wheels.
    python -m pip install --only-binary=:all: -r ladder/requirements-board.txt
    python -m pip install --no-index ladder/kit_wheels/pycocotools-*.whl
fi
python -m pip check
python ladder/verify_board.py --repo "$REPO"
