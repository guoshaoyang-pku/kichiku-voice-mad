#!/usr/bin/env bash
# One-command bootstrap: venv + deps. Tested on macOS (Apple Silicon, MPS available).
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Next steps:"
echo "  1. Put voice clips under materials/ (see README '素材库' sections)."
echo "  2. Put the target song under prototype/materials/ and separate stems:"
echo "     demucs -n htdemucs_ft prototype/materials/<song>.wav"
echo "  3. Run the engine: python3 prototype/engine/token_dp.py --help"
