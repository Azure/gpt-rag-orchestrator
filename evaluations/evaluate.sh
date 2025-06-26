#!/usr/bin/env bash
set -euo pipefail

# Default: run evaluation
SKIP_EVAL=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-eval)
      SKIP_EVAL=true
      shift
      ;;
    *)
      echo "❌ Error: Unknown argument: $1"
      echo "Usage: $0 [--skip-eval]"
      exit 1
      ;;
  esac
done

# 1) Validate
if [ -z "${APP_CONFIG_ENDPOINT:-}" ]; then
  echo "❌  Error: APP_CONFIG_ENDPOINT is not set"
  exit 1
fi

# 2) Create & activate venv
python -m venv evaluations/.venv
source evaluations/.venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r evaluations/requirements.txt

# 4) Ensure Python can see your src/ package
export PYTHONPATH="$(pwd):$(pwd)/src"

# 5) Generate eval-input
echo "▶ Generating eval input…"
python evaluations/generate_eval_input.py

# 6) Conditionally run evaluation
if [ "$SKIP_EVAL" = false ]; then
  echo "▶ Running evaluation…"
  python evaluations/evaluate.py
else
  echo "▶ Skipping evaluation as requested (--skip-eval)."
fi

# 7) Teardown
deactivate
rm -rf evaluations/.venv

echo "✅  All done."
