#!/usr/bin/env bash
# Cross-platform runner (macOS / Linux)
# Usage: ./run_experiment.sh [--config path/to/config.json] [--extra-args "--flag value"]
# By default this runs: venv/bin/python3 run_experiment.py --config configs/config_test.json

set -euo pipefail

# Default config
CONFIG="configs/config_test.json"

# If first arg is --config, consume it
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --) # end args
      shift
      break
      ;;
    *)
      # pass through other args to the python command
      EXTRA_ARGS+="$1 "
      shift
      ;;
  esac
done

# Prefer venv python if available
if [[ -x "venv/bin/python3" ]]; then
  PYTHON="venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "No python3 found. Please install Python 3 or create a virtualenv named 'venv' with the project's dependencies." >&2
  exit 1
fi

echo "Using python: $PYTHON"
echo "Config: $CONFIG"
echo "Extra args: $EXTRA_ARGS"

# Run
"$PYTHON" run_experiment.py --config "$CONFIG" $EXTRA_ARGS
