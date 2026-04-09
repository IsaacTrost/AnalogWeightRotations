#!/usr/bin/env bash
# Run the aihwkit Docker container with GPU access and W&B credentials from .env.
# Usage: ./run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

sudo docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v "$(pwd)":/workspace \
  --env-file .env \
  aihwkit-min
