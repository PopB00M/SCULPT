#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GRAPH_PATH="${SCULPT_GRAPH_PATH:-optimized_graphs/latest_graph.pkl}"
OUTPUT_DIR="${SCULPT_TRAINING_OUTPUT_DIR:-training_results}"

python phase2_train_with_graph.py \
  --graph_path "$GRAPH_PATH" \
  --output_dir "$OUTPUT_DIR"
