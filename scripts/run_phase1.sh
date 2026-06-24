#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_PATH="${SCULPT_DATA_PATH:-data/bigvul_10GB_CWE.csv}"
CODET5P_PATH="${SCULPT_CODET5P_PATH:-Salesforce/codet5p-110m-embedding}"
OUTPUT_DIR="${SCULPT_OUTPUT_DIR:-optimized_graphs}"

args=(
  --data_path "$DATA_PATH"
  --codet5p_path "$CODET5P_PATH"
  --output_dir "$OUTPUT_DIR"
)

if [[ "${SCULPT_USE_LLM:-0}" == "1" ]]; then
  args+=(--use_llm)
fi

python phase1_graph_construction.py "${args[@]}"
