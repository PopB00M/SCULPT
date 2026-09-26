# SCULPT

SCULPT is a vulnerability-type detection framework that combines semantic code embeddings, topology-aware graph construction, optional LLM-based graph refinement, and an enhanced GAT detector.

The code is organized as a two-phase pipeline:

1. **Graph construction and optional LLM refinement**: build a similarity graph from vulnerable functions, compute topology features, select hub nodes by an importance rank-aggregation score, then optionally ask an OpenAI-compatible LLM to prune edges around those hubs and generate node descriptions.
2. **Graph-based detector training**: train an enhanced GAT model with topology feature fusion on the optimized DGL graph.

## Quick Guide

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Code Embedding Model](#code-embedding-model)
- [Phase 1: Build Graph](#phase-1-build-graph)
- [Phase 2: Train Detector](#phase-2-train-detector)
- [Outputs](#outputs)
- [Notes](#notes)

## Environment Setup

Create a Python environment:

```shell
conda create -n sculpt python=3.10
conda activate sculpt
```

Install PyTorch and DGL for your CUDA version. For example, choose the matching commands from the official PyTorch and DGL installation pages, then install the remaining dependencies:

```shell
pip install -r requirements.txt
```

If you prefer CPU-only FAISS, `faiss-cpu` in `requirements.txt` is sufficient. For GPU FAISS, install the appropriate `faiss-gpu` package with Conda and remove `faiss-cpu` if necessary.

## Data Preparation

SCULPT expects a CSV file with the following columns:

```text
func_before,CWE ID
```

Place the full dataset under:

```text
SCULPT/data/bigvul_10GB_CWE.csv
```

or provide a custom path:

```shell
python phase1_graph_construction.py \
  --data_path /path/to/bigvul_10GB_CWE.csv
```

A small schema example is provided at:

```text
sample_data/bigvul_sample50.csv
```

The sample file is for format inspection only. It is too small for the default stratified training split. See [DATA.md](DATA.md) for dataset and redistribution notes.

## Code Embedding Model

By default, SCULPT loads CodeT5+ from Hugging Face:

```text
Salesforce/codet5p-110m-embedding
```

If the execution environment has no internet access, download the model in advance and pass a local path:

```shell
python phase1_graph_construction.py \
  --codet5p_path /path/to/codet5p-110m-embedding
```

You can also set:

```shell
export SCULPT_CODET5P_PATH=/path/to/codet5p-110m-embedding
```

## Phase 1: Build Graph

Run graph construction without LLM refinement:

```shell
python phase1_graph_construction.py \
  --data_path data/bigvul_10GB_CWE.csv \
  --codet5p_path Salesforce/codet5p-110m-embedding \
  --output_dir optimized_graphs
```

This creates:

```text
optimized_graphs/latest_graph.pkl
```

To enable LLM refinement, set an API key through the environment and add `--use_llm`:

```shell
export DEEPSEEK_API_KEY=your_api_key_here

python phase1_graph_construction.py \
  --data_path data/bigvul_10GB_CWE.csv \
  --output_dir optimized_graphs \
  --use_llm \
  --llm_model deepseek-chat \
  --llm_base_url https://api.deepseek.com/v1
```

LLM refinement is disabled by default so that the base graph pipeline can be reproduced without external API access.

## Phase 2: Train Detector

Train SCULPT on the graph artifact produced by Phase 1:

```shell
python phase2_train_with_graph.py \
  --graph_path optimized_graphs/latest_graph.pkl \
  --output_dir training_results
```

Use LLM description-enhanced features if Phase 1 generated node descriptions:

```shell
python phase2_train_with_graph.py \
  --graph_path optimized_graphs/latest_graph.pkl \
  --use_enhanced_feat
```

Convenience scripts are also provided:

```shell
bash scripts/run_phase1.sh
bash scripts/run_phase2.sh
```

## Outputs

Phase 1 writes graph artifacts and logs to `optimized_graphs/`:

- `latest_graph.pkl`: latest DGL graph artifact for training.
- `optimized_graph_*.pkl`: timestamped graph artifact.
- `node_descriptions_*.json`: LLM-generated node descriptions, if enabled.
- `llm_evaluation_log_*.json`: LLM edge-refinement log, if enabled.

Phase 2 writes model checkpoints, logs, and metrics to `training_results/`:

- `best_model_*.pth`: best checkpoint selected by validation F1.
- `results_*.json`: final test report and training history.
- `training_*.log`: training log.

## Repository Layout

```text
SCULPT/
  README.md
  DATA.md
  MODEL_CARD.md
  NOTICE.md
  LICENSE
  requirements.txt
  environment.yml
  config.example.yaml
  .env.example
  .gitignore
  phase1_graph_construction.py
  phase2_train_with_graph.py
  sample_data/
    bigvul_sample50.csv
  scripts/
    run_phase1.sh
    run_phase2.sh
  docs/
    OPEN_SOURCE_CHECKLIST.md
```

## Notes

- Experiments are intended for a Linux-based GPU server.
- Set `CUDA_VISIBLE_DEVICES` outside the scripts when selecting GPUs.
- Random seeds default to `42`, but LLM API responses and GPU kernels may still introduce small nondeterminism.
- Do not commit `.env`, raw API keys, private checkpoints, or data files whose redistribution license is unclear.
