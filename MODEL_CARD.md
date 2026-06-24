# SCULPT Model Card

## Model Summary

SCULPT is a graph-based vulnerability-type detector. It embeds vulnerable functions with CodeT5+, builds a semantic similarity graph, optionally refines high-degree graph neighborhoods with an LLM, and trains an enhanced GAT classifier with topology feature fusion.

## Inputs

- Vulnerable function source code.
- CWE labels for supervised training.
- Optional LLM-generated node descriptions.

## Outputs

- Predicted CWE class for each function node.
- Validation and test classification metrics.
- Optional LLM edge-refinement logs and node descriptions.

## Intended Use

SCULPT is intended for research on vulnerability detection, graph learning, and LLM-assisted graph optimization. It is not a standalone security scanner and should not be used as the only basis for production security decisions.

## Limitations

- The detector is trained on historical vulnerable-function datasets and may not generalize to all languages, projects, or vulnerability types.
- LLM refinement depends on the selected model, prompt behavior, API availability, and cost constraints.
- Saved graph artifacts may contain raw source code and should be handled as dataset artifacts.
- False positives and false negatives are expected.

## Reproducibility Notes

The default seed is `42`. GPU kernels, package versions, and LLM responses can still introduce nondeterminism. Record package versions, CUDA version, GPU type, and whether LLM refinement was enabled for each run.
