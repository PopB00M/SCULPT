# Data

SCULPT uses a vulnerable-function classification dataset with CWE labels.

## Expected Schema

The main CSV file must contain:

```text
func_before,CWE ID
```

- `func_before`: source code of the vulnerable function before the fix.
- `CWE ID`: vulnerability class label, such as `CWE-119`.

The default path is:

```text
data/bigvul_10GB_CWE.csv
```

You can override it with:

```shell
python phase1_graph_construction.py --data_path /path/to/file.csv
```

## Preprocessing Used by SCULPT

Phase 1 performs the following steps:

1. Load the CSV.
2. Keep CWE classes with at least `--min_samples_per_class` samples.
3. Encode CWE labels with `LabelEncoder`.
4. Create a stratified train/validation/test split with an 8:1:1 ratio.
5. Extract CodeT5+ embeddings from `func_before`.
6. Build a KNN-style similarity graph and topology features.

The default minimum class size is `100`.

## Included Sample

`sample_data/bigvul_sample50.csv` is included only to show the expected file format. It is not large enough for the default stratified experiment.

## Redistribution

The full dataset is not bundled in this release. Before uploading a complete CSV or cached graph artifact, verify that the original dataset license permits redistribution. Note that `optimized_graphs/*.pkl` may contain raw function code through the saved `code_texts` field, so cached graph files can also be considered data redistribution.
