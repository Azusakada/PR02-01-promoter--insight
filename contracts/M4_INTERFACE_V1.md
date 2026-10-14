# M4 interface v1

The M4 workflow extends the frozen 87bp v3 data contract. It never scores source
test labels and never replaces M3 runs. The executable protocol is
`configs/m4_protocol_v1.json`; accepted results use `runs/m4_complete_20261014_v1/`.

## Data and identity

- Source rows and labels are unchanged. All supervised fits use `train`; fixed
  grid selection and descriptive reporting use `val`. Historical random splitting
  and previous test usage limit generalization claims.
- Source prediction CSVs retain dataset/data/split/sample identity, method/run,
  subset, raw and log10 true/predicted values, status, length and scale. Requested
  IDs remain present. Failed values must be empty, with `error_reason`.
- Source sequence length is 87; TSS index is 67. Coordinates are zero-based,
  half-open intervals. Relative TSS coordinates skip zero.
- Canonical windows: UP [7,31), -35 [32,38), spacer [38,55), -10 [55,61).
  Fixed-window and PC-inferred evidence are always separate.
- Target data have 61bp and unknown assay units, TSS and orientation. The frozen
  sequence-group split is copied verbatim. Target test has been used historically.

## Features and model references

`features.npz` contains finite float32 `X` and Unicode `sample_ids`, with no labels
or pickle arrays. `feature_manifest.json` binds feature dimensions, ordered IDs,
input hashes and extractor/recipe identity. Consumers join by ID and reject
duplicates, unknown IDs and changed checksums.

`model_refs/` provides repository-relative `checkpoint_path`, SHA256, model kind,
run/config/transform identities and predictor entry. The adapter is
`pr02.context87.m4_pipeline.predict_log10(sequences, model_ref)`. CNN explicitly
instantiates 87/61bp models with identical weights and adaptive pooling. PC only
accepts 87bp with TSS 67. DNA arbitrary-sequence prediction downloads the fixed
official revision if unavailable locally.

## Experiment rules

- Seeds: 20260928/29/30 for CNN. Four configurations, one grid with mean validation
  MSE selection. Representative checkpoint is always seed 20260928. Deterministic
  methods are not repeated to manufacture variance.
- IG: uniform 0.25 baseline, 64 trapezoid steps, signed scores and absolute ranks.
  ISM: mean absolute change over 3 alternatives. Both explain the pure CNN.
- Validation attribution cohort is the frozen 64 IDs; guided feature selection
  uses a separate seeded 256 training IDs for each selected CNN seed.
- Top-k=5/10, ties by increasing position. Compare overlap to region length/87 and
  1000 random position sets. These intervals are descriptive and uncorrected for
  multiple comparisons.
- Amplitude mutations: High/Low/Random 5 positions, 10 repeats, 64 parents, 1920
  mutants. Random positions exclude High and Low. All changes replace the base.
- Direction mutations: High 5 positions x 3 alternatives x 64 parents = 960.
  Expectations use the wild-type gradient alternative-channel minus reference-
  channel derivative in log10 units. Threshold is 1e-4. Expected direction is
  saved before mutant inference. Actual deltas are mutant minus wild type.
- A0 kmer, A1 +CNN, A2 +DNA, A3 +training-IG-guided regions. Every head is
  StandardScaler(train)+Ridge, alpha 1/10/100, selected on val. Each seed is paired
  across A1/A2/A3. This is distinct from standalone CNN/DNA regression.
- Transfer has 2 zero-shot and 6 few-shot conditions. Few-shot groups 20/50/100
  are nested, selected only from target train. Select heads on target val.

## Statistics and provenance

Score each run separately; report CNN means and sample SD across three seeds.
Do not pool three prediction copies as independent observations. Constant
prediction Spearman is undefined. Report all-success and common-cohort coverage.

Bootstrap uses 1000 repeats, seed 20260928 and percentile 95% intervals. Mutation
and attribution units are parent sequences; target units are exact-sequence
groups. Direction agreement is mutation-weighted with parent-cluster resampling.
Seed SD and bootstrap intervals do not establish assay-to-assay generalization.

`run_manifest.json` binds input, upstream artifact, environment and executed code
hashes. Exact earlier source versions remain in `reports/source_snapshots`.
Completed stages are immutable and checksum-checked before reuse. Copy the
protocol with a new output directory for a new experiment. The visual review
field records actual inspection; it cannot substitute for looking at figures.
