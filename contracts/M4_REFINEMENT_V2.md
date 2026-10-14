# M4 refinement v2

The revision is a controlled extension of sealed run `m4_complete_20261014_v1`.
Its protocol is `configs/m4_refinement_v2.json`; outputs belong exclusively to
`runs/m4_refinement_20261014_v2/`. Published Word/PPT and v1 numerical outputs stay
unchanged. This revision does not declare the final presentation content settled.

## Predeclared comparisons

1. **Regularization:** keep v1 k-mer, frozen DNABERT-2, selected CNN checkpoints
   and training-guided features fixed. Refit StandardScaler + Ridge with alpha
   1, 10, 100, 1e3, 1e4, 1e5, 1e6. The inherited validation selects alpha, exactly
   as in the original benchmark. All three CNN seeds are retained and paired.
   This is a descriptive diagnosis of the bounded grid, not an independent test.
2. **Spatial control:** exclude all historical source-test rows. Partition the
   remaining genome into 20kb blocks; exclude 87bp windows crossing a block or
   circular-genome boundary. Tie exact/reverse-complement duplicate blocks, then
   shuffle groups with seed 20261014 into approximately 60/20/20 train/dev/evaluation.
   Check both strands' genomic intervals for zero cross-subset overlap.
   Refit source label scaling, all supervised models, thermodynamic calibration
   and attribution-guided recipes using this new train only. Use dev for early
   stopping/alpha, never evaluation. CNN C0 and k=4 are fixed before this control;
   three seeds are reported, with no architecture search on evaluation results.
   Fixed external pretrained DNA features can be reused because they used no
   project labels. This is one internal spatial control, not external validation.
3. **Attribution stability:** IG/ISM for the same original 64 validation parents
   on each of the three original selected CNNs. Keep 64-step IG, Top-5/10 and
   fixed/PC-inferred windows. Report cross-seed rank correlation/Jaccard and
   regional enrichment separately from the old within-model IG/ISM comparison.
4. **Transfer sampling:** five fixed training-group permutations, seeds
   20261014 through 20261018. Within each permutation use nested 20/50/100 groups,
   for k-mer and frozen representative CNN heads. Fit on target train; choose
   alpha on target val. Report all 30 few-shot fits plus two zero-shot conditions;
   never select a draw by target-test score. Target test has historical exposure.

## Coordinates and statistical units

For a one-based genomic TSS `p`, the zero-based genomic interval is
`[p-68, p+19)` on `+`, `[p-20, p+67)` on `-`. Both contain 87 genomic bases;
the oriented TSS is sequence index 67. The end is exclusive.

CNN/ablation means are means of per-seed metrics, not metrics of an ensemble.
SD describes training-seed variation. Spatial paired intervals resample 20kb
evaluation blocks while preserving each method's predictions. Attribution
resamples parents after averaging repeated seed pairs within each parent.
Transfer SD describes variation across training draws; per-draw bootstrap CIs
resample target-test sequence groups. These uncertainties are kept separate.
Bootstrap repeats, seed and confidence are read from the protocol.

Regional overlap uses length/87 as a descriptive uniform-position reference,
with no multiple-testing significance claim. Model sensitivity is not a measured
binding annotation or a wet-experiment causal effect. Cross-species absolute
errors remain scale diagnostics because assay units are unknown.

## Provenance and restart rules

Each new stage binds full project-config and protocol digests, source inputs,
upstream files, executable source hashes and every output. A changed dependency
or source rejects restart into the same output. Source snapshots allow verifying
historical outputs, but do not authorize treating an old fit as new execution.
The original sealed release may resolve changed source/README entries through
content-addressed snapshots; its original numerical outputs and final materials
must still match their files in place.

Reports and figures are run-local. `--stage all` produces the review without
calling either final-material builder. After actually viewing all six images:

```bash
python -m pr02 m4-refine --stage acknowledge --reviewed \
  01_regularization_comparison 02_regularization_search 03_grouped_validation \
  04_grouped_learning_curves 05_attribution_seed_stability 06_transfer_training_draws
python -m pr02 m4-refine --stage verify
python -m pr02 m4-refine --stage publish
```

Publishing here only updates `results/context87/refinement_current.json` after
verification. It does not generate PPT, send messages, or deploy a website.
To reproduce, copy the protocol, change its output to an unused directory, then
run `python -m pr02 m4-refine --protocol <new-protocol> --stage all`.
