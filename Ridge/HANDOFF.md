# 接口交接记录

- task_id：PR02-01 M2+M3 / 李玘航 k-mer + Ridge/SVR
- 接口/产物：feature bundle、predictions_ridge.csv、predictions_svr.csv、ridge_config.json
- 输入文件与版本：
  - 李宇飞 `data/01_Ecoli_strength/data_v1.tsv`（sha256 `0599c5b2…`）
  - `split_manifest.tsv`（sha256 `5d162f64…`）
  - `label_transform.json` / `log10mm_59711a1c2217859e`
- 输出文件与版本：
  - `Ridge/features/feature_manifest.json`（`kmer_count_acgt_lex_v1`）
  - `Ridge/results/predictions_ridge.csv`（`run_id=ridge_kmer_20261003_v1`，`method_name=kmer3_ridge`）
  - `Ridge/results/predictions_svr.csv`（`run_id=svr_kmer_20261003_v1`，`method_name=kmer3_svr`）
- 实际命令：`python Ridge/src/prepare_frozen_inputs.py && python Ridge/src/build_feature_bundle.py && python Ridge/src/train_ridge_svr.py`
- 验证：split 成员与官方 train_ids 一致；val/test 各 1783 条全部 `ok`；k-mer 单元测试通过
- 已知限制/失败：SVR 弱于 Ridge；physchem 拼接未超过 k-mer only；不与六物种分类或 150 bp CNN 混评
- 下游读取方式：按 `sample_id` 对齐 val/test；指标用 `predicted_value_log10` 对 `extra_true_value_log10`，或用 `predicted_value` 对 `true_value`
