# Ridge/SVR 运行记录

- 日期：2026-10-03
- Python：3.13
- 命令：

```text
python Ridge/src/prepare_frozen_inputs.py
python Ridge/src/test_kmer.py
python Ridge/src/build_feature_bundle.py
python Ridge/src/train_ridge_svr.py
```

- 输入：李宇飞冻结 `data_v1` / `split_manifest` 哈希均匹配官方仓库
- 选定：`kmer3_ridge`，alpha=100，train-only fit
- val/test log10 R²：0.0528 / 0.0494
- 候选数：36（k=3/4/5 × 9 个 alpha，外加 kmer3+physchem8 对照）
- 可选 SVR：LinearSVR C=10，val log10 R²=0.0090
- 单元测试：`kmer unit tests passed`
- 证据级别：preliminary
