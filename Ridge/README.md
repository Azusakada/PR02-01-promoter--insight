# 李玘航：k-mer 特征 + Ridge/SVR 传统 ML 主线

本目录完成 PR02-01 M2+M3 中李玘航的连续工作流：从冻结的 50 bp E. coli strength 主表出发，生成固定词表的 k-mer 特征，接入已有 8 类派生理化特征，训练 k-mer + Ridge 主基线，并补充同一划分上的 SVR。

正式输入对齐 [李宇飞仓库](https://github.com/Azusakada/PR02-01-promoter--insight) 的 `data_v1.tsv`、`split_manifest.tsv` 和 `label_transform.json`。不再自行重新切分，也不沿用旧 GSE108535 / 150 bp 字段。

## 复现命令

在项目根目录 `promotorC` 下执行：

```text
python Ridge/src/prepare_frozen_inputs.py
python Ridge/src/test_kmer.py
python Ridge/src/build_feature_bundle.py
python Ridge/src/train_ridge_svr.py
```

## 数据与标签

- 数据对象：`course_ecoli50_strength` / `ecoli50_strength_v1`
- 样本：11,884 条 × 50 bp ACGT，`sample_id` 为 `ecoli50_r000001` … `ecoli50_r011884`
- 标签：模型拟合 `target_log10 = log10(strength)`；写回预测表时同时给出原 strength 尺度和官方 train-only min-max 归一化
- 划分：`ecoli50_random_20260928_v1`，seed=20260928，train 8318 / val 1783 / test 1783
- 标签变换：`log10mm_59711a1c2217859e`（只用于写出 `predicted_value_normalized`，不作为 Ridge 内部目标）
- 主表哈希：与仓库 `0599c5b29198c78c481d035bf32596a53411310f401463a62e160bfdbe93ca7e` 一致
- 划分哈希：与仓库 `5d162f64e68a305749033e5f9e7f4d5bdfeb6b28af757883446f75682f9fd9b0` 一致

## 特征

`Ridge/features/` 中的行序等于 `data_v1` 源序，列序由固定词表决定。

| 名称 | 维度 | 说明 |
|---|---:|---|
| kmer3 | 64 | 主线候选。全部 4^3 个 3-mer，ACGT 字典序计数 |
| kmer4 | 256 | 主线候选 |
| kmer5 | 1024 | 可选对照 |
| physchem8 | 8 | 课程已有派生特征：gc/at content、gc/at skew、melting_temp、bendability、stacking_energy、entropy。不是湿实验新测 |

等长 50 bp 下，计数与频率只差常数 `50-k+1`。词表不依赖训练集是否出现过某个 k-mer。

## 模型规则

- 主方法：k-mer + Ridge。只在 train 拟合 `StandardScaler` 和模型，用 val 的 log10 R² 选超参，**不把 val 并回训练**。
- 记录全部候选：`k ∈ {3,4,5} × alpha ∈ {0.01,0.1,1,3,10,30,100,300,1000}`，外加最佳 k-mer 拼接 physchem8 的对照，共 36 行，见 `results/ridge_search.csv`。
- 主线只在 **k-mer only** 中选 val R² 最高者，理化拼接不覆盖主方法，避免和李宇飞的 physchem KNN 抢同一条主线。
- 可选 SVR：同一 split、同一最佳 k-mer、同一 log10 标签。LinearSVR 扫 `C ∈ {0.03,0.1,0.3,1,3,10}`，再补一个 RBF SVR（C=1）。
- 预测表覆盖 val+test，每条样本都有真实值、预测值、方法名和 `prediction_status`。

## 选定结果

选定主方法：`kmer3_ridge`，`alpha=100`，`run_id=ridge_kmer_20261003_v1`。

| 方法 | subset | log10 R² | log10 Spearman | log10 MAE |
|---|---|---:|---:|---:|
| kmer3_ridge | val | 0.0528 | 0.2198 | 0.4597 |
| kmer3_ridge | test | 0.0494 | 0.2117 | 0.4528 |
| knn_physchem_full（李宇飞，对照） | test | 0.0252 | 0.1644 | 0.4600 |
| kmer3_svr（LinearSVR C=10） | val | 0.0090 | 0.2278 | 0.4464 |
| kmer3_svr（LinearSVR C=10） | test | 0.0087 | 0.2102 | 0.4376 |

观察：

- k=3 全面优于 k=4、k=5；k=5 在小 alpha 下 val R² 为负，更强正则才略好于 0。
- 给 k=3 再拼 8 类理化特征没有超过 k-mer only。
- k-mer Ridge 的 test log10 R² / Spearman 高于仓库中的 physchem KNN，但仍是弱相关，只作为中期传统 ML 基线。
- SVR 的 Spearman 略高，R² 更低，作为第二条序列特征基线保留，不替代 Ridge。

## 交付文件

特征：

- `features/kmer{3,4,5}.npz`：`X` / `sample_ids` / `vocabulary`
- `features/kmer{3,4,5}_vocabulary.txt`
- `features/physchem8.tsv`
- `features/feature_index.tsv`
- `features/feature_manifest.json`
- `features/sample_ids.txt`

模型与预测：

- `results/ridge_model.joblib`、`results/ridge_config.json`
- `results/ridge_search.csv`、`results/ridge_metrics.csv`
- `results/predictions_ridge.csv`
- `results/svr_model.joblib`、`results/svr_search.csv`、`results/svr_metrics.csv`
- `results/predictions_svr.csv`
- `results/coverage_summary.csv`、`results/common_eval_ids.tsv`

输入快照：

- `data_snapshot/data_v1.tsv`
- `data_snapshot/split_manifest.tsv`
- `data_snapshot/label_transform.json`
- `data_snapshot/input_audit.json`

## 给统一评测的读法

李宇飞对齐 `sample_id` 时，主方法读 `results/predictions_ridge.csv`：

- `true_value`：原 strength
- `predicted_value`：还原到原 strength
- `predicted_value_log10`：模型内部尺度
- `predicted_value_normalized`：官方 train-only min-max，不 clip
- `method_name=kmer3_ridge`
- `target_scale_model=log10`
- `label_transform_id=log10mm_59711a1c2217859e`
- `subset ∈ {val,test}`，coverage=1.0

不要和六物种二分类 KNN、旧 150 bp CNN 混表。

## 已知限制

- 8 类理化特征按课程表行序接入；本地 CSV 字节与仓库 `KNN/input/promoter_physicochemical.csv` 可能不同，因此不把 physchem 当作第二条主线。
- Ridge 在很小的 alpha 上会触发病态矩阵警告，选定 `alpha=100` 后不再出现该警告。
- LinearSVR 选定 C=10 时迭代 85425 次后收敛；完整搜索记录在 `svr_search.csv`。
- 本结果证据级别为 `preliminary`，供中期同样本比较，不是 M4 最终深度模型成绩。
