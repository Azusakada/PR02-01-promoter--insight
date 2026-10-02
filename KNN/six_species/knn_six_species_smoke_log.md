# 六物种 KNN 冒烟验证记录

本次将 `reg_and_gen` 的六个物种作为独立的二分类启动子识别任务处理，不与 E. coli 连续 strength 回归混合。

## 固定设置

- 每个物种从 train 中按 `seq_id` 排序后取正类 256 条和负类 256 条。
- 每个物种从 dev 中按 `seq_id` 排序后取正类 64 条和负类 64 条。
- 特征为序列长度、碱基比例、GC/AT skew 和 16 个二核苷酸频率；每个物种独立拟合 StandardScaler。
- 模型为 `KNeighborsClassifier(n_neighbors=5, weights=distance, metric=euclidean)`。
- 每个物种请求 128 条 dev 冒烟样本，六物种合计 768 条；全部样本均完成预测。

## 各物种结果

| 物种 | train | dev smoke | accuracy | balanced accuracy | F1 | ROC-AUC | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bacillus subtilis | 512 | 128 | 0.820312 | 0.820312 | 0.818898 | 0.899780 | 1.0 |
| Baumanii | 512 | 128 | 0.789062 | 0.789062 | 0.780488 | 0.817871 | 1.0 |
| Bradyrhizobium | 512 | 128 | 0.546875 | 0.546875 | 0.539683 | 0.608521 | 1.0 |
| Diphtheria | 512 | 128 | 0.710938 | 0.710938 | 0.704000 | 0.793823 | 1.0 |
| Escherichia coli | 512 | 128 | 0.734375 | 0.734375 | 0.730159 | 0.835938 | 1.0 |
| Staphylococcus | 512 | 128 | 0.820312 | 0.820312 | 0.824427 | 0.881470 | 1.0 |

## 交付文件

- `knn_six_species_smoke_config.json`：模型、特征、样本选择、输入哈希和运行元数据。
- `knn_six_species_smoke_model.joblib`：六个物种独立的可加载模型包。
- `predictions_knn_six_species.csv`：六物种逐样本预测和状态。
- `metrics_knn_six_species.csv`：各物种 accuracy、balanced accuracy、precision、recall、F1、ROC-AUC。
- `coverage_summary_knn_six_species.csv`：各物种请求数、成功数、失败数和覆盖率。
- `common_eval_ids_knn_six_species.tsv`：六物种共同评估样本 ID。

本次结果的证据级别为 `smoke`，用于证明六个分类数据集的 KNN 特征、训练、预测和评测管线可以运行。
