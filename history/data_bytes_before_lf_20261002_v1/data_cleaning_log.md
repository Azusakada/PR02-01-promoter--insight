# M2 数据清洗与 KNN 冒烟记录

## E. coli 主链

- 输入：`promoter/strenth/data/supplementary/E_coli.txt`，11,884 条 50 bp 序列与正连续 `strength`。
- 输出：`data_v1.tsv`、`split_manifest.tsv`、`label_transform.json`。
- 序列清洗：统一大写；要求 `[ACGT]{50}`；源表无缺失、无重复序列。
- ID：按原始数据行生成稳定 `ecoli50_r000001` … `ecoli50_r011884`，保留 `source_row`。
- 标签：保存 `strength` 和 `target_log10=log10(strength)`；min-max 参数只在 train 拟合，不 clip val/test。
- 固定划分：`ecoli50_random_20260928_v1`，seed=20260928，random；train=8,318，val=1,783，test=1,783。

## 六物种 `reg_and_gen`

- 六个物种已逐一核查 Dataset/train/dev/test/positive_samples，结果见 `reg_and_gen_six_species_summary.tsv` 与 `reg_and_gen_six_species_cleaning_log.md`。
- 这些文件是 `label=0/1` 启动子识别数据，不是连续 strength；本次不混入 E. coli KNN 回归训练。

## KNN smoke run

- 方法：8 类理化特征 + `KNeighborsRegressor(n_neighbors=5, weights=distance, metric=euclidean)`。
- 训练：固定划分 train 中按 `sample_id` 排序取前 512 条作为 smoke 训练子集。
- 验证：固定划分 val 中按 `sample_id` 排序取前 128 条；全部成功预测。
- 结果：`predictions_knn.csv`、`metrics.csv`、`common_eval_ids.tsv`、`coverage_summary.csv`、`knn_model.joblib`。
- 证据级别：`smoke`，不是最终模型成绩；预测和指标保留样本 ID、尺度、transform、coverage 和 run ID。
