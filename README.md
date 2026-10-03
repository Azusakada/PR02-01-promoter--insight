# PR02-01 Promoter Insight

本仓库保存 PR02-01 M2/M3 阶段的可复现数据与实验交付物。

## E. coli strength 数据

- `data/01_Ecoli_strength/data_v1.tsv`：统一的 50 bp E. coli strength 主表。
- `data/01_Ecoli_strength/split_manifest.tsv`：固定 train/val/test 划分。
- `data/01_Ecoli_strength/label_transform.json`：训练集拟合的 `log10(strength)` 标签变换契约。
- `data/01_Ecoli_strength/data_cleaning_log.md`：数据清洗、标签处理和运行记录。

## E. coli KNN 回归

`KNN/results/` 同时保留带实验后缀的历史文件和以下标准交付文件：

- `knn_config.json`、`knn_model.joblib`
- `predictions_knn.csv`
- `metrics.csv`、`common_eval_ids.tsv`、`coverage_summary.csv`

标准预测表合并了 val/test 两个固定划分，并通过 `label_transform_id` 关联标签变换契约。

## E. coli k-mer Ridge / SVR

`Ridge/` 是李玘航的传统 ML 主线，基于同一份冻结 `data_v1` 和 `ecoli50_random_20260928_v1` 划分：

- `Ridge/features/`：固定词表的 k=3/4/5 计数特征、8 类派生理化特征和 feature manifest
- `Ridge/results/predictions_ridge.csv`：主方法 `kmer3_ridge`（train 拟合、val 选 `alpha=100`）
- `Ridge/results/predictions_svr.csv`：同一 split/标签尺度上的可选 `kmer3_svr`
- `Ridge/results/ridge_search.csv`：全部 k × alpha 候选，不只保留最优结果

复现命令见 `Ridge/README.md`。该结果证据级别为 `preliminary`，不要和六物种二分类混评。

## 六物种数据与 KNN 冒烟结果

- `data/02_reg_and_gen_six_species/`：Bacillus subtilis、Baumanii、Bradyrhizobium、Diphtheria、Escherichia coli、Staphylococcus 六个独立二分类数据集，以及汇总表和清洗日志。
- `KNN/six_species/`：六物种独立 KNN 分类的配置、模型、逐样本预测、指标、覆盖率、共同评估 ID 和冒烟日志。该结果证据级别为 `smoke`，不与 E. coli 连续 strength 回归混用。
