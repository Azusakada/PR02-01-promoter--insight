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

集成修复中的公共交付状态以统一验收报告为准。

## E. coli CNN 回归

`CNN/`：50 bp A/C/G/T one-hot CNN，固定数据和划分、train-only 标签变换、验证选模、checkpoint 保存加载及逐样本预测。见 [CNN/README.md](CNN/README.md)。

## 六物种数据与 KNN 冒烟结果

- `data/02_reg_and_gen_six_species/`：Bacillus subtilis、Baumanii、Bradyrhizobium、Diphtheria、Escherichia coli、Staphylococcus 六个独立二分类数据集，以及汇总表和清洗日志。
- `KNN/six_species/`：六物种独立 KNN 分类的配置、模型、逐样本预测、指标、覆盖率、共同评估 ID 和冒烟日志。该结果证据级别为 `smoke`，不与 E. coli 连续 strength 回归混用。
