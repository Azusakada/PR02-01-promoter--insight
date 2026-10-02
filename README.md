# PR02-01 Promoter Insight

本仓库保存课程项目清洗后的启动子数据。

## 数据

- `data/01_Ecoli_strength/`：E. coli 50 bp strength 主表及 train/val/test 划分。
- `data/02_reg_and_gen_six_species/`：六个物种的启动子二分类数据及原始划分文件。

本仓库还包含 `KNN/`：E. coli strength 的全量 KNN 回归实验、模型、预测、指标和图表。KNN 目录不包含冒烟实验或六物种分类内容。

`CNN/`：50 bp A/C/G/T one-hot CNN 回归，复用固定数据和划分，train-only 标签变换、验证选模、checkpoint 保存加载、逐样本预测及接口校验。运行与交接见 [CNN/README.md](CNN/README.md)。
Promoter strength prediction and key-site attribution with interpretable machine learning.
