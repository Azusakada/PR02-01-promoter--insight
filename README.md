# PR02-01 Promoter Insight

本仓库保存课程项目清洗后的启动子数据。

## 数据

- `data/01_Ecoli_strength/`：E. coli 50 bp strength 主表及 train/val/test 划分。
- `data/02_reg_and_gen_six_species/`：六个物种的启动子二分类数据及原始划分文件。

本仓库还包含 `KNN/`：E. coli strength 的全量 KNN 回归实验、模型、预测、指标和图表。KNN 目录不包含冒烟实验或六物种分类内容。

本仓库还包含 `thermo/`：E. coli strength 的热力学对照基线（开源 `regseq2 Promoter_Calculator`，宿主 MG1655）。因 50 bp 太短，先用参考基因组恢复上下文再批量运行；输出 Tx_rate 与主表 strength 不同标尺，未校准只做 Spearman。详见 `thermo/README.md` 与 `thermo/thermo_applicability.md`。
Promoter strength prediction and key-site attribution with interpretable machine learning.
