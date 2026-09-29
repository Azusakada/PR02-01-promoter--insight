# E. coli KNN 全量实验

本目录只包含 E. coli `strength` 回归的全量 KNN 实验代码、输入特征、模型、逐样本预测、指标、搜索记录和图表，不包含冒烟实验或六物种分类内容。

- `input/`：8 维理化特征输入。
- `results/`：最终模型、val/test 预测、R²/MAE/Spearman、coverage、k 搜索记录和诊断图。
- `run_ecoli_knn_full.py`：按固定 train/val/test 划分重新运行完整实验。
- `plot_ecoli_knn_analysis.py`：生成 k-指标曲线和 test 诊断图。

最终按验证集 `log10 R²` 选择 `k=1501`；test `log10 R²=0.025224`、`log10 MAE=0.460037`、Spearman=0.164439。

从仓库根目录运行：

```bash
python KNN/run_ecoli_knn_full.py
python KNN/plot_ecoli_knn_analysis.py
```
