# E. coli KNN 基线

当前集成版本使用原验证集搜索选出的 k=1501，StandardScaler 和 KNN 都只拟合 train。
验证预测引用本次实际保存的 train-only checkpoint，并独立重载核验；不做 train+val 重拟合。

```powershell
python -m pr02 run-knn
```

默认配置已有交付时会拒绝覆盖；新实验先执行 `python -m pr02 new-run --tag 新的英文tag`，再使用生成的配置。
当前运行在 `runs/knn_physchem_integrated_20261002_v2/`，全组比较见根目录 `results/`。
`results/predictions_knn.csv` 为 val 标准别名，direct log10 的 normalized 和 label_transform_id 留空。

输入八维特征按主表稳定 source_row 映射，逐条核对 GC、有限值与完整覆盖，并保存 sample_id/source_row 索引。
`results/*ecoli_full*` 和原搜索 / 图表是历史实验，保留但不作为当前模型身份。
`six_species/` 是独立分类 smoke，不能与 strength 回归混用。
