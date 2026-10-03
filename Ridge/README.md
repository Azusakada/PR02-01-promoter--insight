# k-mer Ridge 与 SVR

李玘航的主线已接入全组默认配置。固定输入为公共 50 bp E. coli 主表、8,318/1,783/1,783 的 train/val/test split 和 train-only transform。Ridge 内部拟合 log10 strength。

## 当前入口

安装仓库根目录 requirements.txt；本次验证环境为 Python 3.12 和 scikit-learn 1.9.1。

```powershell
python Ridge/src/prepare_frozen_inputs.py
python Ridge/src/test_kmer.py
python Ridge/src/build_feature_bundle.py
python -m pr02 verify
```

prepare_frozen_inputs 检查随交付保存的快照与公共入口一致，不访问私人路径、网络或重新随机划分。build_feature_bundle 默认独立重算并核对现有特征；指定 --output runs/features_NEW 可生成到新目录，非空目录拒绝覆盖。新 bundle 使用 Unicode sample_ids，不需要 pickle 加载 ID。

已有正式模型不能覆盖。使用根 README 的 new-run/run-all 重跑全组流程；只重跑 Ridge/SVR 时，使用带新 run_id 的配置：

```powershell
python -m pr02 new-run --tag ridge_check_v1
python -m pr02 --config configs/integration_ridge_check_v1.json run-ridge-svr
```

此时只生成两条模型运行，其他新配置指定的模型尚未执行，不能直接发布为全组结果。兼容训练入口为 python Ridge/src/train_ridge_svr.py --config 配置路径，同样拒绝覆盖。

## 特征与选择

Ridge/features 保留 k=3/4/5 的 ACGT 字典序计数、行索引及 8 类派生理化对照。11,884 条样本的全部 k-mer 值已由独立 base-4 窗口算法重算，并核对理化特征的 ID 与数值。

Ridge/selection 保留原来的完整选择记录：36 个 Ridge 候选及 7 个 SVR 候选。主线只在 k-mer-only 中按 val log10 R² 选择，理化拼接仅为对照。当前沿用 k=3、Ridge alpha=100、LinearSVR C=10；本次不扩大搜索，不读取 test 选参。

原交付使用 scikit-learn 1.8.0；本次在统一 1.9.1 环境中只用 train 重新拟合。两种保存模型均能重现全部 val 预测；Ridge 系数还由独立 train 正规方程复算。与原预测仅有浮点舍入差异，详见每个 run 的 save_load_check.json。

## 产物

- runs/ridge_kmer_integrated_20261003_v1 与 runs/svr_kmer_integrated_20261003_v1 是不可覆盖的模型、预测、日志、输入核查与 manifest。
- Ridge/results/predictions_ridge.csv 和 predictions_svr.csv 为当前完整 val 别名，各 1,783 行；直接 log10 输出不填写 normalized 或 label_transform_id。
- Ridge/results 的模型、配置、各方法 metrics 和 coverage 同步注册到 results/current.json。
- results/performance_summary.csv 为五方法 1,782 条共同 val 的统一比较；本目录的单方法指标使用各自完整成功 val，人数和用途不同。
- history/liqihang_delivery_495ee43.zip 完整保留原提交 495ee43 中的源码、搜索、模型、混合 val/test 预测和历史成绩。

旧交付已经报告过 test；统一默认入口不新增 test 成绩，但后续同一 test 不能称为从未查看过的盲测。所有结果为 preliminary；当前预测能力仍有限，不能据此证明生物学机制。
