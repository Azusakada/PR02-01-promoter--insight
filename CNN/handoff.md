# CNN 接口交接记录

状态：ready_for_review

- task_id：cnn_ecoli50_m2_m3
- 接口：fit_cnn 与 predict_sequences，API v2.0，schema 2.0.0
- 输入主表：data/01_Ecoli_strength/data_v1.tsv，data_version=ecoli50_strength_v1，SHA256=99757ee971c24765870f46aebe31e829cc64cd8ffc8604b17277703178dfacb8
- 输入划分：data/01_Ecoli_strength/split_manifest.tsv，split_id=ecoli50_random_20260928_v1，SHA256=0019d3bf898f5ec97ca1237821fd8b3ae1ac6e4bb6df2252fdafc3cc5ccb74ec
- 标签变换：CNN/runs/cnn_ecoli50_preliminary_20261002_v1/label_transform.json，transform_id=log10mm_f8b713911692db19；只用冻结 train 的 8,318 条样本拟合，未经全量归一化。
- 运行：cnn_ecoli50_preliminary_20261002_v1，evidence_level=preliminary
- 模型：CNN/runs/cnn_ecoli50_preliminary_20261002_v1/best_checkpoint.pt，SHA256=0c81c6c1bce959501c2147444c42e033ee7e1c488a26daef2c35147614ceb0c8
- 输出预测：CNN/runs/cnn_ecoli50_preliminary_20261002_v1/validation/predictions_cnn.csv，共 1,783 条 val 请求，1,783 条成功，0 条失败。
- 模型和日志：同运行目录的 model_manifest.json、cnn_run_config.yaml、training_log.jsonl、training_history.csv、training_curve.png、save_load_smoke_test.json。
- 作者和阶段说明：CNN/results_report.md。接口字段不包含人员信息。

## 实际执行和验证

从仓库根目录执行：

```text
python CNN/run_cnn.py smoke
python CNN/run_cnn.py train --config CNN/configs/cnn_run_config.yaml
python CNN/run_cnn.py predict --model-dir CNN/runs/cnn_ecoli50_preliminary_20261002_v1 --input work/cnn_inputs/cnn_ecoli50_preliminary_20261002_v1/val.tsv --config CNN/configs/cnn_run_config.yaml --output CNN/runs/cnn_ecoli50_preliminary_20261002_v1/reload_validation
python -m unittest discover -s CNN/tests -v
python CNN/contracts/scripts/validate.py self-test
python CNN/plot_cnn_diagnostics.py
```

实际解释器、完整命令、环境和训练来源 commit 记录在各 run_manifest.json 中。开发基于 main 的 `8ba8e215eae56e034c6ff2d6846aabdb276d3bd4`，当前分支为 csy。工作区代码的 source_patch.diff 和 source_snapshot.zip 保存实际执行版本，没有伪造新 commit。

CNN 14 项测试、契约 48 项测试、预测跨文件对齐、6 项指标和三份运行 manifest 校验通过；重载预测逐字节一致。证据日志与独立重算结果见 CNN/verification/。

## 下游读取

统一评价按 sample_id 连接预测，不按行号连接。主比较采用 predicted_value_log10 与 log10(true_value)，原始尺度使用 predicted_value 与 true_value；normalized 数值仅与模型绑定的 transform 配套使用。

李宇飞可读取 validation/predictions_cnn.csv，将该 CNN 的 val 结果纳入全组统一评价，并和其他方法计算共同有效 ID；CNN 自己的 common_eval_ids_cnn.tsv 只表示本方法有效样本，不是全组共同子集。胡昊铭可使用同一逐样本预测和 analysis/validation_plot_source.tsv 作误差分析。

M4 加载 best_checkpoint.pt 时同时保留 model_manifest.json、cnn_run_config.yaml 和 label_transform.json。Log10Predictor 保留可微分的 log10 输出及 50 位输入映射，适合继续接 IG；当前未生成 IG 或突变成绩。

## 限制

本次只完成固定 train/val 上的 CNN preliminary；test 尚未预测或评价。当前验证能力有限，曲线存在过拟合趋势。仓库缺少团队发布的 transform，故本运行用附件 helper 严格从 train 生成；若团队后来发布不同 transform，须显式核验或创建新运行，不静默替换现有模型的标尺。

本次没有将 CNN 分数宣称为全组最终比较，没有补造 TSS、方向、盒区坐标或实验测量单位。接收方应核对预测、模型和尺度，再确认交接。
