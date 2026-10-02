# 胡昊铭 M2/M3 交接

- task_id：`pr02-eda-annotation-m2` / `pr02-error-analysis-m3`。
- 分支：`feature/eda-annotation-m2m3`，从 main `8ba8e215eae56e034c6ff2d6846aabdb276d3bd4` 开发。所有新增文件限定在 `analysis_m2m3/`；没有改共享数据或队友代码。组长负责整合，不在本分支代做其他成员模块。
- 输入：正式 11,884 条 50 bp 数据 `ecoli50_strength_v1`，固定 split `ecoli50_random_20260928_v1`（8,318/1,783/1,783）。KNN 取 main 的 val 结果；热力学取 member-b `89fcc66` 的不可变快照。路径与哈希见各 manifest。
- M2 输出：`runs/eda_m2_20261001_v2/`，5 张图各含 PNG/SVG、逐样本及统计源表、分析用 train-only transform、分布与特征关系说明。注释来源核查在 `reports/annotation_source_review.md`。
- M3 输出：`runs/error_m3_val_20261001_v2/`，4 张图各含 PNG/SVG、完整预测请求、精确评价 ID、raw/log10 指标、低中高分组误差、最大误差/模型分歧/固定随机案例和报告。只使用 val，无 test 模型比较。
- 实际命令：见各 run_manifest 的 command 数组；复现需使用 README 的 Python 环境，并换新 run 名防止覆盖。
- 验证：8 项自动测试；数据/输入/产物/逐图源表 SHA256 校验；从精确 ID 独立重算全部 26 条指标；9 张 PNG 逐张人工检查。实际通过记录见 `qa_report.json` 和 `figure_manifest.json`，接口校验结果见 `reports/validation_report.md`。
- 实际边界：M2 有可用分析与注释缺口报告，但没有可靠 −10/−35/TSS 注释；分析用归一化未替代团队正式入口。M3 目前是 KNN+热力学的阶段交付，不是完整最低要求 Ridge+热力学。热力学 1 个失败 ID 保留，共同有效 val 为 1,782 条，禁止悄悄改为 1,783。
- 李宇飞依赖：确认/发布正式 `label_transform.json` 和对应 targets；当前可以独立推进 EDA，不是只能等这一份文件。若接入物化特征，另需明确 row_index → sample_id 映射及 feature_manifest。
- 其他依赖：Ridge 的正式完整 val predictions（含失败行、模型版本和 train-only 训练/变换依据）；可靠 TSS/strand/坐标证据或完整课程源目录。CNN 可选，不阻塞 Ridge+热力学最低对照。
- 下游读取：看 README；Ridge 到位后将其 CSV 加入 `--predictions` 并创建新 M3 run，不能改写已发布 run。李宇飞正式变换用 `--transform` 接入，版本/训练 ID/hash 不符直接报错。已发布图中的来源、尺度、样本数与注释缺口必须在汇报中保留。

## 汇报可直接采用的结论

原始 strength 极端右偏，log10 更适合当前比较；GC 与 log10 强度只有弱负相关。当前 KNN 与热力学模型在共同验证集的 log10 R² 约 0.018/0.029，整体解释力弱，高强度样本常被低估。数值略高不能称显著优于，也不构成生物学机制证据。案例中的序列特征只能作描述，不能把工具推断当实验注释。
