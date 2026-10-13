# 87 bp 工作流数据契约（v3）

默认入口 `python -m pr02` 读取 `project_config.json`，调用 `pr02/context87/`。
v2 的共享接口、50 bp 数组、特征与已发布结果保持原版本，使用
`python -m pr02 --config configs/project_ecoli50_legacy.json ...` 复现。
v3 不把 87 bp 数据送入限制 50 bp 的 v2 校验器。v3 的执行校验与独立复核位于
`pr02/context87/data.py` 和 `pr02/context87/audit.py`，回归测试位于 `tests/test_context87.py`。

## 主表与坐标

`data/03_Ecoli_strength87/samples.tsv` 每个唯一参考位置一行，包含 dataset_id、data_version、
split_id、sample_id、source_row、parent_sample_id、sequence、sequence_length、strength、
target_log10、split、tss_index_0based、tss_1based、reference、strand、annotation_status。

- sequence 为方向已校验的 87 bp ACGT 字符串；TSS 索引固定 67。
- source_row 是原 NPY 的一基行号；新 ID 为 ecoli87_rXXXXXX，parent_sample_id 保留原 ecoli50 ID。
- sequence[17:67] 与原 50 bp 严格一致，strength 不变，target_log10 = log10(strength)。
- Python 坐标为零基、右端不含。生物坐标为 −67…−1、+1…+20，没有 0 位。
- reference/strand/tss_1based 是 U00096.2 的基因组坐标，逐条对照原论文 TSS 表。
- 87 bp 足够容纳计算器全部 30 种组合：spacer 15…20，discriminator 6…10，UP24、gap1、−35箱6、−10箱6、ITR20。
- annotation_status 仅说明 TSS 证据。盒区坐标另存为 model_inferred；固定窗口另标 canonical_window_assumption。

split_manifest.tsv 与主表一一对应，继承原样本 split，不重新抽签。
label_transform.json 只读取保留后的训练行，记载训练 ID 摘要、范围与 clip=false。
source_manifest.json 固定原数组、参考基因组、原论文附件、验证记录、排除记录、旧划分和教师 CSV 的 SHA256。
`validate-data` 重做双链唯一定位、原始标签和原论文 TSS 序列核对。

## 模型、预测与运行

五种方法都使用相同的 87 bp 序列、训练/验证 ID 和 log10 标签。选参只用验证集；
scaler 与热力学校准只用训练集。CNN checkpoint 固定 model_config、输入文件摘要与 label_transform。
50 bp checkpoint 不得隐式用于 87 bp 输入。跨物种仅显式将自适应池化 CNN 实例化为 61 bp，并重载相同权重。

预测 CSV 固定 sample_id、数据和划分版本、method_name、run_id、subset、true_value、
predicted_value、true_value_log10、predicted_value_log10、prediction_status、sequence_length。
预测保留完整训练和验证集合；E. coli test 不用于此次选参或评价。
每个运行有不可覆盖的目录、run_manifest.json、配置快照、输入/代码/产物 SHA256。
verify 重载模型并复算验证预测、公共指标、训练 scaler 与热力学校准，检查计算突变和跨物种分组。

## M2/M3补充分析接口

`analyze-m2m3 --output runs/<新run_id>` 只读取正式87bp数据、transform、五模型val预测及固定TSS工具状态，不训练/校准/选择模型。
`verify-m2m3 --output ...` 是只读核验，复算全部源表，并用独立标量公式核验指标、分组、区域与案例。
输出采用v3样本身份与坐标规则；请求逐行保留ok/failed与原因，raw/log10指标分列。低中高阈值仅取train三分位数。
区域表 `annotation_status` 分别为 `canonical_window_assumption` / `model_inferred`，不把盒区升级为实验注释。
补充运行有独立manifest、图像/源表hash、命令、环境、代码版本和只读配置副本；同名非空目录拒绝覆盖。
实际查看全部PNG后使用 `acknowledge-m2m3 --reviewed <全部十个figure_id>` 登记；`publish-m2m3` 才更新
`results/context87/m2m3_current.json`。原project_config和模型运行绑定不变，全项目verify检查适用于当前配置的补充索引。
历史v2校验器不接收这些87bp表；旧运行保持原件。此扩展增加分析产物，不改变既有v3模型或预测字段语义。

## 枯草杆菌目标域

591 条教师 CSV 记录、580 个唯一序列，61 bp；重复测量保留，不平均标签。
相同序列的 SHA256 分组不得跨 train/val/test。TSS、单位、测量条件记为 unknown。
zero-shot 首要指标为 Spearman；原始跨域 MAE/R² 仅作尺度诊断。
few-shot 使用目标训练分组拟合回归头、目标验证集选 alpha，目标测试只在预定实验末尾评价。
不对缺少 TSS 和完整上下文的枯草杆菌输入运行 E. coli 热力学模型。
