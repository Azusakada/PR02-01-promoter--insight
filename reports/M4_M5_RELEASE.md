# M4与M5交付说明

日期：2026年10月14日。实验目录：`runs/m4_complete_20261014_v1/`。
版本标签：`m5-20261014-v1`。本轮正常推送main，不要求另行合并PR。

## 完成范围

| 任务 | 本轮交付 |
| --- | --- |
| 必做1 模型比较 | KNN、Ridge、LinearSVR、三种子CNN、校准热力学及冻结DNABERT-2+Ridge；公共1768条val，覆盖率100% |
| 必做2 位点分析 | 64条冻结val的IG、ISM；Top-5/10与UP、−35、spacer、−10的重合，长度校正和随机对照；固定/推定窗口分开 |
| 必做3 逐步消融 | A0 kmer、A1 +CNN、A2 +DNA、A3 +训练IG选区；三个CNN种子配对，10个回归头 |
| 必做4 突变检验 | 1920条等数量幅度变体、960条单点方向变体；野生型梯度预期、CNN及热力学变化，母序列bootstrap |
| 选做5 跨物种 | 2项zero-shot、6项few-shot；固定目标序列组划分；按序列组bootstrap |
| 选做6 多归因一致性 | 同一CNN、同一64条序列的IG/ISM排序相关、Top-k Jaccard、高低一致性案例 |
| M5 | 最终报告、Word、可编辑PPT、案例与实验说明、复现入口、结果和验收索引 |

## 主要结果

CNN四个配置各三个种子，以平均验证MSE选择原C0架构。代表模型固定为种子20260928，
没有挑选最佳单种子。CNN平均验证R²为0.1160，样本标准差0.0071；Spearman为
0.3077±0.0143，RMSE为0.5812±0.0023。相对训练均值基线RMSE改善约5.98%。

冻结DNABERT-2+Ridge的R²为0.0102。融合A0至A3的R²均值依次为
0.0727、0.0418、−0.0215、−0.0273，没有发现本次融合设计的性能增益。

−10窗口具有两种方法共同支持的描述性Top-k富集。IG/ISM位点排序相关均值约0.5105，
Top-5 Jaccard约0.3677，不能把两方法的具体位点选择视为等价。
High突变的平均绝对变化约0.1270 log10，Low约0.0476，Random约0.0671。
野生型梯度与CNN单点变化方向一致率98.23%，CNN与热力学一致率44.27%。

跨物种zero-shot Ridge/CNN的Spearman约0.177/0.216，100组few-shot冻结CNN约0.263。
区间宽且重叠，没有据最高观察点宣称显著优势。

## 文件入口

- `M5_FINAL_REPORT.md`：在线报告。
- `M5_FINAL_REPORT.docx`：14页可编辑报告，8张结果图。
- `M5_DEFENSE_FINAL.pptx`：16页答辩材料，8张原生可编辑图表、3张原生表格。
- `M5_REPORT_CONTENT.json`：Word/PPT共用的报告文字与数值来源。
- `M4_CASE_ANALYSIS.md`、`M4_EXPERIMENT_NOTES.md`：计算案例与实验范围。
- `M5_ARTIFACT_QA.json`：逐页视觉检查和图表数值对照记录。
- `M4_M5_CHECKS.json` 与 `m4_m5_checks/`：命令、测试和复核证据。
- `../results/context87/m4_current.json`：独立M4结果索引；M3索引保持历史身份。

## 复核与重跑

Python 3.12，先安装基础依赖。读取已交付DNA特征不需要重新下载预训练权重。

```bash
python -m pip install -r requirements.txt
python -m pr02 m4 --stage verify
```

完整重跑先复制`configs/m4_protocol_v1.json`，修改output为新的`runs/`子目录。
不修改相同运行身份的科学规则或覆盖已完成产物。

```bash
python -m pip install -r requirements_dnalm.txt
python -m pr02 m4 --protocol configs/m4_protocol_my_run.json --stage all --device cpu
```

DNA编码器可用mps/cuda。权重固定为官方`zhihan1996/DNABERT-2-117M`的
`7bce263b15377fc15361f52cfab88f8b586abda0`版本；原始权重不提交Git。
可通过`--dna-model-dir`提供该revision的本地完整快照。
CPU与本轮MPS输出的独立重算差异保存在`dna_cpu_reload_check.json`。

`--stage ml/dna/cnn/attribution/mutation/ablation/transfer/evaluate/figures/report`支持分阶段运行。
依赖不足时显式报错，不以缺失的DNA向量或常数预测替代真正模型。

报告文字可通过`--stage report`从源表重建。Word构建使用`python scripts/build_m5_docx.py`
及python-docx。PPT构建使用`scripts/build_m5_slides.mjs`，需指定Artifact Tool所在的
`PR02_NODE_MODULES`、演示文稿技能目录`PR02_PRESENTATIONS_SKILL`、运行时Python
`PR02_RUNTIME_PYTHON`；若输出已存在，使用`PR02_PPTX_OUTPUT`指定新文件。
PPT构建器保留原生图表和工作簿，图表数值保留8位小数，完整精度在CSV。

## 验收与解释边界

根目录测试54项、CNN测试17项、数据契约48项、历史分析20项，共139项。
M4复核重载12个CNN、3个ML/DNA回归模型、10个融合回归头和8个迁移条件，检查训练专用
scaler、ID、指标、突变设计、方向统计、图表源表及原M3复核。具体状态以机器验收文件为准。
保留的历史数据、运行和结果共311份文件按摘要检查，未被本轮替换。

所有预训练序列向量均不使用本课程标签提取；监督CNN表征、回归头和归因选区遵守train边界。
val参与选参，继承随机划分可能共享邻近上下文。历史源test及目标test均有先前使用，
不能称为新的盲测；本轮源test不评分。目标域单位和TSS仍未知。

本轮完成的是课程数据上的计算实验与材料交付。没有新增湿实验数据，不能把预测变化或
98.23%的同模型方向自洽率称为真实生物因果验证。
模型低分和消融无增益均保留，未为追求正结论追加同一验证集上的搜索。

原始训练运行时Git父提交为14458c1，新增代码当时尚未提交；执行源码SHA256及精确历史快照
明确记录实际实现。最终代码与产物由本发布版本标签标识，不能只用父提交解释本轮执行代码。
