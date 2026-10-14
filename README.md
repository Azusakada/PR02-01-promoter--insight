# PR02-01 Promoter Insight

默认主数据已迁移为 **11,809 条经原论文 TSS 表核验的 E. coli 87 bp 序列**。
强度值来自原 50 bp 数据，未替换标签；87 bp 上下文从历史参考基因组 U00096.2 恢复。
教师给出的 **591 条 61 bp 枯草杆菌强度记录**作为跨物种目标数据。

原 50 bp 的代码、数据、模型和报告是历史版本，入口见 [旧版说明](README_ecoli50_legacy.md)。
当前配置是 [project_config.json](project_config.json)，当前结果在 `results/context87/`，
数据契约见 [CONTEXT87_V3.md](contracts/CONTEXT87_V3.md)，迁移记录见 [报告](reports/ECOLI87_MIGRATION.md)。

## 当前完善版本（2026年10月14日）

当前实验修订采用 [m4_refinement_v2.json](configs/m4_refinement_v2.json)，
规则见 [M4_REFINEMENT_V2.md](contracts/M4_REFINEMENT_V2.md)。
修订包含扩展正则化、重新训练的基因组分组验证、三个CNN种子的解释稳定性和五次跨物种训练抽样。
最新结论以 [完善核验报告](runs/m4_refinement_20261014_v2/reports/REFINEMENT_REVIEW.md)
及 `results/context87/refinement_current.json` 为准。
项目判断与结论解释见 [本轮完善结果](reports/PROJECT_REFINEMENT_20261014.md)。

**最终项目内容尚待确认，PPT、Word和旧版正式报告保留历史版本，暂不重新制作。**
旧报告中的参数范围和结论不能直接当作本次修订结论。

```bash
python -m pr02 m4-refine --stage verify
```

完整重跑先复制修订协议并修改 `output` 为新的 `runs/` 子目录；保留 `baseline` 指向原封存运行。
常规修订复用已交付的冻结DNA表征，不需要重新下载编码器：

```bash
python -m pr02 m4-refine --protocol configs/你的新修订协议.json --stage all
```

完成后实际查看图，再按接口文档登记检查和发布结果索引。新报告写入自己的run目录，
不会自动重建PPT。断点续跑会检查完整配置、协议、上游和源码；变化时须使用新目录。

## 历史M4/M5 v1交付（2026年10月14日）

正式实验位于 `runs/m4_complete_20261014_v1/`，协议为
[m4_protocol_v1.json](configs/m4_protocol_v1.json)，接口见
[M4_INTERFACE_V1.md](contracts/M4_INTERFACE_V1.md)。已完成四项必做实验，以及
选做5跨物种迁移、选做6两种归因的一致性。

- [完整报告（在线阅读）](reports/M5_FINAL_REPORT.md)
- [可编辑Word报告](reports/M5_FINAL_REPORT.docx)
- [可编辑答辩PPT](reports/M5_DEFENSE_FINAL.pptx)
- [案例分析](reports/M4_CASE_ANALYSIS.md)、[实验说明](reports/M4_EXPERIMENT_NOTES.md)
- [最终验收与复现说明](reports/M4_M5_RELEASE.md)

CNN四配置各三种子，共12个候选记录，平均验证MSE选择原C0架构。
在共同1768条验证序列上，CNN三种子R²为 **0.1160±0.0071**，Spearman为
**0.3077±0.0143**。冻结DNABERT-2+Ridge的R²为0.0102；四阶段融合没有R²增益。
−10窗口得到IG/ISM的描述性Top-k富集支持；高归因位点突变更敏感，但计算结果不等于湿实验因果。

安装基础依赖后，直接复核已交付结果，无须下载DNA原始权重：

```bash
python -m pr02 m4 --stage verify
```

完整重跑先安装DNA依赖，并复制协议到新文件、将 `output` 改成新的 `runs/` 目录：

```bash
python -m pip install -r requirements_dnalm.txt
python -m pr02 m4 --protocol configs/m4_protocol_my_run.json --stage all --device cpu
```

DNA模型固定为官方 `zhihan1996/DNABERT-2-117M` revision
`7bce263b15377fc15361f52cfab88f8b586abda0`。可用 `--device mps` 或 `cuda` 加速DNA推理；
也可通过 `--dna-model-dir` 提供该revision的本地完整快照。原始预训练权重约469MB，按引用下载，
不存入Git。已完成阶段按摘要复用，不能通过修改同一协议来覆盖旧结果。

可分别执行 `ml`、`dna`、`cnn`、`attribution`、`mutation`、`ablation`、`transfer`、
`evaluate`、`figures`、`report` 阶段。`report` 从源表重建Markdown、案例说明和Word/PPT共用内容，
Word构建入口为 `scripts/build_m5_docx.py`（另需python-docx）；PPT构建器为
`scripts/build_m5_slides.mjs`（使用交付说明中的Artifact Tool运行时）。

胡昊铭的新版 M2/M3 补充分析见 [交接与复现说明](reports/ECOLI87_M2M3_HANDOFF.md)：
10张中文图、三种尺度的EDA、固定/推定区域分析、五方法分组误差和80条规则化案例记录。
分析报告入口为 [新版报告](runs/ecoli87_m2m3_20261013_v1/analysis_report.md)，不沿用历史50/81bp结论。

M3 收尾见 [补齐分析与基线诊断](runs/ecoli87_m3_completion_20261013_v1/analysis_report.md)
和 [验收说明](reports/M3_COMPLETION_20261013.md)：补齐 spacer 与强度关系、强弱组区域对照、
训练均值基线、训练/验证差距和 CNN 学习曲线；新增5张图与12份源表，复用冻结模型。
M3 分析交付与模型准确率分别解释；M4及本次修订另有独立运行记录。

## 安装与运行

建议 Python 3.12。依赖固定在 requirements.txt。
ZIP 解压后的目录也可直接使用；没有 Git 元数据时，运行记录仍保存代码文件 SHA256，code_commit 记为空。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pr02 validate-data
python -m pr02 verify
```

已交付运行目录用于复核，不覆盖重跑。新实验先创建配置，再运行：

```bash
python -m pr02 new-run --tag experiment01
python -m pr02 --config configs/project_ecoli87_experiment01.json run-all --research
```

`run-all` 完成 KNN、k-mer Ridge、LinearSVR、CNN、固定 TSS 的 Promoter Calculator、
同批验证样本比较、EDA、坐标及误差分析。加 `--research` 还会完成三种归因、High/Low/Random
等数量突变、四项重新训练的消融，以及 E. coli→B. subtilis 的 zero-shot/few-shot 对照。
也可分别调用 `run-knn`、`run-ridge-svr`、`run-cnn`、`run-thermo`、`evaluate`、`analyze`、
`explain`、`mutate`、`ablate`、`transfer`。各运行目录不可覆盖；归因和突变依赖当前 CNN checkpoint。

检查生成的 figures 后，登记视觉检查并发布本地结果：

```bash
python -m pr02 --config configs/project_ecoli87_experiment01.json acknowledge-figures
python -m pr02 --config configs/project_ecoli87_experiment01.json publish
```

`acknowledge-figures` 是操作者已实际查看图后的记录，不能代替看图。
`publish` 只更新本地结果索引；不会发布网站或合并 GitHub 分支。

## 数据和解释范围

主数据的原 50 bp 位于 `[17:67]`，TSS 在索引 67，整个序列对应 −67…−1、+1…+20。
验证时重做基因组双链精确定位，并逐条检查原标签、继承划分和原论文 51 nt（−50…+1）序列。
75 条多位置命中数据保留在排除文件中，避免任意选择上下文。训练/验证/测试数量是 8267/1768/1774。
归一化参数只由训练集拟合。

热力学模型在已验证 TSS 上枚举 30 种 spacer/discriminator 组合，选择最小自由能状态。
五个模型均使用同一段 87 bp；其热力学转录率另用训练集拟合 log10 线性校准。
TSS 有原论文证据，−10/−35/UP 等盒区仍是模型推定；固定窗口分析注明其假设。

枯草杆菌相同序列的重复测量按组划分，不能跨集合。k-mer 用窗口频率处理长度差异；
CNN 显式使用 61 bp 自适应池化实例及冻结源权重。few-shot 对照拟合目标域回归头。
教师数据缺少单位、实验条件、TSS 和方向信息，因此跨域 zero-shot 优先解释 Spearman，
不把跨域绝对误差作为共同强度单位下的结论。

归因、突变和消融属于计算实验；突变预测不是湿实验测量。此次 E. coli test 保持不评价。
沿用的是历史随机划分，邻近参考上下文可能重叠，且旧项目曾报告 test 指标，不能称为全新盲测。
M4已补齐三种子与冻结DNA语言模型；当前修订另含基因组分组验证，与PR01-02外部结果比较仍属后续研究。

## 复核

```bash
python -m pr02 rebuild-data --output work/rebuilt_ecoli87
python -m unittest discover -s tests -v
python -m unittest discover -s CNN/tests -v
python contracts/scripts/test_contracts.py
```

verify 会重载模型、复算预测和公共验证指标，检查 scaler、热力学校准、计算突变和目标域序列分组。
旧数据复现使用显式配置：

```bash
python -m pr02 --config configs/project_ecoli50_legacy.json validate-data
```
