# E. coli 87 bp 项目迁移与验收

默认工作流已从 50 bp M2/M3 版本迁移到经原论文核验的 87 bp 主数据。枯草杆菌 61 bp 数据用于跨物种强度迁移。旧数据、旧模型和旧报告保留，通过显式历史配置复现。

## 数据来源与坐标

- Deep_promoter 原提交：c0e16c8218a983fbb07bf46d59c44109d4371b0e，原 11,884 对序列与强度。
- 历史参考：U00096.2，4,639,675 bp，与 Thomason 等论文所用 NC_000913.2 对应。
- 全部原序列重新做双链精确匹配；11,809 条唯一位置保留，75 条多位置命中单独记录。
- 保留记录逐条匹配原论文 TSS 位置、方向以及 −50…+1 的 51 nt 序列，全部通过。
- sequence[17:67] 等于原 50 bp；TSS 索引 67；87 bp 覆盖 −67…−1、+1…+20。
- 原强度完全保留；原 split 按 parent_sample_id 继承；train/val/test = 8267/1768/1774。
- 训练行重新拟合 log10 minmax，不读取验证/测试标签，不裁剪范围。

来源：[Deep_promoter](https://github.com/HaochenW/Deep_promoter)、[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4288677/)、[原始 TSS 附件](https://www.ebi.ac.uk/biostudies/files/S-EPMC4288677/JB.02096-14_zjb999093409sd1.xlsx)。文件摘要见 data/03_Ecoli_strength87/source_manifest.json。

## 工程改动

- project_config.json 默认 schema 3，公共入口统一管理 87 bp 运行；旧 schema 2 转交 legacy_cli。
- v3 校验覆盖序列长度、坐标、父样本、标签、原划分、基因组唯一定位、原论文 TSS、输入与产物摘要。
- KNN 重新计算长度适配的 8 项序列特征；新增特征定义记录在代码与运行参数中，不复用旧 CSV。
- Ridge/SVR 使用固定词表、k-mer 窗口频率及仅训练集拟合的 scaler。所有参数重新在验证集选择。
- CNN 支持声明长度，保留旧 50 bp 行为；87 bp 模型重新训练。61 bp 迁移通过自适应池化显式重载相同权重。
- SVR 使用 squared_epsilon_insensitive、dual=False 的原始空间求解；候选必须收敛。原 dual 搜索的未完成尝试没有纳入结果。
- Promoter Calculator 固定经实验表核验的 TSS、方向；枚举全部 30 组合，与先前独立验证输出逐条一致。
- 五个模型接收同一段 87 bp，在相同 1768 条验证样本上比较。热力学输出只在训练集拟合 log10 线性校准。
- EDA、区域频率、误差、盒区推定与 TSS 相对坐标统一输出；全部 5 张图已视觉检查，修复 TSS 两侧刻度重叠。

## 当前五方法结果

2026-10-13 的 M3 收尾已补齐 spacer 与强度关系、强弱组区域比较、均值基线和训练/验证诊断，详见 [M3验收说明](M3_COMPLETION_20261013.md) 与 [补齐分析](../runs/ecoli87_m3_completion_20261013_v1/analysis_report.md)。复用下列冻结模型与预测；准确率未因补充分析而改变。M4 后续工作另行安排。

仅为继承随机划分上的验证集初步结果；此次不评价 E. coli test。

| 方法 | R²（log10） | MAE（log10） | RMSE（log10） | Spearman |
|---|---:|---:|---:|---:|
| knn | 0.0519 | 0.4569 | 0.6019 | 0.2202 |
| ridge | 0.0727 | 0.4529 | 0.5953 | 0.2498 |
| svr | 0.0705 | 0.4501 | 0.5960 | 0.2638 |
| cnn | 0.1079 | 0.4484 | 0.5839 | 0.2913 |
| thermo | 0.0136 | 0.4681 | 0.6140 | 0.1047 |

## 已跑通的解释与选做实验

- 64 条按固定种子随机抽取的验证样本，完成 IG、Saliency、ISM 三种归因，输出逐位分数、排序、Spearman 和 Top-k Jaccard 一致性。
- IG 使用每位 A/C/G/T=0.25 基线与 64 步梯形积分；最大 completeness 误差约 0.003 log10。
- High/Low/Random 各 5 位同时替换，10 次重复，共 1920 条计算突变；同时导出 CNN 与固定 TSS 热力学的预测变化。
- High/Low/Random 平均绝对 CNN 变化分别约 0.127/0.048/0.067 log10；不把预测当作湿实验或因果证据。
- 四项 CNN 消融重新训练：去掉补回上下文、遮蔽典型 −10、−35、UP 窗口；相同划分、种子、训练与早停方案。

| 消融 | R²（log10） | Spearman |
|---|---:|---:|
| full_87bp | 0.1079 | 0.2913 |
| remove_recovered_context | 0.0676 | 0.2147 |
| mask_canonical_minus10 | 0.0984 | 0.2815 |
| mask_canonical_minus35 | 0.1023 | 0.3007 |
| mask_canonical_UP | 0.0957 | 0.2837 |

- 枯草杆菌 591 条原测量、580 个唯一 61 bp 序列，重复测量不删除、不平均，按相同序列分组隔离。
- 目标域 train/val/test = 354/119/118；预先固定 20/50/100 训练序列组的 few-shot 实验。
- zero-shot：冻结 E. coli Ridge 与 CNN；few-shot：目标训练拟合 k-mer 或冻结 CNN 特征的回归头，目标验证集选 alpha。
- 目标测试仅作预定条件的末次比较；没有用它选超参数。
- 教师只提供 CSV，目标 TSS、方向、单位和测量条件未知。zero-shot 以 Spearman 为主，跨域 MAE/R² 仅作尺度诊断。

| 跨物种条件 | 目标测试 n | Spearman |
|---|---:|---:|
| zero_shot_ridge | 118 | 0.1773 |
| zero_shot_cnn | 118 | 0.2161 |
| few_shot_kmer_20_groups | 118 | 0.1357 |
| few_shot_frozen_cnn_20_groups | 118 | 0.0854 |
| few_shot_kmer_50_groups | 118 | 0.0567 |
| few_shot_frozen_cnn_50_groups | 118 | -0.0384 |
| few_shot_kmer_100_groups | 118 | 0.1765 |
| few_shot_frozen_cnn_100_groups | 118 | 0.2626 |

## 验收与使用

- 28 项根目录测试（其中 9 项新增）、17 项 CNN、48 项共享接口、20 项历史分析测试通过；另 4 项 Ridge k-mer 检查通过。
- rebuild-data 对 samples、split、label_transform、data_manifest 四份文件逐字节复现通过。
- verify 重载五个模型、复算共同验证指标、训练 scaler 与热力学校准，检查归因坐标、突变位置及预测、目标域分组与 few-shot 拟合 ID，全部通过。
- 每个运行固定输入、产物和代码摘要；运行期间用于其他方法的旧源码版本保存在 reports/source_snapshots/，由 SHA256 校验。
- 当前本地发布索引 results/current.json 和 results/context87/current.json；原索引保存为 results/legacy50_current.json。

## 后续研究范围

工程迁移和上述计算实验已完成。当前不声称课题所有科学结论已完成：多种子稳定性、更严格基因组分组或外部验证、DNA 语言模型、PR01-02 外部结果对照仍需后续安排。

已核验的 TSS 与模型推定盒区分开记录；典型固定窗口只是假设。原随机划分可能存在邻近基因组上下文重叠，且历史项目曾使用 test，因此当前 E. coli test 不是全新盲测基准。跨物种结果来自单次分组划分，不选择最好一次划分来报告。
