# M3 补齐与验收

基于 main 606fdca 的已发布 87 bp 数据和五模型结果。当前 M3 的完整入口由 [已有 M2/M3 分析](../runs/ecoli87_m2m3_20261013_v1/analysis_report.md) 与 [本次补齐分析](../runs/ecoli87_m3_completion_20261013_v1/analysis_report.md) 组成。本次没有重训、替换标签、修改划分或新增 M4；汇报材料按用户要求排除。

## 补齐内容

1. spacer 的长度—强度关系：train/val 分别报告 n、中位数、Q25/Q75、低/高组占比和 Spearman。保留15–20bp的全部候选长度；未出现的长度为 n=0、统计未定义。
2. 强弱组区域比较：采用训练集三分位阈值，比较 UP、−35、spacer、−10 的 A/T、GC、共识和 spacer 长度。固定与工具推定坐标分开，训练与验证分开；−35/−10另存逐位碱基频率。中组和每条分组身份也保留。
3. 五模型效果诊断：添加仅取训练标签 log10 均值的常数预测；重算同一批 ID 的 R²、MAE、RMSE、Spearman、Pearson、预测范围、均值基线改善比例和训练/验证差距，恢复原 CNN 学习曲线。KNN训练自邻居插值、常数预测相关性未定义、日志 dropout 模式均显式标记。

完整统计为12份 CSV，图为5份 PNG/SVG；所有源表、输入、代码与图像绑定 SHA256。没有以弱相关性宣称没有生物学规律，也没有把工具择优后的 spacer 长度解释成真实注释或独立因果效应。具体标签单位和测量条件仍未知，数据/模型限制的机制解释是待进一步对照的假设。

## 复现

使用根 requirements.txt 的 Python 3.12 环境。生成必须换新运行名，不能覆盖已发布目录：

```bash
python -m pr02 complete-m3 --output runs/ecoli87_m3_completion_REPRODUCE
python -m pr02 verify-m3-completion --output runs/ecoli87_m3_completion_REPRODUCE
```

实际查看五张 PNG 后，登记检查并更新本地补充索引：

```bash
python -m pr02 acknowledge-m3-completion --output runs/ecoli87_m3_completion_REPRODUCE --reviewed \
  01_spacing_strength 02_strength_group_regions 03_box_base_differences \
  04_model_diagnostics 05_cnn_learning_curve
python -m pr02 publish-m3-completion --output runs/ecoli87_m3_completion_REPRODUCE
```

本地索引为 results/context87/m3_completion_current.json；发布后 `python -m pr02 verify` 同时复算本次补齐和原 M2/M3 分析。`verify-m3-completion` 为只读，不产生训练或修改产物。

释放前完整验证：

```bash
python scripts/verify_m2m3_release.py --output work/m3_completion_release_checks \
  --m3-completion runs/ecoli87_m3_completion_20261013_v1
```

检查器记录根测试、CNN、历史分析、数据契约、k-mer检查、两份分析的数值复算、模型重载、两个不可覆盖保护，以及全部 data/runs/results 文件字节未改变。验证记录见 [M3_COMPLETION_CHECKS.json](M3_COMPLETION_CHECKS.json)。

## 完成边界

M3 的数据与区域探索、五模型初步比较、误差/案例分析及本轮补齐已完成；所有结果为继承随机划分下的初步结果。原有 M2/M3 运行保留10张图和80条案例选择记录，本运行增加5张图，不覆盖历史结果。当前模型分数较低也属于需要如实交付的结果，M3 完成不代表模型优化完成。

E. coli test 未在本次评价；强弱阈值和常数只使用 train。TSS有来源证据，盒区边界仍是推定；邻近上下文可能重叠且历史 test 曾使用，不称为全新盲测。M4 的模型对照、多种子、归因/突变后续解释、消融与选做后续工作由用户另行安排。
