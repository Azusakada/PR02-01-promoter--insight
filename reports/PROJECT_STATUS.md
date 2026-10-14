# 项目数据与实验状态

PR02-01 启动子强度预测与关键位点归因。资料日期为2026年10月14日，代码发布分支为main。

## 数据

主数据为11,809条大肠杆菌87 bp序列，TSS索引67，标签继承天然表达数据。跨物种目标为591条枯草杆菌61 bp记录，相同序列按组划分。

## 任务与结果

| 阶段 | 内容 | 结果入口 |
| --- | --- | --- |
| M1 | 数据来源 回归目标 坐标核验 | data/03_Ecoli_strength87/ |
| M2 | 训练集分布 区域特征 spacer关系 | [研究报告第2节](final/PR02-01_REPORT.md) |
| M3 | KNN Ridge LinearSVR 热力学基线 | runs/m4_complete_20261014_v1/ |
| M4 | CNN DNA表征 归因 消融 计算突变 | 原始完整实验及修订实验目录 |
| 选做5 | 跨物种zero-shot与五次few-shot抽样 | runs/m4_refinement_20261014_v2/transfer_stability/ |
| 选做6 | IG与ISM一致性 跨CNN种子稳定性 | runs/m4_refinement_20261014_v2/attribution_stability/ |
| M5报告 | 完整研究报告与复现索引 | [Word](final/PR02-01_REPORT.docx)与[在线报告](final/PR02-01_REPORT.md) |
| M5答辩 | 修订结果，16页正文与3页备查 | [可编辑PPT](defense/PR02-01_DEFENSE.pptx) |

## 评价范围

原划分验证集1,768条，参与超参数选择。空间分组train/dev/evaluation为5,985/2,003/1,998条，重新拟合所有监督模型与预处理，跨集合基因组窗口重叠为零。历史源域test不重新计分；目标域test属于已使用基准。

模型在天然表达标签上的R²约为0.1。−10区域归因较稳定，精确碱基的一致性有限；DNA增量和跨物种few-shot未显示稳定收益。详细解释见[研究报告](final/PR02-01_REPORT.md)。

## 验证记录

数值实验通过149项测试、实际配置变更检查、独立环境模型重载和图像检查。原始实验与修订实验均按文件摘要关联数据、代码及产物。

```bash
python -m pr02 m4-refine --stage verify
python -m pr02 m4 --stage verify
```
