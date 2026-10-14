# PR02-01 启动子强度预测与关键位点归因

以11,809条TSS经核验的大肠杆菌87 bp序列为主数据，比较传统回归、CNN、冻结DNA表征和Promoter Calculator，分析重要位点与计算突变，并使用591条枯草杆菌61 bp记录开展跨物种对照。

## 研究报告

- [完整研究报告](reports/final/PR02-01_REPORT.md)
- [可编辑Word报告](reports/final/PR02-01_REPORT.docx)
- [数据与实验状态](reports/PROJECT_STATUS.md)
- [正则化与稳健性结果](reports/PROJECT_REFINEMENT_20261014.md)

报告包含数据构建、探索分析、性能比较、区域归因、消融、计算突变、跨物种迁移及RQ1—RQ3的回答。当前正文和表格采用修订实验结果。历史报告与答辩PPT见文末版本索引。

## 主要结果

共同验证集为1,768条序列，指标在log10表达尺度计算。CNN与消融阶段先对每个训练种子计分，再报告三个种子的均值。验证集参与选参，分数属于内部基准评价。

| 模型或表征 | 验证R² | Spearman |
| --- | ---: | ---: |
| k-mer Ridge | 0.0842 | 0.2733 |
| 冻结DNABERT-2与Ridge | 0.0659 | 0.2497 |
| 纯CNN | 0.1160 | 0.3077 |
| k-mer与CNN表征 A1 | 0.1174 | 0.3118 |
| 继续加入DNA表征 A2 | 0.1100 | 0.3043 |
| 继续加入归因区域 A3 | 0.1144 | 0.3094 |

区域分组的1,998条评价序列与train/dev之间的双链窗口重叠为零，A1的平均R²为0.1162。−10区域在三个CNN种子中保持描述性富集，具体Top-k位点的一致性较低。高归因位点的替换对CNN输出更敏感；变体结果属于计算证据。跨物种100组few-shot冻结CNN的五次抽样Spearman为0.2075±0.0162，未显示相对zero-shot的稳定收益。

## 数据与坐标

源域标签沿用Deep_promoter天然表达数据；原50 bp在参考U00096.2上定位并扩展为87 bp。11,884条记录中保留11,809条位置唯一的记录，75条多位置记录单独排除。训练、验证、历史测试行数为8,267/1,768/1,774。

序列覆盖TSS上游−67至−1与下游+1至+20，TSS索引为67，原50 bp位于[17,67)。TSS有原论文序列证据；盒区采用固定窗口假设或模型推定。

目标域为教师提供的591条61 bp枯草杆菌记录，580种序列按组划分为354/119/118行。目标单位、实验条件、TSS和方向未知，跨物种主要解释Spearman。数据契约见[CONTEXT87_V3.md](contracts/CONTEXT87_V3.md)。

## 安装与核验

建议Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pr02 validate-data
python -m pr02 m4-refine --stage verify
python -m pr02 m4 --stage verify
```

两个验证命令分别检查修订实验与原始归因、突变记录，包含模型重载、预处理来源、指标复算、序列分组与文件摘要。已交付的冻结DNA特征可直接核验；重新编码新序列时需固定版本的官方权重。

## 结果目录

| 内容 | 入口 |
| --- | --- |
| 主数据与分组 | `data/03_Ecoli_strength87/`、`data/04_Bsub_strength61/` |
| 原始完整实验 | `runs/m4_complete_20261014_v1/` |
| 正则化与稳健性实验 | `runs/m4_refinement_20261014_v2/` |
| 修订协议 | [m4_refinement_v2.json](configs/m4_refinement_v2.json) |
| 实验接口 | [M4_REFINEMENT_V2.md](contracts/M4_REFINEMENT_V2.md) |
| 当前结果索引 | `results/context87/refinement_current.json` |
| 当前报告索引 | `results/context87/report_current.json` |
| 测试与重载记录 | [REFINEMENT_CHECKS_20261014.json](reports/REFINEMENT_CHECKS_20261014.json) |

新实验复制协议并将output改为新的runs子目录，baseline保留为原封存运行：

```bash
python -m pr02 m4-refine --protocol configs/your_protocol.json --stage all
```

完成后查看全部结果图，按接口文档登记图像检查，再验证与发布本地结果索引。续跑核对完整配置、上游文件和源码，变化时要求新运行目录。报告保存在各自目录，历史实验保留原始版本。

## 报告重建

```bash
python -m pip install -r requirements_report.txt
python scripts/build_course_report.py
python scripts/build_course_docx.py
```

Word采用思源黑体排版；渲染环境需安装可用的中文字体。正文、表格和图表索引由同一结构化内容生成，源文件摘要保存在reports/final/report_manifest.json。

检查现有报告的源文件、指标复算、Word表格和版本摘要：

```bash
python scripts/verify_course_report.py
```

重建报告后需重新渲染并逐页检查，更新`reports/final/layout_review.json`中的文档摘要和检查记录，再运行`python scripts/verify_course_report.py --record`登记新版本。

## 测试

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s CNN/tests -v
python contracts/scripts/test_contracts.py
python -m unittest discover -s analysis_m2m3/tests -v
```

数值实验通过149项测试及独立环境重载核验。报告的数值、引用、图表和文档结构另作检查。

## 历史版本

- [87 bp实验v1报告](reports/M5_FINAL_REPORT.md)与[历史答辩PPT](reports/M5_DEFENSE_FINAL.pptx)：使用修订前的参数和结果。
- [M3分析](runs/ecoli87_m3_completion_20261013_v1/analysis_report.md)：区域特征、spacer和基线诊断。
- [50 bp版本说明](README_ecoli50_legacy.md)：原始代码与数据入口。

历史记录用于追溯。研究结论以当前研究报告及其源表为准。
