# M2 数据探索与 M3 分析交接

胡昊铭的数据探索、来源核查、模型误差与案例分析已接入公共 data_v1、split 和 transform。2026-10-05 在 csy 增加区域分析修订版：修正来源匹配与注释标记，补齐中文图表、源表和可复现记录。

## 当前报告

- 数据探索：`runs/eda_m2_integrated_20261002_v2/`，5 张图，11,884 条样本。
- 五方法误差与案例：`runs/error_m3_five_methods_20261005_v3/`，3 张图，KNN、热力学、CNN、Ridge、SVR 使用共同 val 1,782 条；完整请求 1,783 条。
- [区域特征与强度报告](reports/region_analysis_20261005_v2/region_relationship.md)：字母频率、模式相似度分组、样本分布三张中文图，附原始记录编号、覆盖统计、重复抽样区间和实际运行脚本快照。
- [中期汇报与修订说明](../reports/MIDTERM_REGION_UPDATE_20261005.md)：可直接用于汇报的文字和本轮修改清单。

区域报告先讲发现：−10 区富含 A/T，−35 的字母特征较弱，简单特征与强度的关系有限。方法、样本数和固定布局规则集中在方法说明与图注中。完整实验注释的补取属于后续来源完善，中期分析按现有数据推进。

## 检查与复现

```powershell
python -m unittest discover -s analysis_m2m3/tests -v
python analysis_m2m3/verify_run.py analysis_m2m3/reports/region_analysis_20261005_v2
python contracts/scripts/validate.py table annotations analysis_m2m3/reports/region_analysis_20261005_v2/annotations.tsv
```

重新生成时指定新目录，已有非空运行不会覆盖：

```powershell
python analysis_m2m3/map_81bp.py --output analysis_m2m3/reports/region_analysis_reproduce_20261005
```

`map_81bp.py` 保留每条 81 bp 原始记录身份，只接受所有正样本匹配具有相同位置、方向和长度的情况；同时命中负样本的记录排除。按固定布局生成的区域标为 `tool_inferred`；`summarize_regions(..., include_inferred=True)` 显式开展探索统计，外部确认注释单独计数。

实际逐张看过 PNG 后，使用 `verify_run.py --acknowledge-manually-reviewed` 记录图像验收。`project_config.json` 与 `results/current.json` 注册当前分析入口。新模型版本由 `python -m pr02 analyze` 生成对应误差图；通过验收后再 publish。

## 历史与来源

`reports/annotation_source_review.md` 和 `inputs/source_audit_20261001/` 是当时的来源核查记录。旧报告中“缺少可靠注释”的表述不代表当前探索分析停工。`inputs/thermo_member_b_89fcc66/` 保留测试和来源追溯所需的历史快照；重复旧 runs 已移出工作树。当前误差分析读取正式的五方法预测，恢复旧产物的方法见根 `reports/REPOSITORY_CLEANUP_20261005.md`。

教师 M3 要求简单 ML 与热力学，组内承诺的 Ridge 已完成工程验收。热力学使用补取的 150 bp 上下文，其他方法来自 50 bp；预测尺度与输入范围见各运行记录。后续工作见根 `reports/OPEN_ITEMS.md`。
