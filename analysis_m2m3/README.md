# M2 数据探索与 M3 误差分析

胡昊铭的 EDA、注释来源核查、预测可视化和案例分析已接入全组项目。使用根 requirements.txt 的 Python 3.12 环境和公共 data_v1、split、transform。

当前 EDA 位于 runs/eda_m2_integrated_20261002_v2，共 5 张图；当前误差分析位于 runs/error_m3_five_methods_20261003_v2，共 3 张图，包含 KNN、热力学、CNN、Ridge 和 SVR，附各方法预测、覆盖、精确 ID、分组误差和案例源表。

project_config.json 注册图表运行，results/current.json 绑定其 manifest。创建新配置后用 python -m pr02 analyze 生成该配置的误差分析；默认配置对应已有成果，重复执行会拒绝覆盖。run-all 生成新图后等待实际视觉核查，再用同一配置 publish。

```powershell
python -m unittest discover -s analysis_m2m3/tests -v
python analysis_m2m3/verify_run.py runs/eda_m2_integrated_20261002_v2
python analysis_m2m3/verify_run.py runs/error_m3_five_methods_20261003_v2
```

--acknowledge-manually-reviewed 只能在实际逐张看过图后使用。当前图源表哈希、62 条误差指标复算和视觉检查均通过。主比较使用相同 1,782 条 val；完整请求 1,783 条，热力学失败保留在 coverage。共同比较 ID 与根评测一致，并绑定方法和预测版本。分组边界只用 train 确定。

Ridge 和 SVR 已接入，CNN 也已正式接入。教师 M3 要求简单 ML 与热力学；Ridge 来自组内分工承诺，当前已完成工程验收。预测整体较弱，热力学使用 150 bp 参考上下文，其他方法来自 50 bp，不能称输入信息量完全相同。

reports/annotation_source_review.md 和 inputs/source_audit_20261001 保留原始数据来源核查；inputs/thermo_member_b_89fcc66 和旧 runs 保存历史快照，不能作为当前结果入口。缺少可靠 TSS、实验方向及盒区坐标，计算推断不充当实验注释；具体研究待办见根 reports/OPEN_ITEMS.md。

区域注释缺口在 `reports/region_gap_20261003/`。它为 11,884 条样本的五种区域写出契约 `annotations.tsv`，坐标留空。可靠坐标子集为空时不计算盒区与强度的相关系数。

`reports/region_map_81bp_20261004/` 只把能唯一落在 81 bp 正样本里的 50 bp 按 [-60, +20] 布局转移 −10/−35 等坐标，其余样本保持 missing。已有输出目录不会被覆盖：

```powershell
python analysis_m2m3/region_annotation.py --output analysis_m2m3/reports/region_gap_20261003
python analysis_m2m3/map_81bp.py --output analysis_m2m3/reports/region_map_81bp_20261004
python contracts/scripts/validate.py table annotations analysis_m2m3/reports/region_gap_20261003/annotations.tsv
python contracts/scripts/validate.py table annotations analysis_m2m3/reports/region_map_81bp_20261004/annotations.tsv
```

EDA 新实验应显式传 --transform data/01_Ecoli_strength/label_transform.json；原 error_analysis.py 的任意方法读取和历史热力学校准适配功能继续保留，默认新流程读取完整校准 val 文件。
