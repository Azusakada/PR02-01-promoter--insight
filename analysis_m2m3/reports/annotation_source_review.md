# 注释来源核查与数据溯源

核查日期：2026-10-01（America/Los_Angeles）。分析主数据为 `course_ecoli50_strength / ecoli50_strength_v1`，50 bp、11,884 条。公共接口采用组长提供的 `pr02-team` Skill 2.1.0 / schema 2.0.0；该 Skill 约束缺失状态和坐标表达，人员分工依据独立 M2/M3 文档。

## 已查到的直接证据

1. 当前主表包含 `sequence / strength / target_log10 / annotation_status`，没有逐样本 TSS、链方向、−10/−35、spacer、UP 或基因组坐标；全部 `annotation_status=missing`。
2. 2023 年 Wang 等的原始研究 [Deep Learning-Assisted Design of Novel Promoters in Escherichia coli](https://doi.org/10.1002/ggn2.202300184) 明确发布了 [Promoter_design 作者仓库](https://github.com/wangxinglong1990/Promoter_design)。作者 README 将 PromoA 训练集指向 Thomason 等人的天然启动子 dRNA-seq/TSS 数据来源。
3. 在作者仓库固定提交 `56c1b331bfbca0665b2c3f6b02345de794fe25df`，`PromoS_and_PromoA_strength_prediction/PromoA_train/promoter.npy` 与 `gene_expression.npy` 各有 11,884 项。序列统一大写后，与课程主表按 source_row 顺序全部相同；字符串标签转 float 后也全部相同。逐条结果与原文件哈希见 `analysis_m2m3/inputs/source_audit_20261001/`。这确认了数据继承关系，强于仅凭相同样本量猜测来源。
4. 作者 NPY 只含序列和标签，没有逐行 TSS/−10/−35 注释。作者 README 中生成器的固定 motif 条件描述的是另一个设计功能，不能据此给 PromoA 天然序列固定坐标。
5. 队友 `member-b` 固定提交 `89fcc6600185100eac511de2a27fc0784774604c` 提供 `context_map.tsv`：用 MG1655 `NC_000913.3` 基因组做序列精确定位；其脚本明确只负责定位、没有推断 TSS/盒区。它可以支持上下文获取，却不能自动证明测量方向、TSS 或盒区。
6. `thermo/run_thermo.py` 原始结果记录工具选出的 TSS、hex10/hex35 字符串，但未完整记录选中扫描方向和各盒区在原 50 bp 中的对应坐标。工具预测本身也只能标为 `tool_inferred`，不能作为实验注释；本轮不从这些字符串倒推真值框。

## 当前可做与不可做的分析

强度分布、GC/AT/entropy 关系、局部碱基频率、验证集模型误差均可开展。

截至本轮核查，未恢复课程样本逐行可验证的 TSS/−10/−35/UP 证据。因此课程要求的“−10/−35 与强度关系”暂不可计算。应报告注释覆盖缺口，而不是输出 0 重合率、使用固定窗口或按旧 150 bp 坐标比例缩放。序列定位链方向不能等同于实验转录方向。作者数组的匹配也不验证标签单位或测量质量。

## 已查范围与仍待确认的材料

已查：当前 main 全部数据字段与代码、组长 v2.1 接口包、工作区旧 M1 150 bp/GSE108535 报告、作者仓库 README/PromoA NPY、队友热力学上下文恢复与批处理代码。旧 M1 材料涉及另一个数据对象，其 TSS 和盒区坐标不适用当前主数据。

尚未取得完整课程 `promoter/` 原始资源、Thomason 原始 TSS 坐标/计数表或 Wang 的对应补充数据中逐条坐标映射。本报告不宣称已穷尽所有公开资源。

## 需向教师或数据负责人确认

- 能否提供课程原始 `promoter/` 目录、数据整理脚本及 Wang/Thomason 对应补充表？
- 50 bp 序列相对实测 TSS 的截取范围、方向、基因组版本是什么？
- strength 取自哪个计数字段/培养条件；单位、聚合和标准化方式是什么？
- 若从原始 TSS 表恢复坐标，能否按序列、方向和计数值建立逐样本可追溯映射？
- 若实验注释无法恢复，是否允许独立、带 `tool_inferred` 标记的 motif 分析？这需要另行记录推断算法与适用条件。

## 注释恢复后的接口

独立 `annotations.tsv` 使用 `sample_id + region_type + annotation_id` 主键，0-based 左闭右开坐标限制在当前 50 bp 内；保留 `strand / annotation_status / evidence_ref / annotation_version / data_version`。只有可追溯外部证据才标 `reliable_external`；未知坐标留空。区域相关性仅在适用且可靠的注释子集计算，并同时报告分母和覆盖率。
