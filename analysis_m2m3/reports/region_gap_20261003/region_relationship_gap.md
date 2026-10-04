# 区域注释与盒区–强度关系缺口

日期：2026-10-03。注释版本 `region_gap_20261003_v1`。
证据指向 `analysis_m2m3/reports/annotation_source_review.md`。作者 PromoA 数组和主表都没有逐条实验 TSS、方向或盒区坐标。

本目录的 `annotations.tsv` 按公共 annotations 契约，为每条样本的五种区域各写一行。
状态是 `missing`，链方向是 `unknown`，起点、终点和分数留空。
这些空坐标不是固定窗口，工具扫描结果也没有被写成 `reliable_external`。

覆盖率的分母是当前主表样本数。分子只计 `reliable_external` 且坐标完整的行。
`tool_inferred` 单独计数，不进入盒区与强度的计算。

- `minus10`：行数 11884，分母 11884，缺失 11884，未采用的工具推断 0，可靠且坐标完整 0，覆盖率 0.000000。关系状态 `not_computed_no_reliable_coordinates`。
- `minus35`：行数 11884，分母 11884，缺失 11884，未采用的工具推断 0，可靠且坐标完整 0，覆盖率 0.000000。关系状态 `not_computed_no_reliable_coordinates`。
- `spacer`：行数 11884，分母 11884，缺失 11884，未采用的工具推断 0，可靠且坐标完整 0，覆盖率 0.000000。关系状态 `not_computed_no_reliable_coordinates`。
- `UP_element`：行数 11884，分母 11884，缺失 11884，未采用的工具推断 0，可靠且坐标完整 0，覆盖率 0.000000。关系状态 `not_computed_no_reliable_coordinates`。
- `TSS`：行数 11884，分母 11884，缺失 11884，未采用的工具推断 0，可靠且坐标完整 0，覆盖率 0.000000。关系状态 `not_computed_no_reliable_coordinates`。

五种区域的可靠坐标子集都为空或不足以比较，因此本次没有输出 Spearman 或 Pearson。
恢复可追溯实验坐标后，重新运行本脚本才会在对应子集上计算区间 GC 与 log10(strength) 的 Spearman，并同时保留分母和覆盖率。
