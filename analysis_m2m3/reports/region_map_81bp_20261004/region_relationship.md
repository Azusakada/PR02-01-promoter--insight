# 81 bp 启动子布局转移到 50 bp 强度序列

日期：2026-10-04。注释版本 `region_map_81bp_20261004_v1`。

81 bp 正样本使用数据集窗口 [-60, +20]，TSS 在 1-based 第 61 位。
−35 取 −35 到 −30，−10 取 −12 到 −7。方向按这个布局确定：−35 在 −10 上游。
只有和 label=1 的 81 bp 序列精确、唯一重合的 50 bp 才转移坐标。
重合位置不唯一、只落在负样本、或正负样本都命中的序列保持 missing，不猜测坐标。
区域必须完整落在当前 50 bp 内；探出窗口的区域记 not_applicable，坐标留空。
这是 81 bp 数据集构建坐标的转移，不是每条 50 bp 单独重测的实验盒区。

主表分母 11884。
唯一正样本匹配 614。
正样本多重位置 40。
同时命中正负样本 1。
只命中负样本 92。
没有命中 11137。

覆盖率分母仍是全部主表样本。Spearman 只在 reliable_external 且坐标完整的子集上计算。

- `minus10`：可靠坐标 577 / 11884，覆盖率 0.048553，状态 `computed`。
  区间 GC 与 log10(strength) 的 Spearman 为 -0.063415（n=577）。
- `minus35`：可靠坐标 573 / 11884，覆盖率 0.048216，状态 `computed`。
  区间 GC 与 log10(strength) 的 Spearman 为 0.054564（n=573）。
- `spacer`：可靠坐标 614 / 11884，覆盖率 0.051666，状态 `computed`。
  区间 GC 与 log10(strength) 的 Spearman 为 -0.001002（n=614）。
- `UP_element`：可靠坐标 6 / 11884，覆盖率 0.000505，状态 `computed`。
  区间 GC 与 log10(strength) 的 Spearman 为 -0.318874（n=6）。
- `TSS`：可靠坐标 300 / 11884，覆盖率 0.025244，状态 `computed`。
  区间 GC 与 log10(strength) 的 Spearman 为 0.053886（n=300）。

- `minus35` 与共识 `TTGACA` 的一致比例和 log10(strength) 的 Spearman 为 -0.013434（n=573）。
- `minus10` 与共识 `TATAAT` 的一致比例和 log10(strength) 的 Spearman 为 0.052701（n=577）。

n 小于 30 的 Spearman 只保留数值，不解释为稳定关系。UP 元件几乎都落在 50 bp 窗口外面。
−10、−35 与强度的相关都很弱，不能据此声称盒区序列决定了 strength。
