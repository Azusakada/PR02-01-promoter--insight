# M2 数据探索结果

输入：ecoli50_strength_v1；固定划分 ecoli50_random_20260928_v1；样本 11,884 条，每条 50 bp。

## 实际观察

- strength 最小值 8.67，中位数 137.085，99% 分位数 26015.1609，最大值 2755441.06；原始标尺右尾很长。
- log10 变换是确定性派生，原始标签始终保留。归一化使用固定 train 的 min=0.9380190975、max=5.9142356017；不裁剪，val/test 可以超出 [0,1]。
- 归一化产物范围：provided_shared_transform。当前分析用变换保存在本 run，未写入共享 data/ 或替代李宇飞的正式 transform。
- GC 与 log10 strength 的 Spearman=-0.1395，是弱负相关。该观察不能证明 GC 因果决定强度。
- 注释状态计数：{'missing': 11884}。missing 表示没有可靠证据，不表示元件不存在；−10/−35 关系分析当前不可计算。
- GC/AT 和各碱基比例存在确定性关系，不应当作多条独立生物学证据；本轮特征直接从序列计算，避免按未声明行序拼接 KNN 8 维表。

## 图与源表

每张图的源表、尺度、样本数、参数、hash、注释等级见 figure_manifest.json。source_tables/eda_samples.tsv 保留 sample_id、split、三种尺度和特征；distribution_summary.csv 给出各子集分位数。

## 限制与交接

主表未给出实验单位、来源论文、逐样本 TSS/方向/盒区坐标。本轮展示全部样本的分布用于 EDA/QC，没有用 test 选择模型或调整分析方案。模型误差分析仅在 val 开展。−10/−35 注释来源和待确认事项见 analysis_m2m3/reports/annotation_source_review.md。
