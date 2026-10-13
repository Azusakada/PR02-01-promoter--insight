# 胡昊铭：新版87bp M2/M3交接

范围：只做E.coli87bp数据探索、区域特征关系、五模型误差/案例、图表与中文汇报说明。复用队友冻结预测；不重训、不增加M4或枯草杆菌实验。

## 入口与产物

- 数据版本：`ecoli87_tss_verified_v1`；划分：`ecoli87_inherited_20260928_v1`；公共schema：3.0.0。
- 正式运行：[分析报告](../runs/ecoli87_m2m3_20261013_v1/analysis_report.md)、[汇报简稿](../runs/ecoli87_m2m3_20261013_v1/presentation_summary.md)、[案例表](../runs/ecoli87_m2m3_20261013_v1/source_tables/case_review.csv)。
- 10张图均导出PNG及SVG，位于上述run的figures/，逐图源表、n、尺度、子集、注释性质、图片hash与视觉状态在figure_manifest.json。
- 独立补充索引：`results/context87/m2m3_current.json`；不改已有模型/数据/配置或覆盖队友原analysis运行。
- 来源与边界：沿用已核验87bp上游数据，source_validation.json保存逐条来源检查结果；TSS可靠，盒区仅固定窗口假设或工具推定。没有移植旧614条子集或71.5%等旧数字。

## 环境与复现

Python3.12，依赖严格按根requirements.txt。正式运行前先核验上游模型；重新生成须换新run名，不能覆盖已发布目录。

```bash
python -m pr02 validate-data
python -m pr02 analyze-m2m3 --output runs/ecoli87_m2m3_REPRODUCE
python -m pr02 verify-m2m3 --output runs/ecoli87_m2m3_REPRODUCE
python -m unittest discover -s tests -v
python -m unittest discover -s CNN/tests -v
python -m unittest discover -s analysis_m2m3/tests -v
python contracts/scripts/test_contracts.py
```

需要系统中文字体：Noto Sans CJK SC、思源黑体、黑体等；找不到会明确报错。SVG把字形转为路径，不依赖阅读电脑安装相同字体。实际使用字体与hash写入环境记录。

10个图ID为 `01_strength_distribution`、`02_split_distribution`、`03_feature_relationships`、`04_tss_base_frequency`、`05_region_relationships`、`06_model_comparison`、`07_scatter`、`08_residual`、`09_strength_group_errors`、`10_model_disagreement`。

逐张看过PNG后才能执行：

```bash
python -m pr02 acknowledge-m2m3 --output runs/ecoli87_m2m3_REPRODUCE --reviewed \
  01_strength_distribution 02_split_distribution 03_feature_relationships \
  04_tss_base_frequency 05_region_relationships 06_model_comparison \
  07_scatter 08_residual 09_strength_group_errors 10_model_disagreement
python -m pr02 publish-m2m3 --output runs/ecoli87_m2m3_REPRODUCE
python -m pr02 verify
```

图像检查与数值核验分别记录；不能只运行acknowledge就声称已经看图。`verify-m2m3`不写文件；全项目原`verify`入口仍写其verification.json，若只想只读全项目核验，可在Python中调用`pr02.context87.audit.verify(cfg)`。

## 数值口径与失败处理

五方法请求各1,768个验证ID；当前均成功，共同集合为1,768。代码仍保留失败行、原因、coverage及各方法all_success/common两组评价ID。失败不能填0；版本、错位、缺失请求或尺度冲突直接报错。

阈值仅取train标签三分位点，具体值见parameters.json；预测未参与分组。最大误差每方法5条，最大分歧每对方法5条，固定种子随机共同样本5条，共80条选择记录；同一ID可按多种规则重复入选，选择规则身份保留。案例的原始强度、父ID、87bp、TSS/方向、整体/区域特征、五方法预测/误差均可追溯。

独立复核使用标量平方和、绝对误差和与秩Pearson计算80项指标，复核30项分组误差、141,708条区域记录、16项区域统计和80条案例选择；另重生成所有数值源表对照。实际测试与最终验收见本run中的verification.json及TEST_REPORT.md。

## 解释边界

数据是继承随机划分；邻近基因组上下文可能重叠，历史项目曾评价test。本次不再评价E.coli test，不称全新盲测或外部泛化。TSS表证明定位证据，不证明每条−10/−35/UP盒区实验边界；工具间隔是模型输出。共识评分和GC相关性只是探索；不编造案例的机制，不宣称方法显著优于，也不把模型计算当湿实验。
