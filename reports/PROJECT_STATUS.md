# PR02-01 csy 统一集成交付

日期：2026-10-02。开发仓库：D:\PR02-01-promoter--insight，工作分支 csy。

## 完成范围

已合入最新 main、member-b 和 feature/eda-annotation-m2m3；原 main 未被改写。原 CNN 开发已先提交，README 冲突已解决。原实验、文件字节和旧标准别名保存在历史目录或原运行目录。

- 公共 schema 校验器移到根目录 contracts，统一数据、split 与 train-only transform。LF 换行规则固定，内容及划分未改变；公共 transform 哈希校验通过。
- 修复热力学正反向候选覆盖，保留选中方向和正确映射坐标，增加回归测试。全量 11,884 请求成功 11,883，失败 1（ecoli50_r003290，未定位）。相比旧结果，1382 条成功预测发生变化；未把失败记录删除或填成零。
- 校准只用 8,318 条 train，校准 CSV 完整保留请求。按子集报告人数与 coverage，run_manifest 满足接口并绑定产物哈希。
- 重建 KNN train-only 模型，沿用原 val 搜索选出的 k=1501；验证预测能由保存模型精确重现。直接 log10 预测不填写 normalized / label_transform_id，标准别名只含 val。
- CNN 使用公共 transform 重新训练，最佳 epoch 4、16 epoch 早停；训练配置未调优，验证结果与原实验一致。保存加载及实际预测通过校验。
- 三方法按 sample_id 对齐，共同 val 成功集 1,782 条。完整 val 请求 1,783 条的 coverage 另报；补充唯一参考上下文匹配共同集 1,767 条的敏感性结果。
- EDA 5 张与三模型误差 / 案例 3 张图已更新，源表、哈希、人数和人工图像检查记录齐全。
- 统一 CLI、版本配置生成器、当前结果索引、任务状态、依赖和交接文档已提供。默认不生成 test 成绩；新运行拒绝覆盖非空输出。

## 当前验证集比较

以下全部使用相同 1,782 条 val 样本、log10 strength 尺度，为 preliminary：

| 方法 | log10 R² | Spearman | log10 MAE | 完整请求成功率 |
|---|---:|---:|---:|---:|
| KNN | 0.017841 | 0.123234 | 0.467671 | 1783/1783 |
| 热力学（train 校准） | 0.029041 | 0.151955 | 0.464151 | 1782/1783 |
| CNN | 0.054623 | 0.207132 | 0.457034 | 1783/1783 |

CNN/KNN 输入来自原 50 bp，热力学使用补取的 150 bp 参考上下文，信息范围不同。单次训练结果不能证明稳定优劣。修复后预测能力仍有限，不能因为工程验收通过就宣称模型表现强或发现生物学因果机制。

## 验收

- 76 项测试通过：集成 5、CNN 14、分析 9、原接口 48；非空输出覆盖防护实测通过。
- 主表、split、transform、各模型预测与运行 manifest 校验通过。
- KNN 全 val 保存加载精确一致；CNN 全 val checkpoint 重载数值一致。
- 校准参数由独立 numpy 最小二乘复算，确认只使用 train。
- 三模型 raw/log10 指标由 sklearn/scipy 独立复算；误差分析的 38 项指标复算通过。
- 图像检查确认标签、尺度、人数与布局，未把工具推断当作实验注释。

## 管理入口

从仓库根目录、已安装 requirements.txt 的环境运行：

```powershell
python -m pr02 validate-data
python -m pr02 verify
python -m pr02 new-run --tag next_experiment_v1
python -m pr02 --config configs/integration_next_experiment_v1.json run-all
```

当前版本由 project_config.json 和 results/current.json 注册。模型 / 原始输出在不可覆盖的运行目录，results 是可更新的当前别名；别名与索引哈希也由 verify 检查。

- results/performance_summary.csv、metrics.csv、coverage_summary.csv、common_eval_ids.tsv：共同成绩、覆盖与精确 ID。
- reports/integration_validation.json、test_summary.json、test_logs/：验收证据。
- runs/eda_m2_integrated_20261002_v2、runs/error_m3_integrated_20261002_v3：当前图和源表。
- history、原 KNN / thermo 结果与 CNN v1：保留历史，不作为当前标准入口。

成员在自己的分支开发；新增方法交付完整 val 预测、train-only 模型、配置、失败状态和 manifest，再注册至统一配置。每次方法 / 数据版本变化都重新生成评价集合，不复用旧 comparison_set_id。

## 分工与待办

| 成员 | 当前可验收交付 | 尚需继续 |
|---|---|---|
| 李宇飞 | 主表、固定 split、共享 transform、已修复的 KNN 与统一评测接口 | 维护公共数据与跨方法评价；接入后续 Ridge |
| 田惠今 | 已修复并重跑的热力学、校准、失败与覆盖记录 | 说明多重定位和实验上下文适用性 |
| 陈思远 | CNN、三线集成、统一管理与验收 | M4 泛化对照、归因与突变接口 |
| 胡昊铭 | EDA、三模型误差 / 案例、注释来源核查 | 恢复可靠区域注释并支持盒区关系分析 |
| 李玘航 | 本次远端尚未见 k-mer Ridge 交付 | 补交组内承诺的 Ridge 或明确调整范围 |

工程问题已修复。可靠逐样本 TSS / 实验方向 / 盒区坐标、强度单位与实验条件仍未恢复；这些是研究资料待办，不能通过代码合并补造。教师 M3 要求简单 ML + 热力学，KNN 可承担简单 ML；组内另承诺的 Ridge 仍需处理。教师 M2 盒区分析与 M4 生物学解释的证据边界见 reports/OPEN_ITEMS.md。
