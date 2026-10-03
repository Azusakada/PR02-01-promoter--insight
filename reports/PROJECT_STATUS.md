# PR02 01 五人项目当前交付

日期 2026-10-03。仓库 D:\PR02-01-promoter--insight。默认配置 project_config.json，发布入口 main，成员开发分支保留。

已整合 main 数据/KNN、member-b 热力学、csy CNN、liqihang Ridge/SVR 和 feature/eda-annotation-m2m3。李玘航主线已到位；工程集成和已确认问题修复已完成，可靠注释及 M4 研究仍待推进。

## 当前验证集结果

五方法共同 val 1,782 条，完整请求 1,783 条。唯一参考上下文匹配的共同子集另报 1,767 条。R² 和 MAE 为 log10 strength 尺度，证据级别 preliminary。

| 方法 | log10 R² | Spearman | log10 MAE | 成功请求 |
|---|---:|---:|---:|---:|
| knn_physchem_full | 0.017841 | 0.123234 | 0.467671 | 1783/1783 |
| thermo_regseq2 | 0.029041 | 0.151955 | 0.464151 | 1782/1783 |
| cnn_1d | 0.054623 | 0.207132 | 0.457034 | 1783/1783 |
| kmer3_ridge | 0.052757 | 0.219489 | 0.459778 | 1783/1783 |
| kmer3_svr | 0.009121 | 0.227552 | 0.446457 | 1783/1783 |

CNN/KNN/Ridge/SVR 基于原 50 bp，热力学使用补取的 150 bp，信息范围不同。单次 val 结果不足以证明稳定优劣。SVR 的 Spearman 较高、MAE 较低，但 R² 较低，因此不能只用单一指标概括其表现。

## 本轮修复与核验

- 替换 Ridge 私人路径及自行恢复 split 的默认脚本，统一使用公共冻结输入；独立核对全部 k=3/4/5 计数、词表、行索引及理化特征。
- 修复混合 val/test 标准入口与 log10 输出填写 normalized/transform 的接口错误；保留原始交付 ZIP。
- 沿用原 val 选定 k=3、alpha=100、LinearSVR C=10，在统一依赖环境只用 train 重拟合；重载全部 val，Ridge 系数独立复算。
- 新增模型 run_manifest、哈希、拟合 ID、完整 val 与覆盖；CLI/new-run/run-all 及当前结果索引接入五方法。
- 比较 ID 绑定样本、方法、run_id 与预测哈希；新增模型而共同 ID 不变时也产生新版本。统一误差分析使用相同 ID 定义。
- 修复分析运行对全仓未提交改动的差异捕获遗漏，保持 dirty 标记与 source_patch 记录一致。
- 新增图表注册和过期预测检查；已发布验证报告保持只读，防止跨环境诊断数值变化破坏索引哈希。
- 五方法图改为多行布局，62 项误差指标复算并实际检查 3 张图。已有非空输出拒绝覆盖。

验收为 86 项 unittest 测试与 4 项 k-mer 检查，共 90 项；详细日志见 reports/test_logs。数据、模型、校准、统一指标与图表清单验收见 reports/integration_validation.json。干净检出证据见 reports/clean_checkout_verification.json。

## 成员职责

| 成员 | 当前已交付 | 后续任务 |
|---|---|---|
| 李宇飞 | 公共主表、split、transform、KNN 与统一评测 | 数据维护与跨方法评价 |
| 李玘航 | k-mer 特征、Ridge、SVR、搜索与预测 | 传统 ML 对照及后续泛化 |
| 田惠今 | 热力学推断、train 校准、失败与覆盖 | 多重定位及实验适用性说明 |
| 陈思远 | CNN、五线集成、管理与验收 | CNN 泛化、消融、归因与突变 |
| 胡昊铭 | EDA、注释来源审查、五方法误差与案例 | 恢复可靠区域注释与盒区关系分析 |

## 当前路径

results/current.json 注册唯一当前结果，旧结果归档到 history。不可覆盖的模型运行、五方法比较和误差图由 project_config.json 绑定。新增实验依照 README 创建新 tag，完成图像检查后再 publish。

本轮新增临时文件和检出副本在 D:\CodexAnalysis\PR02-01\20261003-main；复用原有 Python 环境，未新建 C 盘大型运行环境。未清理现有缓存。
