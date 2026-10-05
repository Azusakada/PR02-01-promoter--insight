# CNN 当前交接

负责人：陈思远。更新：2026-10-05。接口 schema 2.0.0，固定数据 ecoli50_strength_v1、划分 ecoli50_random_20260928_v1，公共标签变换 log10mm_59711a1c2217859e。

当前模型为 `cnn_optimization_20261004_round1_avg5_medium_s20260928`，目录见 [README 的交付文件](README.md#交付文件)。checkpoint、model_manifest、实际配置、label_transform、训练/验证 ID、训练日志和验证预测必须配套保留。

统一入口 `project_config.json` 和 `results/current.json` 已绑定选定模型。五方法共同验证 1,782 条，CNN log10 R²=0.073950、Spearman=0.227578、MAE=0.453482；CNN 自己完整验证 1,783 条，R²=0.073958、Spearman=0.227814、MAE=0.453396。样本数不同，数值略有差异。

代码提供 fit_cnn、load_model、predict_sequences 和可微分 Log10Predictor；以 sample_id 对齐，保持 50 个输入位置。当前只完成验证集选模和预测，M4 归因、位点突变、消融与后续泛化仍需推进。

全部 6 个配置、10 次训练和种子复跑已保留。选定模型有完整重载预测和独立核验。复现命令见 [CNN README](README.md)，本轮细节见 [优化报告](optimization_report_20261004.md)。根 `python -m pr02 verify` 同时核对接口、模型身份、指标与分析产物。

2026-10-02 的旧交接、旧模型及验证日志从 Git 历史或 D 盘备份恢复；见 [仓库清理说明](../reports/REPOSITORY_CLEANUP_20261005.md)。
