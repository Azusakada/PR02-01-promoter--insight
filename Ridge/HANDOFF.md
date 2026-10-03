# 李玘航交付的集成交接

2026-10-03，来源分支 liqihang，提交 495ee435774a1eab31bbfb8b1d2aa8f166873b7f。k-mer Ridge 和补充 SVR 已完成交付并进入全组默认配置。

公共样本、固定 split 与 transform 一致；全部 k=3/4/5 特征及原模型 val 重载核对通过。本次使用统一依赖、原选定参数和 train-only 数据重新拟合，保存规范的 val 预测、训练 ID、模型及 run_manifest。原私人路径、混合子集标准入口和不适用的 normalized 字段已修复。

当前路径、哈希、覆盖、共同评价 ID 和 EDA/误差分析清单见 results/current.json。全组成绩使用共同 val 1,782 条，Ridge/SVR 完整请求均为 1,783 条且全部成功。旧 test 报告保存在原交付 ZIP，未用于本次选参或统一评价。

后续在成员分支开发，依照根 README 的新配置和验收流程交付。新增搜索需要新记录和运行版本，不能覆盖既有输出。
