# E. coli 数据与固定划分

- 11,884 条大写 A/C/G/T 50 bp 序列，稳定 source_row 与 sample_id；strength 为正连续值。
- 只保存 strength 和 log10(strength)；原值、样本与划分未因集成修复而改变。
- split_id=ecoli50_random_20260928_v1，seed=20260928；train=8318，val=1783，test=1783。
- 公共 log10 min-max 参数只由 train 拟合，不裁剪 val/test，transform_id=log10mm_59711a1c2217859e。
- 主表、split 和 transform 的工作区换行固定为 LF；`.gitattributes` 保证跨平台检出一致。修复前文件字节备份见 history/data_bytes_before_lf_20261002_v1。
- 不再把全量 KNN 的标准文件描述成 k=5、512/128 的 smoke。当前 KNN 使用原 val 搜索选择的 k=1501，train-only 保存模型，标准预测只含 val。
- 当前运行和全组指标由 project_config.json、results/current.json、reports/PROJECT_STATUS.md 统一索引。
- 六物种数据为独立 0/1 识别任务，不并入此 strength 回归。
- 可靠 TSS / 方向 / 盒区与测量条件仍待核实；主表 missing 注释状态不被工具推断替代。
