# 热力学基线

regseq2 对原始 50 bp 无合法状态。保留已有参考基因组补取输入：原序列在 150 bp 中段，11,808 唯一匹配、75 多重匹配、1 未找到。
精确定位不等于已恢复实验 TSS / 方向 / 上下文；来源限制见 `reports/OPEN_ITEMS.md` 和注释核查报告。

```powershell
python -m pr02 run-thermo
python -m pr02 evaluate
```

新版生成器与统一运行使用 `input/calculator_inputs_both_strands_v2.tsv`，显式声明 `scan + both_strands`；旧 `calculator_inputs.tsv` 为历史运行输入，其方向字段描述已在新版修正。新执行器拒绝把单方向或固定 TSS 请求静默按双方向扫描。序列、原 50 bp 位置及失败请求保持一致。

前者默认运行 ID 已存在时拒绝覆盖。新实验先 `python -m pr02 new-run --tag 新的英文tag`，用新配置运行。
旧 run_thermo.py / evaluate_thermo.py 是以上入口的兼容包装，不再接受 --rebuild-from-raw 来复用旧错误结果。

当前结果在 `runs/thermo_regseq2_integrated_20261002_v2/`：

- 正反方向所有候选都保留，以 (方向,TSS) 身份比较最小 dG_total。
- 原始诊断记录选中方向、映射 TSS、反向工具内部坐标和盒区半开区间；这是工具推断，不是实验注释。
- `predictions/thermo_{train,val,test}_raw.csv` 保留 Tx_rate 与全部失败请求。
- `thermo_calibration.json` 只由 train 拟合；对应 calibrated CSV 保留全部请求。
- coverage 按子集计数；run_manifest 含环境、实际 code_commit、配置和产物哈希。
- 默认仅生成 train/val 指标；test 预测保留用于后续冻结评估，未输出 test 成绩。

`thermo/results/` 保留旧 v1，已标记为历史结果。请使用根目录 results/current.json 指定的有效产物。
