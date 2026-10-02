# 热力学基线（thermo/）— PR02-01

课程 E. coli 50 bp `strength` 回归的**热力学对照基线**，使用开源
`regseq2 Promoter_Calculator`（Salis Lab 2022 热力学模型，宿主 MG1655，σ70）。

> 详细适用性核查与限制见 [`thermo_applicability.md`](thermo_applicability.md)。

## 结论速览

- 50 bp **不能直接**运行该工具（需要 ≥78 bp：上游 58 bp + 下游 20 bp）。
- 99.99%（11883/11884）的 50 bp 能在 E. coli MG1655 参考基因组精确定位，
  据此补两侧各 50 bp 上下文构造 150 bp 输入，**已跑通全量**。
- 工具输出 Tx_rate 与 `strength` 不同标尺：未经校准只做 Spearman，校准仅用 train 拟合。

## 目录

```
thermo/
├── thermo_applicability.md      适用性核查报告（结论/证据/限制）
├── README.md                    本文件
├── handoff.md                   接口交接记录
├── recover_context.py           50 bp → 参考基因组坐标（context_map.tsv）
├── build_thermo_inputs.py       主表 → 工具输入（calculator_inputs.tsv）
├── run_thermo.py                批量调用 regseq2 → 预测/失败/原始输出
├── evaluate_thermo.py           Spearman + train-only 校准 → 指标
├── config/thermo_config.json    运行配置
├── input/                       context_map.tsv, calculator_inputs.tsv
├── results/                     预测、失败、原始输出、指标、run_manifest
├── smoke/                       50 bp 适用性 smoke test 及输出
└── tools/regseq2/               开源热力学模型（vendored）
```

## 复现（从仓库根目录）

```bash
# 1) 恢复基因组上下文（首次会缓存参考基因组到 thermo/.cache/，不提交）
python thermo/recover_context.py

# 2) 构建工具输入
python thermo/build_thermo_inputs.py

# 3) 批量运行热力学工具（全量约 16 分钟；可加 --limit 100 调试）
python thermo/run_thermo.py
#    若已有 results/calculator_raw_outputs/thermo_raw_outputs.jsonl，可只重建预测表：
#    python thermo/run_thermo.py --rebuild-from-raw

# 4) 评测
python thermo/evaluate_thermo.py
```

依赖：Python 3 + numpy + scipy（regseq2 另需 biopython；`viz.py` 未使用）。

## 关键产物（契约 schema 2.0.0）

| 文件 | 说明 |
|---|---|
| `input/context_map.tsv` | 逐样本基因组定位结果（match_status/strand/坐标） |
| `input/calculator_inputs.tsv` | 契约表 `calculator_inputs`：工具输入与状态 |
| `results/predictions/thermo_{train,val,test}.csv` | 契约表 `predictions`（未校准）：每请求一行，含 Tx_rate/失败 |
| `results/predictions/thermo_{train,val,test}_calibrated.csv` | 契约表 `predictions`（train-only 校准后） |
| `results/calculator_predictions_all.csv` | 未校准预测合并件（非契约表，便于人工查看） |
| `results/calculator_failures.csv` | 失败/阻塞清单 |
| `results/calculator_raw_outputs/thermo_raw_outputs.jsonl` | 原始结果（最佳 TSS、dG 分解） |
| `results/thermo_metrics.csv` | 契约表 `metrics`：tool_rank Spearman + 校准后 R²/MAE |
| `results/thermo_run_manifest.json` | 运行元数据（版本/命令/覆盖） |
