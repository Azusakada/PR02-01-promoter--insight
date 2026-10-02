> 历史交付记录：保留原实验和来源核查。当前集成状态、修复结果与有效运行见仓库根 reports/PROJECT_STATUS.md、results/current.json；本文件中的旧运行分数和哈希不作为当前默认入口。

# 接口交接记录（thermo/）

- **task_id**：pr02-thermo-m2m3
- **接口/产物**：热力学对照基线（契约表 `calculator_inputs` / `predictions` / `metrics`）
- **输入文件与版本**：
  - `data/01_Ecoli_strength/data_v1.tsv`（`data_version=ecoli50_strength_v1`，11884 条）
  - `data/01_Ecoli_strength/split_manifest.tsv`（`split_id=ecoli50_random_20260928_v1`）
  - 参考基因组 E. coli K-12 MG1655 `GCF_000005845.2 ASM584v2 / NC_000913.3`
- **输出文件与版本**：
  - `thermo/input/context_map.tsv`（上下文恢复，input_version=thermo_input_v1）
  - `thermo/input/calculator_inputs.tsv`（ready=11883 / blocked=1）
  - `thermo/results/predictions/thermo_{train,val,test}.csv`（契约 predictions，run_id=thermo_regseq2_ecoli50_v1）
  - `thermo/results/predictions/thermo_{train,val,test}_calibrated.csv`（train-only 校准后）
  - `thermo/results/calculator_failures.csv`（1 条）
  - `thermo/results/calculator_raw_outputs/thermo_raw_outputs.jsonl`
  - `thermo/results/thermo_metrics.csv`、`thermo_calibration.json`、`thermo_run_manifest.json`
- **实际命令**：
  ```bash
  python thermo/recover_context.py
  python thermo/build_thermo_inputs.py
  python thermo/run_thermo.py
  python thermo/evaluate_thermo.py
  ```
- **验证**：
  - smoke test：50 bp→0 有效 TSS；100 bp→23；150 bp→73（`thermo/smoke/`）。
  - 全量覆盖：11883/11884 成功，仅 `ecoli50_r003290` 不可定位（blocked）。
  - 指标：test Spearman(tool_rank)=0.231，校准后 log10 R²=0.054。
- **已知限制/失败**：
  - 50 bp 本身不适用，依赖参考基因组恢复的上下文；75 条多命中（取最小坐标）；
    1 条不可定位；Tx_rate 与 strength 标尺不同，未校准不做数值比较；
    工具为开源 regseq2，与付费网页版数值不保证一致。
- **下游读取方式**：
  - 统一评测：按 `sample_id` join `predictions/thermo_*.csv`，用 `predicted_tx_rate`
    做 tool_rank Spearman，或用 `predictions/thermo_*_calibrated.csv` 做校准后 R²。
  - 归因：热力学工具无逐位点贡献，不参与 RQ2。
