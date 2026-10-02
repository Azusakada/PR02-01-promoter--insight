> 历史交付记录：保留原实验和来源核查。当前集成状态、修复结果与有效运行见仓库根 reports/PROJECT_STATUS.md、results/current.json；本文件中的旧运行分数和哈希不作为当前默认入口。

# 热力学基线适用性核查（thermo_applicability.md）

> 对应 PR02-01 M2+M3「热力学基线全流程」（田惠今）
> 工具：开源 `regseq2 Promoter_Calculator`（Salis Lab 2022 热力学模型，宿主 E. coli K-12 MG1655，σ70）
> 状态：**已定论并已跑通全量**（11883/11884），非阻塞

## 0. 一句话结论

课程 50 bp 主表**不能直接**喂给 regseq2：该工具扫描一个启动子状态至少需要
**上游 58 bp + 下游 20 bp**（即输入 ≥ 78 bp），50 bp 无法构成任何合法状态（实测 0 个有效 TSS）。
但经核查，**11883/11884（99.99%）的 50 bp 序列都能在 E. coli MG1655 参考基因组上精确定位**，
因此改用「恢复上下文」（两侧各补 50 bp → 150 bp 输入，`input_origin=recovered_context`）
即可正规批量运行。**未使用任何伪造的旧 TSS/坐标**。

## 1. 工具与运行条件（核查结论）

| 项 | 核查结果 |
|---|---|
| 工具 | `regseq2 Promoter_Calculator`（`bnjenner/PromoterCalc_Comparison` 的 `ext/regseq2`，纯 Python，本地免费） |
| 模型 | Salis 2022 线性自由能模型（dG_total → Tx_rate = K·exp(−β·dG_total)） |
| 宿主 | `Escherichia coli str. K-12 substr. MG1655`（K=42.0, β=1.6362），与数据宿主一致 |
| 需要的输入 | 一条 DNA 序列；**不需要**外部 TSS 字段；工具自行扫描 |
| TSS 语义 | `tss_mode=scan`：对每个候选 TSS 计算所有 (disc, spacer) 状态，取 dG_total 最小者 |
| 方向 | 工具对输入的正链与反向互补链都扫描 |
| 运行代价 | ≈0.045 s/样本（150 bp），全量 11884 条 ≈ 950 s |
| 商业版对照 | De Novo DNA 网页版（同后端）需「币」，本组无币无法提交；开源版可替代（M1 已记录） |

## 2. 硬约束推导（为什么 50 bp 不行）

`Promoter_Calculator.predict()` 中：

```
MinimumTSS = UPS(24) + 1 + HEX35(6) + SPACER_min(15) + HEX10(6) + DISC_min(6) = 58
ITR_length = 20
```

即对每个候选 TSS 必须同时满足 `TSS ≥ 58` 且 `TSS + 20 ≤ len(sequence)`。
对 `len=50`：需要 `58 ≤ TSS ≤ 30`，**交集为空** → 没有任何合法状态。

## 3. Smoke test 证据（`thermo/smoke/`）

同一条真实样本序列在不同长度下的有效 TSS 数（`python thermo/smoke/smoke_test.py`）：

| 输入 | 长度 | 有效 TSS 数 | 最优 TSS | 结论 |
|---|---|---|---|---|
| 原始 50 bp | 50 | 0 | — | 不可用 |
| 补 +25/+25 | 100 | 23 | 70 | 可用 |
| 补 +50/+50 | 150 | 73 | 95 | 可用（本项目采用） |

同时记录：50 bp 直接调用不报错但返回空结果（`Forward/Reverse_Predictions_per_TSS` 均为空），
必须在 `input_status` 中标 `blocked`，不能把空值当 0。

## 4. 上下文恢复方案（recovered_context）

1. `recover_context.py`：下载 E. coli K-12 MG1655 参考基因组
   （`GCF_000005845.2 ASM584v2 / NC_000913.3`，sha256 记录于运行日志），
   对每条 50 bp 做正/反向互补精确匹配。
   结果：`found=11808`、`ambiguous=75`、`not_found=1` → `thermo/input/context_map.tsv`。
2. `build_thermo_inputs.py`：以命中坐标为中心，**两侧各补 50 bp**（环形基因组取模），
   构造 150 bp 输入；`source_sequence_start_0index=50`；`context_source_ref` 记录
   `accession:strand:start-end`；`input_origin=recovered_context`。
   结果：`ready=11883`、`blocked=1` → `thermo/input/calculator_inputs.tsv`。
3. `run_thermo.py`：调用 regseq2 批量运行 → 逐样本 Tx_rate + dG 分解 + 失败清单。

**方向处理**：`strand=-` 的样本，输入取该基因组窗口的反向互补，保证原始 50 bp 以原方向落在
输入中段（`build_thermo_inputs.py` 内有断言校验）。

## 5. 标尺与评测

- 工具输出是**转录速率 Tx_rate**，与课程 `strength` **不是同一标尺**，未经校准**只做排序比较**
  （`target_scale=tool_rank`，Spearman），不填入 `predicted_value`（strength 尺度）字段。
- 数值比较仅用 **train-only** 线性校准：`log10(strength) = 0.5341·log10(Tx_rate) + 0.4890`
  （n_fit=8318），再用于 val/test；原始与校准结果分列保存（`thermo_calibration.json`）。
- 正式结果（`thermo/results/thermo_metrics.csv`）：

| 子集 | Spearman(tool_rank) | 校准后 log10 R² | 校准后 log10 MAE | n |
|---|---|---|---|---|
| train | 0.193 | 0.042 | 0.467 | 8318 |
| val | 0.158 | 0.029 | 0.463 | 1782 |
| test | 0.231 | 0.054 | 0.453 | 1783 |
| all | 0.193 | 0.042 | 0.464 | 11883 |

（evidence_level=preliminary；用于 RQ1 与数据驱动方法的同尺度对照。）

## 6. 局限（必须在报告中保留）

1. **方向/位置未知**：主表无逐条 TSS、方向、−10/−35 字段；上下文恢复只给出**基因组坐标**，
   不声称 50 bp 窗口相对天然 TSS 的位置。
2. **多处命中 75 条**：序列在基因组中有多个精确匹配（重复序列/IS 元件），取最小坐标，
   该子集的上下文不唯一，已单独记录（`context_map.tsv` 的 `n_positions>1`）。
3. **不可定位 1 条**（`ecoli50_r003290`）：标 `blocked`，不伪造上下文，保留在失败清单。
4. **工具外推性**：Salis 模型基于自有数据拟合，与课程数据是否重叠未知；热力学模型对
   人工/工程序列未必最优。
5. **标尺不可直接比**：Tx_rate 与 strength 的绝对值不可比；校准关系依赖 train 子集。
6. **商业版差异**：本项目用开源 regseq2 替代付费网页版，两者数值不保证一致。

## 7. 下游交接

- 统一评测：`thermo/results/predictions/thermo_{train,val,test}.csv`（`method_name=thermo_regseq2`,
  `target_scale_model=tool_raw`, `predicted_tx_rate`），比较时用 Spearman 或校准后 R²。
- 归因（RQ2/CNN+IG 主线）：热力学工具**无逐位点梯度/归因**，不参与 RQ2。
