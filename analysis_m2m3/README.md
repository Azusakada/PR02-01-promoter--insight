# M2 数据探索与 M3 误差分析

本模块对应胡昊铭的 EDA、注释核查、预测可视化和案例分析。输入使用 main 上的正式 50 bp `data_v1.tsv` 和固定 split；所有新增产物集中在本目录，方便组长独立合并。

## 环境与运行

Python 3.11，依赖见 requirements.txt。仓库根目录运行：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r analysis_m2m3/requirements.txt
.venv/bin/python -m unittest discover -s analysis_m2m3/tests -v
.venv/bin/python analysis_m2m3/eda.py --output analysis_m2m3/runs/eda_m2_20261001_v2
.venv/bin/python analysis_m2m3/error_analysis.py \
  --predictions KNN/results/predictions_knn_ecoli_full_val.csv \
    analysis_m2m3/inputs/thermo_member_b_89fcc66/thermo_val.csv \
  --calibrated analysis_m2m3/inputs/thermo_member_b_89fcc66/thermo_val_calibrated.csv \
  --calibration analysis_m2m3/inputs/thermo_member_b_89fcc66/thermo_calibration.json \
  --calibration-train analysis_m2m3/inputs/thermo_member_b_89fcc66/thermo_train.csv \
  --output analysis_m2m3/runs/error_m3_val_20261001_v2
```

已发布 run 目录非空时程序拒绝覆盖。复现时将 `--output` 改为新的 run 名，例如 `eda_m2_reproduce_v1`，不要删除原产物。图表默认导出 PNG 和 SVG，并保存数值源表、逐图 manifest、运行 manifest 和简短中文报告。

## 输入来源

- data_version：`ecoli50_strength_v1`。
- split_id：`ecoli50_random_20260928_v1`。
- KNN val：main 提交 `8ba8e215eae56e034c6ff2d6846aabdb276d3bd4`，train-only 预测。
- 热力学快照：`member-b` 提交 `89fcc6600185100eac511de2a27fc0784774604c`；字节原样保存于 `inputs/thermo_member_b_89fcc66/`，来源路径与 hash 见 import_manifest.json。未合并队友业务代码。
- 原始数据溯源：作者 [Promoter_design](https://github.com/wangxinglong1990/Promoter_design) 固定提交 `56c1b331bfbca0665b2c3f6b02345de794fe25df`；11,884 条序列（仅转大写）及标签全部按原始行匹配。核查结果见 `inputs/source_audit_20261001/`。

快照和溯源可重新生成到新目录：

```bash
.venv/bin/python analysis_m2m3/import_thermo.py \
  --ref 89fcc6600185100eac511de2a27fc0784774604c \
  --output analysis_m2m3/inputs/thermo_reproduce
.venv/bin/python analysis_m2m3/source_audit.py \
  --output analysis_m2m3/inputs/source_audit_reproduce
```

## 归一化与特征接口

没有 `--transform` 时，EDA 只在固定 train 上计算 min/max，将分析用 transform 和 targets 保存在当前 run。标记 `analysis_only_pending_team_adoption`，不修改全组数据入口或要求其他模型采用。李宇飞发布正式文件后，使用 `--transform 路径`；程序核验数据/split/训练 ID/hash、参数和 clip=false。

本轮 GC/AT、A/C/G/T 比例、entropy 直接从序列计算。外部 8 维理化表可用 `--features` 接入，但必须同时提供 `--feature-index`（TSV，row_index、sample_id），并通过与序列直接计算值的核对；不依赖未知的矩阵行序。

## 预测接入与比较口径

error_analysis.py 的 `--predictions` 可接收任意数量的团队 predictions CSV，每个文件一种方法/一种 val 子集。每个请求 ID 必须有成功或失败行。校准热力学表遗漏失败行时，使用完整原始表加 `--calibrated/--calibration/--calibration-train`；程序验证 train-only 系数并保留失败记录。

未校准工具只计算排序指标，不计算 strength R²/MAE，不生成数值残差图；模型数值预测在 raw/log10 共同尺度上分析。每方法报告全部成功子集与共同有效子集，保存精确评价 ID、n_requested/n_success/n_used。低中高边界固定为 train log10 的三分位点。默认只分析 val。

Ridge 交接后把其正式 val predictions 路径加入 `--predictions`，创建新 run；CNN 可选加入。当前结果只有 KNN+热力学，最低中期的 Ridge+热力学对照仍待补齐。

## 交付内容与证据边界

- `runs/eda_m2_20261001_v2/eda_summary.md`：分布、特征关系和分析用归一化。
- `reports/annotation_source_review.md`：当前证据、源头匹配、可靠注释缺口及需确认问题。
- `runs/error_m3_val_20261001_v2/m3_error_analysis.md`：验证集公平比较、分组误差、案例与失败说明。
- `reports/handoff.md`：交接、测试和剩余依赖。

缺失注释不输出伪造区域框，相关性与计算预测不构成生物学因果证据。本模块遵守组长 pr02-team 2.1.0 接口包；Skill 原包未安装进共享仓库，避免多人同时改公共规则文件。

只读核验已发布产物：

```bash
.venv/bin/python analysis_m2m3/verify_run.py analysis_m2m3/runs/eda_m2_20261001_v2
.venv/bin/python analysis_m2m3/verify_run.py analysis_m2m3/runs/error_m3_val_20261001_v2
```

`--acknowledge-manually-reviewed` 仅用于实际逐张看过图之后填写视觉检查记录，不能由自动测试冒充人工检查。
