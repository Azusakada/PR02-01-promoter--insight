# E coli 50 bp CNN 中期训练交付

陈思远负责的 CNN 主线实现位于本目录。代码使用仓库已有 `data_v1.tsv` 和固定
`split_manifest.tsv`，训练集拟合标签变换，验证集选择最佳 checkpoint，并交付逐样本
预测。当前是 M2/M3 preliminary；正式深度模型比较在 M4。

## 安装与运行

从仓库根目录运行。Python 3.10 或更高版本；本次实际验证环境为 Python 3.12、
PyTorch 2.14.1 CPU。首次运行建议创建虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r CNN/requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s CNN/tests -v
.\.venv\Scripts\python.exe CNN/run_cnn.py smoke
.\.venv\Scripts\python.exe CNN/run_cnn.py train --config CNN/configs/cnn_optimized_round1.yaml
```

已存在的非空输出目录不会覆盖。重新实验前，将配置中的 `run_id`、`output_dir`
改成新值；不要删除已验收运行。`CNN/runs/` 是本地实验产物，不默认进入 Git。
若 Git 未在 PATH 中，设置 `PR02_GIT` 为 Git 可执行文件路径。

## 模型和输入

- 输入 `float32 (N,4,50)`；通道顺序 A/C/G/T。原序列第 i 个碱基对应输入第 i 列。
- 两层 Conv1d：4→32，kernel 7；32→64，kernel 5；stride 1 和 same-length padding。
  当前默认配置在卷积后做 5 段平均池化，再接 32 维全连接、dropout 0.3 和单值回归头，共 21,537 参数。原 flatten 结构仍受代码和测试支持；旧运行从 Git 历史或 D 盘归档恢复。
- 输出是 train-only min-max 的 log10 strength。末层不加 sigmoid，预测值不 clip。
- 模型不猜 TSS、方向或 −10/−35 坐标，也不截取、补齐或 reshape 旧 150 bp 输入。
  本仓库没有旧 CNN，当前是新增的 50 bp 实现。
- `Log10Predictor` 是可微分的 log10 输出包装，供 M4 后续归因使用；输入梯度为 `(N,4,50)`。

## 一轮有限优化

```powershell
python CNN/optimize_cnn.py --study-config CNN/configs/cnn_optimization_round1.yaml
```

该计划预先固定 6 个配置、一个搜索种子和两个额外复跑种子。搜索比较小网络、
更强正则化和位置池化；按验证 MSE 选出两个配置，再比较三个种子的平均验证 MSE。
交付选定配置的原搜索种子模型，不挑分数最高的种子。全部尝试、训练日志、逐样本
验证预测和选择依据保存到计划的独立输出目录。输出非空时拒绝覆盖；复现实验须
给计划设置新的 study_id/output_dir。测试集不参与本轮模型选择。

模型新增可选 `pooling: adaptive_avg` / `adaptive_max` 和 `pooled_length`（1 到 50）。
它们在两层卷积后压缩位置维度，减少全连接层参数；输入及输入梯度仍为 `(N,4,50)`。
不提供这些字段时保持原有 flatten 结构，原 checkpoint 可以继续加载。
本轮选择后仍属 preliminary；同一验证集同时用于配置选择和结果报告，不能当作
独立测试效果。完整结果见 `optimization_report_20261004.md`。

## 固定数据和标签尺度

当前输入路径和 split_id 在 `configs/cnn_optimized_round1.yaml` 中显式声明；全组默认配置绑定同一文件。所有连接使用
sample_id，禁止用当前行号建立训练/预测对应关系。数据身份、完整主表 SHA256、
split 文件 SHA256、训练和验证 ID 集合分别绑定到模型元数据。

集成配置显式使用 `data/01_Ecoli_strength/label_transform.json`。代码核对全部训练 ID、
完整主表哈希和 a/b，不重新拟合或静默接受不同版本。主表和 split 的工作区字节统一为 LF，
由 `.gitattributes` 固定，避免跨平台换行导致哈希冲突。`transform: null` 仅用于独立测试数据。

```text
z = log10(strength)
y = (z - a) / (b - a)
predicted_z = a + (b - a) * model(x)
predicted_strength = 10 ** predicted_z
```

val/test 使用同一 a/b，允许 normalized 标签或预测超出 [0,1]。
反变换溢出、下溢或非有限预测会保留 failed 行及原因，不填 0。

## 训练和验证

当前配置采用 AdamW、MSE、batch 128、lr 0.001、weight decay 0.001、最多 100 epochs、patience 15。原基线使用另一份固定配置，各运行保留实际参数。
train shuffle 使用显式种子的 generator；val 不 shuffle；drop_last=false。
验证 MSE 的最小值决定 best checkpoint。早停只由 val 决定，不读取 test 性能，
也不将 train+val 合并后重训。随机种子、设备、线程数和确定性选项写入配置和 manifest。

smoke test 使用独立模型副本验证 forward、loss、backward、参数更新、save/load 和重复
预测一致性，之后重置随机状态进入实际训练。smoke 不是性能成绩。同一环境的重载预测
逐值一致；跨设备或不同软件版本的数值复现仍需重跑核验。

## 交付文件

当前集成运行位于 `CNN/runs/cnn_optimization_20261004_round1/cnn_optimization_20261004_round1_avg5_medium_s20260928/`。本轮完整 10 次训练均保留，避免丢失配置选择和种子复跑证据。旧基线、smoke 和旧验证记录已移出工作树；恢复见 [清理说明](../reports/REPOSITORY_CLEANUP_20261005.md)。当前产物：

| 文件 | 用途 |
|---|---|
| best_checkpoint.pt | 验证集选出的最佳权重和绑定元数据 |
| model_manifest.json | 模型输入、数据、split、transform、ID和哈希 |
| cnn_run_config.yaml | 本次实际训练配置 |
| label_transform.json | 与模型绑定的 train-only 标签变换 |
| train_ids.tsv 和 val_ids.tsv | 训练/验证样本名单 |
| training_history.csv 和 training_log.jsonl | 每个 epoch 的实际 loss 和耗时 |
| training_curve.png | 基于 training_history.csv 生成的曲线 |
| save_load_smoke_test.json | 前向、反向和保存加载证据 |
| run_manifest.json | 命令、环境、代码 commit、产物哈希 |
| source_patch.diff 和 source_snapshot.zip | 未提交开发代码的精确运行快照 |
| validation/predictions_cnn.csv | 每条 val 样本的原始、log10和normalized预测与状态 |
| validation/metrics_cnn.csv | raw/log10 R²、Spearman、MAE，含样本数和preliminary等级 |
| validation/common_eval_ids_cnn.tsv | 本次成功且参与指标计算的ID |
| validation/coverage_cnn.csv | 请求、成功、失败和实际参与样本数及coverage |

逐样本预测采用 schema 2.0.0。全组校验器见仓库根 `contracts/README.md`。
共同有效子集、覆盖率和性能表由 `python -m pr02 evaluate` 统一生成。

## 独立加载和重复预测

预测输入为完整的某个固定 subset，可重新导出到 `work/cnn_reload_inputs_NEW/val.tsv`（见下方命令），
或含 sample_id 和 sequence 的同一 subset 表。未知/重复/不完整 ID 或版本冲突会中止；
单条非法序列或推断失败保留 failed 行。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'CNN'
python -c "from pathlib import Path; from pr02_cnn.common import config; from pr02_cnn.cli import prepared_inputs; prepared_inputs(config(Path('CNN/configs/cnn_optimized_round1.yaml')), Path('work/cnn_reload_inputs_NEW').resolve())"
python CNN/run_cnn.py predict --model-dir CNN/runs/cnn_optimization_20261004_round1/cnn_optimization_20261004_round1_avg5_medium_s20260928 --input work/cnn_reload_inputs_NEW/val.tsv --config CNN/configs/cnn_optimized_round1.yaml --output CNN/runs/cnn_reload_NEW
```

Python API 与附件签名一致：

```python
from pr02_cnn.train import fit_cnn
from pr02_cnn.predict import predict_sequences, load_model
# fit_cnn(train_table, val_table, transform_path, config_path, output_dir) -> ModelArtifact
# predict_sequences(input_table, model_artifact, run_config_path, output_dir) -> PredictionBundle
```

导入前将 `CNN/` 加入 PYTHONPATH。checkpoint、manifest、config 必须作为一组保留。
当前 CLI 服务于固定数据的 train/val/test；未测突变预测和 IG 在 M4 按对应设计接口扩展。

test 默认关闭。冻结方案后，如需要 M4 test 评价，复制配置、设置 `prediction.subset: test`
和 `prediction.allow_test: true`，用独立输出目录执行 predict。当前交付的中期分数只来自 val。

## 接口验收命令

```powershell
python contracts/scripts/validate.py bundle --samples data/01_Ecoli_strength/data_v1.tsv --splits data/01_Ecoli_strength/split_manifest.tsv --transform CNN/runs/cnn_optimization_20261004_round1/cnn_optimization_20261004_round1_avg5_medium_s20260928/label_transform.json --predictions CNN/runs/cnn_optimization_20261004_round1/cnn_optimization_20261004_round1_avg5_medium_s20260928/validation/predictions_cnn.csv
python contracts/scripts/validate.py table metrics CNN/runs/cnn_optimization_20261004_round1/cnn_optimization_20261004_round1_avg5_medium_s20260928/validation/metrics_cnn.csv
python contracts/scripts/validate.py run CNN/runs/cnn_optimization_20261004_round1/cnn_optimization_20261004_round1_avg5_medium_s20260928/run_manifest.json --root .
python contracts/scripts/validate.py run CNN/runs/cnn_optimization_20261004_round1/cnn_optimization_20261004_round1_avg5_medium_s20260928/validation/run_manifest.json --root .
python contracts/scripts/validate.py self-test
# 生成新实验的图时指定其目录；已交付运行的 analysis 非空，不重复覆盖。
# python CNN/plot_cnn_diagnostics.py --run-dir CNN/runs/你的新运行
```

结果解读与交接分别见 `results_report.md` 和 `handoff.md`。任何具体性能数值以运行产生的
CSV 为准，不使用 smoke 指标充当中期成绩。

实现参考：[PyTorch 模型保存和加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、
[Conv1d 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)。
