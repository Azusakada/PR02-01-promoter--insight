# PR02-01 Promoter Insight

本仓库保存 PR02-01 M2/M3 阶段的可复现数据与实验交付物。

当前集成开发在 `csy`：接入 main、member-b 和 EDA 分支，统一代码与验收；不直接修改远端 main。
当前有效配置是 [project_config.json](project_config.json)，共同结果见 `results/`，验证和任务状态见 `reports/`。

## 统一入口

建议 Python 3.12。使用 CPU PyTorch 的实际验证环境，依赖版本固定在 `requirements.txt`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pr02 validate-data
.\.venv\Scripts\python.exe -m pr02 verify
```

Git 须在 PATH，或设置环境变量 `PR02_GIT` 指向 Git 可执行文件。

已有结果不可覆盖。新实验用一个新 tag 自动生成配套配置：

```powershell
python -m pr02 new-run --tag my_experiment_v1
python -m pr02 --config configs/integration_my_experiment_v1.json run-all
```

新配置完成验收并发布结果后，把它设为全组默认，并显式提交已验收运行。
运行目录默认忽略；普通 `git add` 不会包含新模型，须按配置逐个登记：

```powershell
Copy-Item -LiteralPath configs/integration_my_experiment_v1.json -Destination project_config.json
git add project_config.json configs CNN/configs results reports
git add -f -- runs/knn_my_experiment_v1 runs/thermo_my_experiment_v1 runs/comparison_my_experiment_v1 CNN/runs/cnn_my_experiment_v1
```

单独步骤：`python -m pr02 run-knn`、`python -m pr02 run-thermo`、
`python CNN/run_cnn.py train --config CNN/configs/cnn_run_config.yaml`、`python -m pr02 evaluate`。
默认配置对应已交付运行；再次运行会拒绝覆盖，须先生成新 tag。

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s CNN/tests -v
python -m unittest discover -s analysis_m2m3/tests -v
python contracts/scripts/validate.py self-test
```

## E. coli strength 数据

- `data/01_Ecoli_strength/data_v1.tsv`：统一的 50 bp E. coli strength 主表。
- `data/01_Ecoli_strength/split_manifest.tsv`：固定 train/val/test 划分。
- `data/01_Ecoli_strength/label_transform.json`：训练集拟合的 `log10(strength)` 标签变换契约。
- `data/01_Ecoli_strength/data_cleaning_log.md`：数据清洗、标签处理和运行记录。

## E. coli KNN 回归

`KNN/results/` 同时保留带实验后缀的历史文件和以下标准交付文件：

- `knn_config.json`、`knn_model.joblib`
- `predictions_knn.csv`
- `metrics.csv`、`common_eval_ids.tsv`、`coverage_summary.csv`

`predictions_knn.csv` 标准入口只包含 val；log10 模型不填写 normalized / label_transform_id。
模型按 train 拟合并保存，val 用于原 k 的选择和初步评价，不做 train+val 重拟合。
带 `ecoli_full_20260929` 后缀的文件是历史实验，不与当前默认结果混用。

## E. coli CNN 回归

`CNN/`：50 bp A/C/G/T one-hot CNN，固定数据和划分、train-only 标签变换、验证选模、checkpoint 保存加载及逐样本预测。见 [CNN/README.md](CNN/README.md)。

## 六物种数据与 KNN 冒烟结果

- `data/02_reg_and_gen_six_species/`：Bacillus subtilis、Baumanii、Bradyrhizobium、Diphtheria、Escherichia coli、Staphylococcus 六个独立二分类数据集，以及汇总表和清洗日志。
- `KNN/six_species/`：六物种独立 KNN 分类的配置、模型、逐样本预测、指标、覆盖率、共同评估 ID 和冒烟日志。该结果证据级别为 `smoke`，不与 E. coli 连续 strength 回归混用。

## 热力学基线

`thermo/` 使用开源 regseq2，参考基因组上下文补取后运行；Tx_rate 与课程 strength 的标尺不同，未经校准只做排序比较。见 [thermo/README.md](thermo/README.md)。

## 统一评价和交接

所有方法使用同一 data_v1 和 split。CNN/KNN 仅在 train 拟合，CNN 在 val 选 checkpoint，
KNN 沿用已发布 val 搜索选出的 k=1501；热力学校准仅在 train 拟合。默认不输出 test 成绩。
按 sample_id 对齐各方法成功预测的交集，主比较同样本、同尺度；完整请求的 coverage 单独报告。
另报唯一参考基因组匹配子集的敏感性结果。热力学使用 150 bp 补取上下文，CNN 使用 50 bp，
因此同样本比较不意味着输入信息量相同。

- `results/current.json`：当前有效运行、文件路径和哈希索引。
- `results/performance_summary.csv`、`metrics.csv`、`coverage_summary.csv`、`common_eval_ids.tsv`：统一成绩和精确评价集合。
- `reports/integration_validation.json`：数据、运行、校准、模型重载及独立指标复算。
- `reports/PROJECT_STATUS.md`：分工、验收与交接。
- `reports/OPEN_ITEMS.md`：可靠注释、测量来源、Ridge 和 M4 待办。
- `runs/eda_m2_integrated_20261002_v2/`、`runs/error_m3_integrated_20261002_v3/`：当前 EDA 和三方法误差 / 案例图，附源表与人工图像 QA。
- `history/`、旧运行目录：历史版本，不能当作当前标准产物。

团队修改共同数据 / split / transform 前须发布新版本；新增方法提交完整 val 预测、失败状态、
train-only 模型及 run_manifest，再注册到统一配置。每次新增方法或数据版本都重新生成评价集合。
提交前执行适用测试和 `python -m pr02 verify`；用自己的开发分支提交集成变更，验收后再通过 PR 进入 main。
