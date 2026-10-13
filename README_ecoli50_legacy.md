# PR02-01 Promoter Insight

本仓库保存 PR02-01 M2/M3 阶段的可复现数据与实验交付物。

统一整合版本包含五方法基线、CNN 优化、区域分析修订和接口修复。`csy` 用于集成开发，`main` 用于全组发布；成员继续在各自分支开发。
当前有效配置是 [project_config.json](project_config.json)，共同结果见 `results/`，验证和任务状态见 `reports/`。

## 统一入口

建议 Python 3.12。使用 CPU PyTorch 的实际验证环境，依赖版本固定在 `requirements.txt`。

```powershell
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.pr02_cache\pip'
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

新配置的 run-all 会生成五模型结果、统一成绩和误差图。实际查看图像后执行下面的核验，再发布；未经过视觉检查时不会更新当前结果索引。

```powershell
python analysis_m2m3/verify_run.py runs/error_my_experiment_v1 --acknowledge-manually-reviewed m3_prediction_scatter m3_residuals m3_group_errors
python -m pr02 --config configs/integration_my_experiment_v1.json publish
```

完成验收并发布结果后，把新配置设为全组默认，并显式提交已验收运行。
运行目录默认忽略；普通 `git add` 不会包含新模型，须按配置逐个登记：

```powershell
Copy-Item -LiteralPath configs/integration_my_experiment_v1.json -Destination project_config.json
git add project_config.json configs CNN/configs KNN/results Ridge/results results reports
git add -f -- runs/knn_my_experiment_v1 runs/thermo_my_experiment_v1 runs/comparison_my_experiment_v1 runs/ridge_my_experiment_v1 runs/svr_my_experiment_v1 runs/error_my_experiment_v1 CNN/runs/cnn_my_experiment_v1
```

单独步骤：`python -m pr02 run-knn`、`python -m pr02 run-thermo`、
`python -m pr02 run-ridge-svr`、`python CNN/run_cnn.py train --config CNN/configs/cnn_optimized_round1.yaml`、
`python -m pr02 evaluate`、`python -m pr02 analyze`。
默认配置对应已交付运行；再次运行会拒绝覆盖，须先生成新 tag。

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s CNN/tests -v
python -m unittest discover -s analysis_m2m3/tests -v
python contracts/scripts/validate.py self-test
python Ridge/src/test_kmer.py
```

## E. coli strength 数据

- `data/01_Ecoli_strength/data_v1.tsv`：统一的 50 bp E. coli strength 主表。
- `data/01_Ecoli_strength/split_manifest.tsv`：固定 train/val/test 划分。
- `data/01_Ecoli_strength/label_transform.json`：训练集拟合的 `log10(strength)` 标签变换契约。
- `data/01_Ecoli_strength/data_cleaning_log.md`：数据清洗、标签处理和运行记录。

## 数据处理复现与中期材料

原始课程 NPY 数组保存在 `data/raw_ecoli50/`，来源、哈希和划分规则在 `configs/data_ecoli50_v1.json`。`pr02/data.py` 重建主表、固定划分、清洗逐行记录和 train-only 标签变换；实测主表及 split 与当前版本逐字节一致。

```powershell
python -m pr02 rebuild-data --output runs/data_rebuild_NEW
```

- [中期报告与演示安排](reports/MIDTERM_REPORT_20261005.md)：统一成绩、数据分析、分工、汇报顺序和演示命令。
- [仓库审查与修复清单](reports/REPOSITORY_AUDIT_20261005.md)：老师要求的逐项验收、接口修复、目录职责和剩余交付。

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

## E. coli k-mer Ridge / SVR

`Ridge/` 是李玘航的传统 ML 主线，基于同一份冻结 `data_v1` 和 `ecoli50_random_20260928_v1` 划分：

- `Ridge/features/`：固定词表的 k=3/4/5 计数特征、8 类派生理化特征和 feature manifest
- `Ridge/results/predictions_ridge.csv`：主方法 `kmer3_ridge`（train 拟合、val 选 `alpha=100`）
- `Ridge/results/predictions_svr.csv`：同一 split/标签尺度上的可选 `kmer3_svr`
- `Ridge/results/ridge_search.csv`：全部 k × alpha 候选，不只保留最优结果

当前标准入口只含完整 val，直接 log10 预测不填 normalized/label_transform_id。Ridge/SVR 使用统一依赖环境重新拟合，k=3、alpha=100、LinearSVR C=10 沿用原 val 搜索选择。原交付及其 test 成绩封存在 `history/liqihang_delivery_495ee43.zip`；选择记录保存在 `Ridge/selection/`。复现见 `Ridge/README.md`，证据级别为 `preliminary`。

## 六物种数据与 KNN 冒烟结果

- `data/02_reg_and_gen_six_species/`：Bacillus subtilis、Baumanii、Bradyrhizobium、Diphtheria、Escherichia coli、Staphylococcus 六个独立二分类数据集，以及汇总表和清洗日志。
- `KNN/six_species/`：六物种独立 KNN 分类的配置、模型、逐样本预测、指标、覆盖率、共同评估 ID 和冒烟日志。该结果证据级别为 `smoke`，不与 E. coli 连续 strength 回归混用。

## 热力学基线

`thermo/` 使用开源 regseq2，参考基因组上下文补取后运行；Tx_rate 与课程 strength 的标尺不同，未经校准只做排序比较。见 [thermo/README.md](thermo/README.md)。

## 统一评价和交接

所有方法使用同一 data_v1 和 split。CNN/KNN 仅在 train 拟合，CNN 在 val 选 checkpoint，
KNN 沿用已发布 val 搜索选出的 k=1501；Ridge/SVR 沿用已发布 val 选择且只在 train 拟合；热力学校准仅在 train 拟合。默认入口不输出新的 test 成绩。历史传统基线已经报告过 test，不能把后续同一 test 称为全新盲测。
按 sample_id 对齐各方法成功预测的交集，主比较同样本、同尺度；完整请求的 coverage 单独报告。
另报唯一参考基因组匹配子集的敏感性结果。热力学使用 150 bp 补取上下文，CNN 使用 50 bp，
Ridge/SVR 同样来自原 50 bp，因此同样本比较不意味着所有方法输入信息量相同。
比较 ID 同时绑定方法、run_id、预测文件哈希和样本 ID；新增方法或版本即使共同样本不变，也会生成新 ID。

- `results/current.json`：当前有效运行、文件路径和哈希索引。
- `results/performance_summary.csv`、`metrics.csv`、`coverage_summary.csv`、`common_eval_ids.tsv`：统一成绩和精确评价集合。
- `reports/integration_validation.json`：数据、运行、校准、模型重载及独立指标复算。
- `reports/PROJECT_STATUS.md`：分工、验收与交接。
- `reports/OPEN_ITEMS.md`：可靠注释、测量来源和 M4 待办。
- `runs/eda_m2_integrated_20261002_v2/`、`runs/error_m3_five_methods_20261005_v3/`：当前 EDA 和五方法误差 / 案例图，附源表与人工图像 QA。
- [启动子区域特征与强度分析](analysis_m2m3/reports/region_analysis_20261005_v2/region_relationship.md)：三张中文图、来源匹配记录、区域统计和运行快照；[中期汇报说明](reports/MIDTERM_REGION_UPDATE_20261005.md)。
- [清理与归档说明](reports/REPOSITORY_CLEANUP_20261005.md)：当前保留范围、旧产物恢复方法和验收记录。
- `history/`：保留传统 ML 原交付 ZIP；旧版重复运行已移出当前工作树，Git 历史可追溯。

团队修改共同数据 / split / transform 前须发布新版本；新增方法提交完整 val 预测、失败状态、
train-only 模型及 run_manifest，再注册到统一配置。每次新增方法或数据版本都重新生成评价集合。
提交前执行适用测试和 `python -m pr02 verify`；该入口同时复算 EDA、误差分析和区域分析的数值，核对配置与实际模型身份；用自己的开发分支提交集成变更，按组内发布授权完成验收后合入 main。`verify` 为只读验收，不重写已发布的验证证据；发布索引同时绑定图表运行清单。
