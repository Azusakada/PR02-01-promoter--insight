# PR02-01 Promoter Insight

默认主数据已迁移为 **11,809 条经原论文 TSS 表核验的 E. coli 87 bp 序列**。
强度值来自原 50 bp 数据，未替换标签；87 bp 上下文从历史参考基因组 U00096.2 恢复。
教师给出的 **591 条 61 bp 枯草杆菌强度记录**作为跨物种目标数据。

原 50 bp 的代码、数据、模型和报告是历史版本，入口见 [旧版说明](README_ecoli50_legacy.md)。
当前配置是 [project_config.json](project_config.json)，当前结果在 `results/context87/`，
数据契约见 [CONTEXT87_V3.md](contracts/CONTEXT87_V3.md)，迁移记录见 [报告](reports/ECOLI87_MIGRATION.md)。

胡昊铭的新版 M2/M3 补充分析见 [交接与复现说明](reports/ECOLI87_M2M3_HANDOFF.md)：
10张中文图、三种尺度的EDA、固定/推定区域分析、五方法分组误差和80条规则化案例记录。
分析报告入口为 [新版报告](runs/ecoli87_m2m3_20261013_v1/analysis_report.md)，不沿用历史50/81bp结论。

## 安装与运行

建议 Python 3.12。依赖固定在 requirements.txt。
ZIP 解压后的目录也可直接使用；没有 Git 元数据时，运行记录仍保存代码文件 SHA256，code_commit 记为空。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pr02 validate-data
python -m pr02 verify
```

已交付运行目录用于复核，不覆盖重跑。新实验先创建配置，再运行：

```bash
python -m pr02 new-run --tag experiment01
python -m pr02 --config configs/project_ecoli87_experiment01.json run-all --research
```

`run-all` 完成 KNN、k-mer Ridge、LinearSVR、CNN、固定 TSS 的 Promoter Calculator、
同批验证样本比较、EDA、坐标及误差分析。加 `--research` 还会完成三种归因、High/Low/Random
等数量突变、四项重新训练的消融，以及 E. coli→B. subtilis 的 zero-shot/few-shot 对照。
也可分别调用 `run-knn`、`run-ridge-svr`、`run-cnn`、`run-thermo`、`evaluate`、`analyze`、
`explain`、`mutate`、`ablate`、`transfer`。各运行目录不可覆盖；归因和突变依赖当前 CNN checkpoint。

检查生成的 figures 后，登记视觉检查并发布本地结果：

```bash
python -m pr02 --config configs/project_ecoli87_experiment01.json acknowledge-figures
python -m pr02 --config configs/project_ecoli87_experiment01.json publish
```

`acknowledge-figures` 是操作者已实际查看图后的记录，不能代替看图。
`publish` 只更新本地结果索引；不会发布网站或合并 GitHub 分支。

## 数据和解释范围

主数据的原 50 bp 位于 `[17:67]`，TSS 在索引 67，整个序列对应 −67…−1、+1…+20。
验证时重做基因组双链精确定位，并逐条检查原标签、继承划分和原论文 51 nt（−50…+1）序列。
75 条多位置命中数据保留在排除文件中，避免任意选择上下文。训练/验证/测试数量是 8267/1768/1774。
归一化参数只由训练集拟合。

热力学模型在已验证 TSS 上枚举 30 种 spacer/discriminator 组合，选择最小自由能状态。
五个模型均使用同一段 87 bp；其热力学转录率另用训练集拟合 log10 线性校准。
TSS 有原论文证据，−10/−35/UP 等盒区仍是模型推定；固定窗口分析注明其假设。

枯草杆菌相同序列的重复测量按组划分，不能跨集合。k-mer 用窗口频率处理长度差异；
CNN 显式使用 61 bp 自适应池化实例及冻结源权重。few-shot 对照拟合目标域回归头。
教师数据缺少单位、实验条件、TSS 和方向信息，因此跨域 zero-shot 优先解释 Spearman，
不把跨域绝对误差作为共同强度单位下的结论。

归因、突变和消融属于计算实验；突变预测不是湿实验测量。此次 E. coli test 保持不评价。
沿用的是历史随机划分，邻近参考上下文可能重叠，且旧项目曾报告 test 指标，不能称为全新盲测。
多种子、严格基因组分组验证、DNA 语言模型和与 PR01-02 外部结果的比较仍可继续扩展。

## 复核

```bash
python -m pr02 rebuild-data --output work/rebuilt_ecoli87
python -m unittest discover -s tests -v
python -m unittest discover -s CNN/tests -v
python contracts/scripts/test_contracts.py
```

verify 会重载模型、复算预测和公共验证指标，检查 scaler、热力学校准、计算突变和目标域序列分组。
旧数据复现使用显式配置：

```bash
python -m pr02 --config configs/project_ecoli50_legacy.json validate-data
```
