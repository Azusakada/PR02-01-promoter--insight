# 发布前验证记录

检查日期：2026-10-01（America/Los_Angeles；运行记录保留 UTC）。代码提交：`1039b9b`。两次正式运行的 tracked 工作区均为 clean。

接口检查器直接来自组长 ZIP `PR02-01_团队协作接口Skills_v2.1_仅接口规范.zip`，SHA256 为 `164ff79505f3b90eaba63b62f8177aba7bca8e9c3a5ea67cd714b28ada6fa9a5`，仅解压到本地临时目录使用，不安装或修改共享接口文件。

| 检查 | 实际结果 | 退出码 |
| --- | --- | --- |
| `python -m unittest discover -s analysis_m2m3/tests -v` | 8/8 通过：ID 错位、遗漏/未知 ID、错误标签、train-only 归一化、不裁剪、原始工具分数尺度、校准与失败行、常量排序无定义 | 0 |
| `python -m compileall -q analysis_m2m3` | 编译检查通过 | 0 |
| `git diff --check origin/main HEAD -- [本模块新增代码、测试、README、reports、requirements 的逐项路径]` | 上述手写代码与文档无格式错误 | 0 |
| `verify_run.py runs/eda_m2_20261001_v2` | 11,884 个 ID、train-only extrema/派生标签、所有输入及产物 SHA256、逐图源表和图片 SHA256 通过 | 0 |
| `verify_run.py runs/error_m3_val_20261001_v2` | 两方法各 1,783 个请求 ID、1,782 个共同 ID；独立重算 26 条 raw/log10/tool_rank 指标通过 | 0 |
| `validate.py bundle --samples … --splits … --transform … --targets …` | 11,884 条 samples、固定 split、分析用 label_transform、targets 接口通过 | 0 |
| `validate.py table metrics …/metrics.csv` | 26 行公共 schema 通过 | 0 |
| `validate.py table predictions …/knn_physchem_full_requested_predictions.csv` | 完整 1,783 行公共 schema 通过 | 0 |
| `validate.py table predictions …/thermo_regseq2_requested_predictions.csv` | 完整 1,783 行（含 1 个失败）公共 schema 通过 | 0 |
| `validate.py run …/eda_m2_20261001_v2/run_manifest.json --root …` | 22 个已声明产物通过 | 0 |
| `validate.py run …/error_m3_val_20261001_v2/run_manifest.json --root …` | 21 个已声明产物通过，execution_status=partial | 0 |

`verify_run.py` 的完整调用见 README；`validate.py` 位于 ZIP 中 `.agents/skills/pr02-team/scripts/`。上表省略号表示路径参数的缩写，不是可直接复制的命令；被检文件路径可从 README/运行 manifest 找到。

9 张 PNG 已逐张视觉检查标签、尺度、样本量、图注、裁剪和重叠；新 v2 PNG 与已检查试运行图逐文件 SHA256 完全相同。视觉状态与源表对应保存在各 `figure_manifest.json`。旧 v1 因运行记录格式需修订而只保存在本地项目旁 `PR02-01-analysis-local-archive/`，未纳入 GitHub 交付，没有覆盖已发布结果。

格式检查不包含 Matplotlib 自动生成的 SVG 及原样导入的队友数据快照：SVG 路径命令自带行末空格，整仓差异检查会报告这些行。没有为消除格式提示而改写源快照或已登记 hash 的图像文件。

这些检查不验证队友原始训练全过程，不证明模型具有生物学有效性或因果解释，不把自动导出的案例变成可靠区域注释。完整 M3、团队正式 transform、可靠区域注释的缺口见 handoff.md。
