# 仓库清理与发布记录

日期：2026-10-05。集成开发分支 csy，统一发布分支 main。

## 清理范围

移出旧产物 290 个文件，约 96.04 MiB；另移出临时输入与 Python 缓存 93 个文件，约 24.53 MiB。工作目录合计减少约 120.58 MiB。测试可能重新生成少量忽略的缓存。

重复版本移出工作树；完整 ZIP 和原文件归档在 D 盘。工具自动审批拦截了直接删除命令，本次采用可恢复的移动。未重写 Git 历史，因此旧提交仍可恢复；本次清理不缩小 Git 历史包。

清理路径：

- `runs/comparison_val_five_methods_20261003_v1`
- `runs/comparison_val_five_methods_20261003_v2`
- `runs/comparison_val_integrated_20261002_v2`
- `runs/error_m3_five_methods_20261003_v1`
- `runs/error_m3_five_methods_20261003_v2`
- `runs/error_m3_integrated_20261002_v2`
- `runs/error_m3_integrated_20261002_v3`
- `analysis_m2m3/runs`
- `thermo/results`
- `CNN/runs/cnn_ecoli50_integrated_20261002_v2`
- `CNN/runs/cnn_ecoli50_preliminary_20261002_v1`
- `CNN/runs/cnn_smoke_20261002_v1`
- `CNN/verification`
- `history/current_results_comparison_val_five_methods_20261003_v1`
- `history/current_results_comparison_val_five_methods_20261003_v2`
- `history/current_results_comparison_val_integrated_20261002_v2`
- `history/data_bytes_before_lf_20261002_v1`
- `history/main_standard_exports_20261002_v1`
- `history/source_before_integration_20261002_v1`
- `history/project_config_before_midterm_20261005.json`

## 保留范围

- 当前五方法模型和比较运行，`results/current.json` 指向的全部输入与结果。
- 原始课程 NPY、统一主表/split/transform、数据重建脚本与实测重建记录。
- CNN 一轮完整 10 次训练、配置、选择记录、种子复跑、源码快照和模型重载证据。
- 当前 EDA、五方法误差/案例和区域分析及其图表源数据。
- KNN 搜索记录、Ridge 特征与选择记录、原交付 ZIP，以及分析测试依赖的原热力学快照。
- 六物种分类数据和已有 smoke 交付。

CNN README 的旧运行命令已更新，handoff/results_report 改为当前模型；旧文档已备份。绘图脚本现在要求显式给出运行目录。旧独立分析和热力学输出目录加入 gitignore，验收通过的运行仍须显式登记。

## 恢复方式

已提交的旧文件在清理前提交 `895bac210cc65165727a65ac0da4269edafc0e9b` 中。例如在仓库根目录导出，不改变当前分支：

```powershell
git archive --format=zip --output D:/PR02-old-comparison.zip 895bac210cc65165727a65ac0da4269edafc0e9b runs/comparison_val_five_methods_20261003_v2
```

本地全部旧产物（包括未提交旧实验）备份：`D:\CodexAnalysis\PR02-01\20261005-cleanup\obsolete_artifacts_before_cleanup.zip`。ZIP SHA256：`9eb2115d6c1f5f77e0e318dc0970bd5070e86317ea03ba7f8bdc9758972598b2`。备份中的每个原文件均按 SHA256 回读校验；逐文件清单位于同目录 `cleanup_plan.json`。移动的原文件在同目录 `moved_files/`，临时输入/缓存在 `moved_caches/`。恢复时先解压到仓库外，需要对照时再使用独立目录。

## 验收与发布

验收与发布状态记录在 [清理验收 JSON](repository_cleanup_validation_20261005.json)。既有审查 JSON 记录的是审查当时的状态；本次后续发布以此清理记录和 Git 远端为准。

发布包含此前已授权的 CNN 优化、区域分析与仓库接口修复，以及本次清理。验收完成后同步 csy/main；当前模型与预测的冻结文件保持原字节。

本轮 108 项测试/检查通过，根全项目 verify 通过，冻结配置、当前索引和既有验收文件字节保持一致。清理另外发现两个测试依赖预先存在的 work 目录，已修正为自动创建；独立检出会再次验证这一点。

独立 Git 检出验证源提交 `55a1cace6daaf0c9bb5e9697929ffb50d4947819`：初始无 work 目录，19 项根测试与全项目 verify 均通过；按当前 README 重新导出输入并加载 CNN，1,783 条预测 CSV 与交付文件逐字节一致，冻结证据保持原字节。见 [独立检出验收](cleanup_clean_checkout_verification_20261005.json)。

源提交已经实际推送，并从远端读回确认 main/csy 均为上述提交；随后仅补充本验收文档和日志。具体提交关系可由 Git 历史核对。
