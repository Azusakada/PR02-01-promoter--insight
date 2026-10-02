# 团队接口规范来源

`scripts/validate.py`、`scripts/label_transform.py`、`scripts/api_contract.py`、
`scripts/test_contracts.py`、`assets/tables.json` 及 self-test 所需样例复制自用户提供的
`PR02-01_团队协作接口Skills_v2.1_仅接口规范.zip`，保持字节不变。
公共 schema 是 `2.0.0`，API 是 `2.0`，规范包版本是 `2.1`。

这些文件提供全组表、标签变换和接口校验。统一管理位于 `pr02/`，CNN 实现位于 `CNN/pr02_cnn/`。
本目录不安装或修改仓库的 agent 配置。

从仓库根目录执行：

```bash
python contracts/scripts/validate.py self-test
python contracts/scripts/validate.py bundle --samples data/01_Ecoli_strength/data_v1.tsv --splits data/01_Ecoli_strength/split_manifest.tsv --transform data/01_Ecoli_strength/label_transform.json
```
