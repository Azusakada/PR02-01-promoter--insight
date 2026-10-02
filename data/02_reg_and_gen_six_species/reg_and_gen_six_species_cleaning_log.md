# reg_and_gen 六物种数据处理记录

本记录对六个物种的 Dataset/train/dev/test/positive_samples 文件逐一读取，保留其原始 0/1 启动子识别任务。该数据不是 E. coli strength 回归标签，因此不并入 `data_v1.tsv`。

## 统一检查

- 所有文件均要求列为 `seq_id, seq_type, seq, label`。
- `label` 保持 0/1 分类语义；不转换为连续强度。
- Dataset 的 train/dev/test 按 `seq_id` 互斥且并集覆盖；positive_samples 与 Dataset 的 label=1 子集核对。
- 发现的重复序列和 80 bp 记录只登记，不静默删除。

## Bacillus subtilis

- Dataset：1,500 条；正类 691；负类 809。
- 序列长度：81 bp；重复序列额外记录：0。
- 划分：train=1,050，dev=150，test=300；ID 互斥且覆盖 Dataset。
- positive_samples：691 条；与 Dataset 的 label=1 序列一致。
- 结论：该目录数据作为独立二分类数据登记，不进入连续 strength 的 KNN 回归训练。

## Baumanii

- Dataset：2,900 条；正类 1,540；负类 1,360。
- 序列长度：81 bp；重复序列额外记录：0。
- 划分：train=2,030，dev=290，test=580；ID 互斥且覆盖 Dataset。
- positive_samples：1,540 条；与 Dataset 的 label=1 序列一致。
- 结论：该目录数据作为独立二分类数据登记，不进入连续 strength 的 KNN 回归训练。

## Bradyrhizobium

- Dataset：3,940 条；正类 1,987；负类 1,953。
- 序列长度：80,81 bp；重复序列额外记录：4。
- 划分：train=2,758，dev=394，test=788；ID 互斥且覆盖 Dataset。
- positive_samples：1,987 条；与 Dataset 的 label=1 序列一致。
- 结论：该目录数据作为独立二分类数据登记，不进入连续 strength 的 KNN 回归训练。

## Diphtheria

- Dataset：3,310 条；正类 1,654；负类 1,656。
- 序列长度：80,81 bp；重复序列额外记录：0。
- 划分：train=2,317，dev=331，test=662；ID 互斥且覆盖 Dataset。
- positive_samples：1,654 条；与 Dataset 的 label=1 序列一致。
- 结论：该目录数据作为独立二分类数据登记，不进入连续 strength 的 KNN 回归训练。

## Escherichia coli

- Dataset：3,350 条；正类 1,645；负类 1,705。
- 序列长度：80,81 bp；重复序列额外记录：8。
- 划分：train=2,345，dev=335，test=670；ID 互斥且覆盖 Dataset。
- positive_samples：1,645 条；与 Dataset 的 label=1 序列一致。
- 结论：该目录数据作为独立二分类数据登记，不进入连续 strength 的 KNN 回归训练。

## Staphylococcus

- Dataset：3,370 条；正类 2,207；负类 1,163。
- 序列长度：80,81 bp；重复序列额外记录：1。
- 划分：train=2,359，dev=337，test=674；ID 互斥且覆盖 Dataset。
- positive_samples：2,207 条；与 Dataset 的 label=1 序列一致。
- 结论：该目录数据作为独立二分类数据登记，不进入连续 strength 的 KNN 回归训练。

## 总计

六物种共 18,370 条，正类 9,724 条，负类 8,646 条；所有目录检查通过。

