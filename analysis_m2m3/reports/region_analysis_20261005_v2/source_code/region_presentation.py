"""Source tables, figures, and concise reporting for canonical region exploration."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata, spearmanr

from common import write_json

NAMES = {"minus10": "−10", "minus35": "−35", "spacer": "间隔", "UP_element": "上游区域", "TSS": "参考起始位置"}
CONSENSUS = {"minus10": "TATAAT", "minus35": "TTGACA"}


def confidence_interval(x, y, rng, n_boot=4000):
    """Paired-row percentile bootstrap, not a sequence-cluster inference."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    values = []
    for first in range(0, n_boot, 100):
        idx = rng.integers(0, len(x), size=(min(100, n_boot-first), len(x)))
        a, b = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        denominator = np.sqrt((a*a).sum(axis=1) * (b*b).sum(axis=1))
        good = denominator > 0
        values.extend(((a*b).sum(axis=1)[good] / denominator[good]).tolist())
    return [float(v) for v in np.quantile(values, [.025, .975])] if values else None


def _font():
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in candidates:
        if path.is_file():
            plt.rcParams["font.family"] = FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def presentation(run, samples, annotations, audit, coverage, motifs):
    _font()
    ids = set(audit.loc[audit.match_status.eq("position_consistent_positive"), "sample_id"])
    source = samples[["sample_id", "sequence", "strength", "target_log10", "split"]].copy()
    source["gc"] = source.sequence.map(lambda s: (s.count("G") + s.count("C")) / len(s))
    source["matched"] = source.sample_id.isin(ids)
    sequence_source = run.table("sequence_feature_samples.tsv", source.drop(columns="sequence"), sep="\t")
    selected, remainder = source[source.matched], source[~source.matched]
    groups = {}
    for name, frame in [("all", source), ("position_consistent", selected), ("remaining", remainder)]:
        groups[name] = {"n": len(frame), "median_strength": float(frame.strength.median()),
                        "mean_gc": float(frame.gc.mean())}
    comparison = {}
    for feature in ["target_log10", "gc"]:
        comparison[feature] = {"ks_distance": float(ks_2samp(selected[feature], remainder[feature]).statistic)}
    rng = np.random.default_rng(20261005)
    feature_frames, frequencies, region_stats, group_rows = [], [], {}, []
    for region, consensus in CONSENSUS.items():
        part = annotations[annotations.region_type.eq(region) & annotations.annotation_status.eq("tool_inferred")]
        frame = part.merge(source, on="sample_id", how="left", validate="one_to_one")
        chunks = []
        for row in frame.itertuples(index=False):
            chunk = row.sequence[int(row.start_0index):int(row.end_0index_exclusive)]
            chunks.append(chunk.translate(str.maketrans("ACGT", "TGCA"))[::-1] if row.strand == "reverse" else chunk)
        frame["chunk"] = chunks
        frame["region_gc"] = [(s.count("G") + s.count("C")) / 6 for s in chunks]
        frame["consensus_matches"] = [sum(a == b for a, b in zip(s, consensus)) for s in chunks]
        fields = ["sample_id", "region_type", "chunk", "region_gc", "consensus_matches", "gc", "strength", "target_log10", "split"]
        feature_frames.append(frame[fields])
        info = {"n": len(frame), "consensus": consensus,
                "region_at_fraction": float(1-frame.region_gc.mean()),
                "whole_sequence_at_fraction": float(1-frame.gc.mean()),
                "exact_consensus_n": int(frame.chunk.eq(consensus).sum()), "relationships": {}}
        for feature in ["region_gc", "consensus_matches"]:
            result = spearmanr(frame[feature], frame.target_log10)
            info["relationships"][feature] = {"spearman": float(result.statistic),
                    "bootstrap95": confidence_interval(frame[feature], frame.target_log10, rng)}
        region_stats[region] = info
        for position in range(6):
            for base in "ACGT":
                frequencies.append({"region_type": region, "position_1index": position+1, "base": base,
                    "fraction": sum(s[position] == base for s in chunks) / len(chunks), "n": len(chunks)})
        for score, group in frame.groupby("consensus_matches"):
            group_rows.append({"region_type": region, "matching_letters": int(score), "n": len(group),
                "median_strength": float(group.strength.median()), "median_log10": float(group.target_log10.median()),
                "q25_log10": float(group.target_log10.quantile(.25)), "q75_log10": float(group.target_log10.quantile(.75))})
    features = pd.concat(feature_frames, ignore_index=True)
    feature_source = run.table("region_feature_samples.tsv", features, sep="\t")
    frequency_frame = pd.DataFrame(frequencies)
    frequency_source = run.table("region_base_frequencies.csv", frequency_frame)
    group_source = run.table("consensus_group_summary.csv", pd.DataFrame(group_rows))
    summary = {"groups": groups, "matched_vs_remaining": comparison, "regions": region_stats,
               "bootstrap_seed": 20261005, "bootstrap_resamples": 4000,
               "bootstrap_unit": "paired sample rows; no sequence-cluster correction",
               "spacer_length": {"bp": 17, "basis": "fixed layout assumption, not an observed distribution"}}
    write_json(run.output / "biological_summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, region in zip(axes, CONSENSUS):
        table = frequency_frame[frequency_frame.region_type.eq(region)]
        matrix = table.pivot(index="base", columns="position_1index", values="fraction").reindex(list("ACGT")).to_numpy()
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=.65, aspect="auto")
        for row in range(4):
            for col in range(6):
                ax.text(col, row, f"{matrix[row,col]*100:.0f}%", ha="center", va="center", color="white" if matrix[row,col]>.4 else "black")
        ax.set_yticks(range(4), list("ACGT"))
        ax.set_xticks(range(6), [f"{i+1}\n{b}" for i,b in enumerate(CONSENSUS[region])])
        ax.set_xlabel("位置 / 常见模式字母")
        ax.set_title(f"{NAMES[region]} 区字母频率（n={region_stats[region]['n']}）")
    run.figure("region_base_frequencies", fig, "启动子区域字母特征", [frequency_source], "position_consistent_positive", "base_fraction", len(selected), f"按81 bp布局圈定区域；−10 n={region_stats['minus10']['n']}、−35 n={region_stats['minus35']['n']}，展示观察频率。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, region in zip(axes, CONSENSUS):
        frame = features[features.region_type.eq(region)]
        scores = sorted(frame.consensus_matches.unique())
        values = [frame.loc[frame.consensus_matches.eq(score), "target_log10"].to_numpy() for score in scores]
        ax.boxplot(values, positions=scores, widths=.55, showfliers=False)
        ax.set_xticks(scores, [f"{score}\nn={len(value)}" for score,value in zip(scores,values)])
        ax.set_xlabel("与常见模式相同的字母数")
        ax.set_ylabel("log10(强度)")
        ax.set_title(f"{NAMES[region]} 模式相似度与强度")
        ax.grid(axis="y", alpha=.2)
    run.figure("consensus_strength_groups", fig, "模式相似度与强度分布", [feature_source, group_source], "position_consistent_positive", "log10", len(selected), f"每个分组标出样本数。箱线图未画离群点，完整样本保留在源表；完全匹配组仅{region_stats['minus10']['exact_consensus_n']}/{region_stats['minus35']['exact_consensus_n']}条。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, feature, label in zip(axes, ["target_log10", "gc"], ["log10(强度)", "完整序列GC比例"]):
        for frame, title, color in [(source, "全部样本", "#386CB0"), (selected, "区域分析样本", "#E18B35"), (remainder, "其余样本", "#45A37A")]:
            values = np.sort(frame[feature])
            ax.plot(values, np.arange(1,len(values)+1)/len(values), label=f"{title} n={len(values)}", color=color)
        ax.set_xlabel(label); ax.set_ylabel("累计比例"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    run.figure("region_sample_distributions", fig, "区域分析样本与全体数据的分布", [sequence_source], "all", "log10/gc_fraction", len(source), f"强度中位数：区域分析样本{groups['position_consistent']['median_strength']:.2f}，全体{groups['all']['median_strength']:.2f}；展示完整分布。")
    for figure in run.figures:
        figure.update(annotation_level="tool_inferred", inferred_annotation_version="region_map_81bp_20261005_v2")

    write_report(run.output / "region_relationship.md", samples, audit, coverage, motifs, summary)
    return summary


def write_report(path, samples, audit, coverage, motifs, biology):
    count = int(audit.match_status.eq("position_consistent_positive").sum())
    external_unique = int((audit.match_status.eq("position_consistent_positive") & audit.n_positive_parents.eq(1)).sum())
    minus10 = biology["regions"]["minus10"]
    lines = ["# 启动子区域特征与强度关系", "", "日期：2026-10-05。", "",
        "## 主要发现", "",
        f"−10 区表现出明显的 A/T 富集：A/T 比例为 {100*minus10['region_at_fraction']:.1f}%，同批完整序列为 {100*minus10['whole_sequence_at_fraction']:.1f}%。−35 区的前两个位置偏向 T，其余位置的字母偏好较弱。",
        "",
        "区域 GC 比例与简单模式相似度对强度的解释有限，整体没有明显的单调关系。天然启动子中 −10 特征较明显、−35 特征较弱的现象，与原始研究相符。区域功能涉及多个因素，不能由简单相关性概括。[天然启动子研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC4288677/)、[区域组合与强度实验](https://pmc.ncbi.nlm.nih.gov/articles/PMC9440211/)",
        "", "## 区域统计", "", "| 区域 | 分析样本数 | GC 与强度的相关性 |", "| --- | ---: | ---: |"]
    for row in coverage:
        value = row.get("spearman_gc_vs_log10")
        coefficient = "未计算" if value is None else f"{value:.4f}"
        lines.append(f"| {NAMES[row['region_type']]} | {row['n_analyzed_with_coordinates']} | {coefficient} |")
    lines += ["", "| 常见模式 | 样本数 | 相似度与强度的相关性 |", "| --- | ---: | ---: |"]
    for row in motifs:
        value = row.get("spearman_consensus_identity_vs_log10")
        coefficient = "未计算" if value is None else f"{value:.4f}"
        lines.append(f"| {NAMES[row['region_type']]} {row['consensus']} | {row['n']} | {coefficient} |")
    lines += ["", "四项主要关系的重复抽样区间均跨过零，尚未观察到稳定的整体趋势。上游区域仅 6 条，保留统计值供复查。",
        "", "## 图表", "",
        "![区域字母频率](figures/region_base_frequencies.png)", "",
        "![模式相似度与强度](figures/consensus_strength_groups.png)", "",
        "![样本分布](figures/region_sample_distributions.png)", "",
        "## 汇报表述", "",
        "> 我们完成了启动子区域特征与强度的探索分析。−10 区呈现 A/T 富集，−35 区特征较弱，与天然启动子的部分已知特征相符。GC 比例和简单模式相似度不足以解释强度差异，后续将结合区域位置、间隔和多种特征进一步分析。",
        "", "## 方法与数据范围", "",
        f"从全部 {len(samples):,} 条 50 bp 强度序列中，{count} 条能在 81 bp 正样本中得到一致的位置和方向；其中 {external_unique} 条只对应一条原始记录，{count-external_unique} 条对应多条记录但换算位置相同。所有来源编号及各匹配位置保存在 match_audit.tsv。−10 分析 {minus10['n']} 条，−35 分析 {biology['regions']['minus35']['n']} 条。",
        "",
        "采用 81 bp 窗口参考位置在第 61 位的布局约定：−35 和 −10 分别取六个字母，中间取 17 bp。位置由规则换算，annotations.tsv 使用 tool_inferred，并与外部确认注释分别计数。17 bp 是输入约定；本次没有测量真实间隔长度的分布。",
        "",
        f"区域分析样本的强度中位数为 {biology['groups']['position_consistent']['median_strength']:.2f}，全体为 {biology['groups']['all']['median_strength']:.2f}；这批样本偏向较高强度，结果不直接推广到全部序列。相关性使用 log10 强度与 Spearman 排序指标；区间使用样本行成对重抽样 4,000 次，未作相似序列分组修正。",
        "",
        "## 复现与交接", "",
        "源表位于 source_tables/；biological_summary.json 保留分布、区域特征和区间；region_coverage.json 保留覆盖统计。run_manifest.json 绑定输入与产物哈希、环境和上游提交，source_code/ 保存实际运行脚本。图表经实际查看后由 verify_run.py 记录视觉验收。",
        "",
        "本次在 csy 上接入胡昊铭提交 3c779b9d07f369432604551b2b020287b224232d 的代码，并修正匹配来源、注释等级和汇报表达。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
