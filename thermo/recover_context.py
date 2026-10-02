# -*- coding: utf-8 -*-
"""
recover_context.py  —  PR02-01 热力学基线 / 上下文恢复

用途
----
课程 E. coli 50 bp strength 主表没有 TSS/方向字段，且 50 bp 太短，regseq2
Promoter_Calculator 无法直接运行（其有效 TSS 需要至少 58 bp 上游 + 20 bp 下游）。
本脚本把每条 50 bp 序列在 E. coli K-12 MG1655 参考基因组上精确定位，得到可追溯
的基因组坐标，供 build_thermo_inputs.py 补取上下游上下文（input_origin=recovered_context）。

输出
----
thermo/input/context_map.tsv

  sample_id  match_status  strand  genome_start_0index  n_positions
  genome_positions  genome_accession  genome_source_url

契约要点
--------
- 只做「定位 + 记录坐标」，不推断 TSS/-10/-35，不伪造旧坐标。
- 定位失败/多处命中的样本显式记录状态，不静默丢弃。
- 参考基因组与其坐标来源可追溯（accession + URL + sha256）。

用法
----
python thermo/recover_context.py                 # 下载/复用缓存基因组并定位
python thermo/recover_context.py --limit 100     # 只跑前 100 条（调试）
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_V1 = os.path.join(REPO, "data", "01_Ecoli_strength", "data_v1.tsv")
OUT_DIR = os.path.join(HERE, "input")
CACHE_DIR = os.path.join(HERE, ".cache")

GENOME_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
    "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
)
GENOME_ACCESSION = "GCF_000005845.2_ASM584v2|NC_000913.3"
CONTEXT_FLANK = 50  # 两侧各留 50 bp

_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


# --------------------------------------------------------------------------
# 输入读取（主表可能是 UTF-8 或 UTF-16，带不带 BOM 都兼容）
# --------------------------------------------------------------------------
def read_tsv(path: str):
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    header = lines[0].split("\t")
    for ln in lines[1:]:
        yield dict(zip(header, ln.split("\t")))


def load_genome() -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    gz = os.path.join(CACHE_DIR, "ecoli_mg1655.fna.gz")
    if not os.path.exists(gz):
        print("[genome] downloading", GENOME_URL)
        urllib.request.urlretrieve(GENOME_URL, gz)
    with gzip.open(gz, "rt") as f:
        parts, cur = [], []
        for line in f:
            if line.startswith(">"):
                if cur:
                    parts.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
        if cur:
            parts.append("".join(cur))
    return "".join(parts).upper()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(read_tsv(DATA_V1))
    if args.limit:
        rows = rows[: args.limit]
    print("[input] %d samples" % len(rows))

    genome = load_genome()
    L = len(rows[0]["sequence"])
    print("[genome] length=%d, window=%d, flank=%d" % (len(genome), L, CONTEXT_FLANK))

    # 查询集：正向 + 反向互补，各自带 (sample_id, strand)
    q = {}
    for r in rows:
        s = r["sequence"].upper()
        q.setdefault(s, []).append((r["sample_id"], "+"))
        q.setdefault(revcomp(s), []).append((r["sample_id"], "-"))

    # 线性扫描基因组，记录命中的 50-mer 位置
    hits = {}
    for i in range(0, len(genome) - L + 1):
        sub = genome[i : i + L]
        if sub in q:
            hits.setdefault(sub, []).append(i)

    gz = os.path.join(CACHE_DIR, "ecoli_mg1655.fna.gz")
    out_path = os.path.join(OUT_DIR, "context_map.tsv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "sample_id\tmatch_status\tstrand\tgenome_start_0index\tn_positions\t"
            "genome_positions\tgenome_accession\tgenome_source_url\n"
        )
        n_found = n_amb = n_missing = 0
        for r in rows:
            sid = r["sample_id"]
            s = r["sequence"].upper()
            cands = []
            for st, seq in (("+", s), ("-", revcomp(s))):
                for p in hits.get(seq, []):
                    cands.append((st, p))
            if not cands:
                n_missing += 1
                f.write("%s\tnot_found\tunknown\t-1\t0\t\t%s\t%s\n"
                        % (sid, GENOME_ACCESSION, GENOME_URL))
                continue
            cands.sort(key=lambda x: x[1])
            strand, pos = cands[0]
            status = "found" if len(cands) == 1 else "ambiguous"
            if status == "found":
                n_found += 1
            else:
                n_amb += 1
            pos_str = ";".join("%s:%d" % (st, p) for st, p in cands)
            f.write("%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\n"
                    % (sid, status, strand, pos, len(cands), pos_str,
                       GENOME_ACCESSION, GENOME_URL))

    print("[out] %s" % out_path)
    print("[stat] found=%d ambiguous=%d not_found=%d" % (n_found, n_amb, n_missing))
    print("[stat] genome_gz_sha256=%s" % sha256_file(gz))


if __name__ == "__main__":
    main()
