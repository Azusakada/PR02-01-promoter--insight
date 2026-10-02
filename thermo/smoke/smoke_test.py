# -*- coding: utf-8 -*-
"""
smoke_test.py  —  50 bp 适用性 smoke test（热力学基线）

验证 regseq2 Promoter_Calculator 的最小输入长度：同一条真实样本在 50 / 100 / 150 bp
下的有效 TSS 数。用于支撑 thermo_applicability.md 的「50 bp 不可直接运行」结论。

运行：python thermo/smoke/smoke_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "thermo", "tools"))
from regseq2.promoter_calculator import Promoter_Calculator  # noqa: E402

# data_v1 第 1 条样本（ecoli50_r000001）的原始 50 bp
S50 = "ATAGCAGCTTCTGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATT"

calc = Promoter_Calculator()
print("organism:", calc.organism, "K=%s" % calc.K, "BETA=%s" % calc.BETA)
print()

cases = [("original_50bp", S50),
         ("padded_100bp", "A" * 25 + S50 + "T" * 25),
         ("padded_150bp", "A" * 50 + S50 + "T" * 50)]

for name, seq in cases:
    calc.run(seq, [0, len(seq)])
    out = calc.output()
    fwd = out["Forward_Predictions_per_TSS"]
    rev = out["Reverse_Predictions_per_TSS"]
    line = "%-16s len=%3d  valid_TSS(fwd=%d rev=%d)" % (name, len(seq), len(fwd), len(rev))
    if fwd:
        best = min(fwd.values(), key=lambda x: x["dG_total"])
        line += "  best_TSS=%d Tx_rate=%.4g" % (best["TSS"], best["Tx_rate"])
    print(line)
