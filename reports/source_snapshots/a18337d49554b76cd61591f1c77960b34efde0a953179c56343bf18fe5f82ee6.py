from __future__ import annotations

import importlib.util
import itertools
from functools import lru_cache

import numpy as np

from .data import ROOT, require, sequence_check


@lru_cache(maxsize=1)
def calculator_module():
    path=ROOT/'thermo/tools/regseq2/promoter_calculator.py'
    spec=importlib.util.spec_from_file_location('context87_promoter_calculator',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def encode(sequences,length):
    result=np.zeros((len(sequences),4,length),dtype=np.float32)
    for i,sequence in enumerate(sequences):
        sequence_check(sequence,length)
        result[i,["ACGT".index(base) for base in sequence],np.arange(length)]=1
    return result


def kmers(sequences,k,normalize=True):
    """Fixed ACGT vocabulary; optional frequency normalization supports 61/87 bp transfer."""
    require(type(k) is int and 1<=k<=6,'Unsupported k')
    vocab={''.join(b):i for i,b in enumerate(itertools.product('ACGT',repeat=k))}
    x=np.zeros((len(sequences),4**k),dtype=np.float64)
    for i,sequence in enumerate(sequences):
        sequence_check(sequence,len(sequence)); require(len(sequence)>=k,'Sequence shorter than k')
        for pos in range(len(sequence)-k+1): x[i,vocab[sequence[pos:pos+k]]]+=1
        if normalize: x[i]/=len(sequence)-k+1
    return x


PHYS_NAMES=['gc_fraction','at_fraction','gc_skew','at_skew','tm_gc_formula',
            'mean_dinucleotide_persistence','mean_dinucleotide_stacking','base_entropy_bits']


def physchem(sequences):
    tool=calculator_module(); output=[]
    for sequence in sequences:
        sequence_check(sequence,len(sequence)); require(len(sequence)>=14,'GC melting approximation requires >=14 bp')
        a,c,g,t=(sequence.count(base) for base in 'ACGT'); n=len(sequence)
        dimers=[sequence[i:i+2] for i in range(n-1)]; frequencies=np.array([a,c,g,t])/n
        output.append([(c+g)/n,(a+t)/n,(g-c)/(g+c) if g+c else 0,(a-t)/(a+t) if a+t else 0,
                       64.9+41*(g+c-16.4)/n,
                       np.mean([tool.persistence[d] for d in dimers]),
                       np.mean([tool.stacking_dict[d] for d in dimers]),
                       -sum(p*np.log2(p) for p in frequencies if p)])
    return np.asarray(output,dtype=np.float64)


def thermal_states(sequences,tss=67):
    calculator=calculator_module().Promoter_Calculator(); output=[]
    for sequence in sequences:
        sequence_check(sequence,87)
        require(tss==67,'This dataset has independently verified TSS index 67')
        minimum,states=calculator.predict(sequence,[tss,tss+1])
        require(tss in minimum and len(states[tss])==30,'Expected all 30 fixed-TSS promoter states')
        best=minimum[tss]
        output.append({k:float(v) if isinstance(v,(np.floating,np.integer)) else v for k,v in best.items()})
    return output
