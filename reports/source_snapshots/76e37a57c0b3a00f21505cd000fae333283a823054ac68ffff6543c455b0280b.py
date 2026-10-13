from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .data import ROOT, coordinate, load, read_table, sha, write_json, write_table
from .experiment import METHODS, fresh, manifest, path

# Fixed canonical windows are hypotheses; experimental evidence fixes the TSS only.
REGIONS={'UP':(7,31),'-35':(32,38),'spacer':(38,55),'-10':(55,61),'discriminator':(61,67),'ITR':(67,87)}


def analyze(cfg):
    rows,_=load(cfg); directory=fresh(path(cfg,'analysis')); figures=directory/'figures'; figures.mkdir()
    data=np.array([float(r['target_log10']) for r in rows]); n=len(rows); figure_records=[]
    def save(name,fig,table):
        fig.tight_layout(); file=figures/f'{name}.png'; fig.savefig(file,dpi=160); plt.close(fig)
        figure_records.append(dict(figure=file.relative_to(directory).as_posix(),sha256=sha(file),
                                   source_table=table,visual_review='pending'))
    counts=[dict(subset=s,n=sum(r['split']==s for r in rows)) for s in ('train','val','test')]
    write_table(directory/'split_counts.csv',counts)
    write_table(directory/'label_distribution.csv',[dict(sample_id=r['sample_id'],subset=r['split'],log10_strength=r['target_log10']) for r in rows])
    fig,ax=plt.subplots(figsize=(7,4)); ax.hist(data,bins=50,color='#4878a8'); ax.set(xlabel='log10(original strength)',ylabel='Samples',title=f'E. coli verified context: n={n}')
    save('label_distribution',fig,'label_distribution.csv')
    frequencies=np.array([[sum(r['sequence'][i]==base for r in rows)/n for i in range(87)] for base in 'ACGT'])
    write_table(directory/'base_frequency.csv',[dict(position_0based=i,position_relative_tss=coordinate(i),**{base:frequencies[j,i] for j,base in enumerate('ACGT')}) for i in range(87)])
    fig,ax=plt.subplots(figsize=(12,3)); heat=ax.imshow(frequencies,aspect='auto',vmin=0,vmax=0.6,cmap='viridis')
    ticks=[0,17,32,55,66,67,76,86]; ax.set_xticks(ticks,[coordinate(i) for i in ticks]); ax.set_yticks(range(4),list('ACGT'))
    ax.axvline(66.5,color='white',lw=1.5); ax.set(xlabel='Position relative to verified TSS (+1)',title=f'Base frequencies, n={n}'); fig.colorbar(heat,ax=ax,label='Frequency')
    save('base_frequency',fig,'base_frequency.csv')
    comparisons=read_table(path(cfg,'comparison')/'metrics.csv')
    write_table(directory/'validation_metrics.csv',comparisons)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric,label in zip(axes,['r2_log10','mae_log10','spearman'],['R² (log10)','MAE (log10)','Spearman']):
        ax.bar([r['method'] for r in comparisons],[float(r[metric]) for r in comparisons],color='#4878a8')
        ax.set(title=label); ax.tick_params(axis='x',rotation=35); ax.axhline(0,color='gray',lw=.5)
    fig.suptitle('Common validation cohort; all models receive the same 87 bp context')
    save('model_comparison',fig,'validation_metrics.csv')
    residual_rows=[]; fig,axes=plt.subplots(1,5,figsize=(16,3.5))
    for ax,kind in zip(axes,METHODS):
        pred=read_table(path(cfg,kind)/'predictions_val.csv')
        y=np.array([float(r['true_value_log10']) for r in pred]); p=np.array([float(r['predicted_value_log10']) for r in pred])
        ax.scatter(y,p,s=4,alpha=.3); bounds=[min(y.min(),p.min()),max(y.max(),p.max())]; ax.plot(bounds,bounds,'k--',lw=.8)
        ax.set(title=f'{kind}, n={len(y)}',xlabel='Observed log10 strength',ylabel='Predicted log10 strength')
        for r,error in zip(pred,p-y): residual_rows.append(dict(method=kind,sample_id=r['sample_id'],residual_log10=error,absolute_error_log10=abs(error)))
    write_table(directory/'validation_residuals.csv',residual_rows); save('validation_scatter',fig,'validation_residuals.csv')
    region_stats=[]; window_rows=[]
    for name,(start,end) in REGIONS.items():
        gc=np.array([sum(base in 'GC' for base in r['sequence'][start:end])/(end-start) for r in rows])
        region_stats.append(dict(region=name,start_0based=start,end_0based_exclusive=end,n=n,mean_gc=float(gc.mean()),
                                 spearman_gc_vs_log10=float(spearmanr(gc,data).statistic),annotation_status='canonical_window_assumption'))
        window_rows.extend(dict(sample_id=r['sample_id'],region=name,gc_fraction=float(value),target_log10=r['target_log10']) for r,value in zip(rows,gc))
    write_table(directory/'canonical_region_summary.csv',region_stats); write_table(directory/'canonical_region_features.csv',window_rows)
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar([r['region'] for r in region_stats],[r['spearman_gc_vs_log10'] for r in region_stats],color='#59a14f')
    ax.axhline(0,color='gray',lw=.8); ax.set(ylabel='Spearman(GC fraction, log10 strength)',title='Canonical TSS-relative windows; exploratory, all samples')
    save('canonical_regions',fig,'canonical_region_summary.csv')
    with (path(cfg,'thermo')/'states.jsonl').open() as f: states=[json.loads(line) for line in f]
    annotations=[]
    for state in states:
        for region,key in [('UP','UP_position'),('-35','hex35_position'),('spacer','spacer_position'),('-10','hex10_position'),('discriminator','disc_position')]:
            start,end=state[key]; annotations.append(dict(sample_id=state['sample_id'],region=region,start_0based=start,end_0based_exclusive=end,
                sequence_length=87,tss_index_0based=67,annotation_status='model_inferred',source='fixed-TSS Promoter Calculator minimum-energy state'))
    write_table(directory/'model_inferred_regions.tsv',annotations)
    write_json(directory/'figure_manifest.json',dict(figures=figure_records,annotation_policy='Verified TSS; model inferred boxes; separately labeled canonical windows'))
    manifest(cfg,'analysis',directory,dict(n_all=n,n_val=sum(r['split']=='val' for r in rows),test_metrics_enabled=False))
    return directory


def acknowledge(cfg):
    """Explicit operator acknowledgment; call only after viewing the generated figures."""
    directory=path(cfg,'analysis'); report=json.loads((directory/'figure_manifest.json').read_text())
    for r in report['figures']: r['visual_review']='reviewed'
    write_json(directory/'figure_manifest.json',report)
    manifest(cfg,'analysis',directory,dict(visual_review='reviewed',test_metrics_enabled=False))
