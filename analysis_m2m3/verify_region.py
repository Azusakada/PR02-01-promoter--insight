"""Independent identity, coordinate, and numeric checks for a saved region run."""
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ROOT, require


def verify_regions(output, data, meta):
    parent_path = next(item['path'] for item in meta['inputs'] if item['path'].endswith('Dataset.csv'))
    parent = pd.read_csv(ROOT / parent_path)
    indices = {0: defaultdict(set), 1: defaultdict(set)}
    for row in parent.itertuples(index=False):
        for offset in range(len(row.seq)-49):
            piece = row.seq[offset:offset+50]
            indices[row.label][piece].add((offset, 'forward', len(row.seq), str(row.seq_id)))
            reverse = piece.translate(str.maketrans('ACGT','TGCA'))[::-1]
            indices[row.label][reverse].add((offset, 'reverse', len(row.seq), str(row.seq_id)))
    audit = pd.read_csv(output / 'match_audit.tsv', sep='\t').set_index('sample_id')
    require(audit.index.is_unique and set(audit.index)==set(data.sample_id), 'Region audit IDs mismatch')
    for row in data.itertuples(index=False):
        positive, negative = indices[1].get(row.sequence, set()), indices[0].get(row.sequence, set())
        positions = {item[:3] for item in positive}
        status = ('both' if positive and negative else 'position_consistent_positive' if len(positions)==1
                  else 'ambiguous_positive' if len(positions)>1 else 'negative_only' if negative else 'no_hit')
        saved = audit.loc[row.sample_id]
        require(saved.match_status==status, 'Region matching classification differs from source: '+row.sample_id)
        require(json.loads(saved.positive_parent_loci)==[list(item) for item in sorted(positive)], 'Positive parent identity lost: '+row.sample_id)
        require(json.loads(saved.negative_parent_loci)==[list(item) for item in sorted(negative)], 'Negative parent identity lost: '+row.sample_id)
        require(saved.n_positive_parents==len({item[3] for item in positive}), 'Wrong positive parent count')
        if status=='position_consistent_positive':
            start, strand, length = next(iter(positions))
            require(saved.offset_0index==start and saved.strand==strand and saved.parent_length==length, 'Wrong local match coordinates')
    annotations = pd.read_csv(output / 'annotations.tsv', sep='\t')
    layout = meta['parameters']['layout_81bp_0based_end_exclusive']
    require(len(annotations)==len(data)*len(layout), 'Annotation row count mismatch')
    require(not annotations.duplicated(['sample_id','region_type']).any(), 'Duplicate annotation')
    require(set(annotations.sample_id)==set(data.sample_id), 'Annotation ID mismatch')
    joined = annotations.merge(data[['sample_id','sequence','target_log10','strength']], on='sample_id', validate='many_to_one')
    gc, chunks = [], []
    for row in joined.itertuples(index=False):
        match = audit.loc[row.sample_id]
        expected = 'missing'
        if match.match_status=='position_consistent_positive':
            left, right = layout[row.region_type]
            offset = int(match.offset_0index)
            bounds = ((left-offset, right-offset) if match.strand=='forward'
                      else (offset+50-right, offset+50-left))
            expected = 'tool_inferred' if 0 <= bounds[0] < bounds[1] <= 50 else 'not_applicable'
            if expected=='tool_inferred':
                require((row.start_0index,row.end_0index_exclusive)==bounds, 'Annotation interval mismatch')
                require(row.strand==match.strand, 'Annotation direction mismatch')
                for parent_id in json.loads(match.positive_parent_ids):
                    require(parent_id in row.evidence_ref.split('seq_id=')[1].split(';')[0].split(','), 'Annotation evidence lost parent identity')
        require(row.annotation_status==expected, 'Wrong annotation evidence status')
        if expected=='tool_inferred':
            piece = row.sequence[int(row.start_0index):int(row.end_0index_exclusive)]
            if row.strand=='reverse': piece=piece.translate(str.maketrans('ACGT','TGCA'))[::-1]
            gc.append((piece.count('G')+piece.count('C'))/len(piece)); chunks.append(piece)
        else:
            require(pd.isna(row.start_0index) and pd.isna(row.end_0index_exclusive), 'Missing/outside coordinates must be empty')
            gc.append(np.nan); chunks.append(None)
    joined['gc']=gc; joined['chunk']=chunks
    coverage = json.loads((output / 'region_coverage.json').read_text(encoding='utf-8'))
    actual = audit[audit.match_status.eq('position_consistent_positive')]
    require(coverage['n_position_consistent_positive']==len(actual), 'Wrong mapped sample denominator')
    require(coverage['n_unique_positive_parent']==int(actual.n_positive_parents.eq(1).sum()), 'Wrong unique parent count')
    n_metrics=0
    for saved in coverage['regions']:
        part=joined[joined.region_type.eq(saved['region_type']) & joined.annotation_status.eq('tool_inferred')]
        require(saved['n_analyzed_with_coordinates']==len(part) and saved['n_reliable_external_with_coordinates']==0, 'Wrong evidence or region count')
        if 'spearman_gc_vs_log10' in saved:
            value=spearmanr(part.gc,part.target_log10).statistic
            require(np.isclose(value,saved['spearman_gc_vs_log10'],atol=1e-12,rtol=1e-12), 'Region GC metric mismatch')
            n_metrics+=1
    features=pd.read_csv(output/'source_tables/region_feature_samples.tsv',sep='\t')
    for saved in coverage['motif_relationships']:
        part=joined[joined.region_type.eq(saved['region_type']) & joined.annotation_status.eq('tool_inferred')].copy()
        scores=part.chunk.map(lambda piece: sum(a==b for a,b in zip(piece,saved['consensus'])))
        value=spearmanr(scores,part.target_log10).statistic
        require(np.isclose(value,saved['spearman_consensus_identity_vs_log10'],atol=1e-12,rtol=1e-12), 'Motif metric mismatch')
        table=features[features.region_type.eq(saved['region_type'])].set_index('sample_id')
        require(table.index.is_unique and set(table.index)==set(part.sample_id), 'Feature source IDs mismatch')
        indexed=part.set_index('sample_id').loc[table.index]
        require(table.chunk.equals(indexed.chunk), 'Feature source sequence mismatch')
        require(np.allclose(table.target_log10,indexed.target_log10), 'Feature source truth mismatch')
        n_metrics+=1
    return {'region_identity_and_coordinates':'pass', 'region_samples_checked':len(data),
            'annotation_rows_checked':len(annotations), 'region_metrics_independently_recomputed':n_metrics,
            'parent_identity_policy':'position-consistent; all source records retained'}
