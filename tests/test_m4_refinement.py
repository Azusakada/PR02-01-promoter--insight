import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pr02.context87.data import ROOT, read_json, sha, write_json
from pr02.context87.m4_data_eval import bootstrap, completed, config_digest, validate_protocol
from pr02.context87.m4_grouped_validation import genomic_interval, overlaps, partition
from pr02.context87.m4_ml_dnalm import fit_head


class RefinementContracts(unittest.TestCase):
    def test_resume_rejects_stale_upstream_and_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'output';output.mkdir();upstream=root/'input.txt';upstream.write_text('before')
            artifact=output/'result.txt';artifact.write_text('stable')
            cfg={'seed':1};protocol={'output':'output'}
            meta=dict(artifacts={'result.txt':sha(artifact)},upstream={'input.txt':sha(upstream)},
                      input_hashes={},code_hashes={},config_sha256=config_digest(cfg),protocol_sha256=config_digest(protocol))
            write_json(output/'run_manifest.json',meta)
            with patch('pr02.context87.m4_data_eval.ROOT',root):
                self.assertTrue(completed(output,cfg,protocol))
                with self.assertRaisesRegex(ValueError,'config'):completed(output,{'seed':2},protocol)
                with self.assertRaisesRegex(ValueError,'Protocol'):completed(output,cfg,{'output':'different'})
                upstream.write_text('after')
                with self.assertRaisesRegex(ValueError,'upstream'):completed(output,cfg,protocol)
                upstream.unlink()
                with self.assertRaisesRegex(ValueError,'upstream'):completed(output,cfg,protocol)

    def test_archived_code_allows_verification_but_never_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'output';out.mkdir();source=root/'model.py';source.write_text('old')
            digest=sha(source);archive=root/'reports/source_snapshots';archive.mkdir(parents=True);(archive/(digest+'.py')).write_text('old')
            write_json(out/'run_manifest.json',dict(artifacts={},upstream={},input_hashes={},code_hashes={'model.py':digest}))
            source.write_text('new')
            with patch('pr02.context87.m4_data_eval.ROOT',root):
                with self.assertRaisesRegex(ValueError,'executed code'):completed(out)
                self.assertTrue(completed(out,archived_code=True))

    def test_bootstrap_uses_declared_seed_repeats_and_confidence(self):
        x=np.random.default_rng(15).normal(size=31)
        protocol=dict(bootstrap_seed=41,bootstrap_repeats=137,confidence=.8)
        got=bootstrap(x,protocol=protocol)
        rng=np.random.default_rng(41)
        expected=np.quantile([x[rng.integers(0,len(x),len(x))].mean() for _ in range(137)],[.1,.9])
        np.testing.assert_allclose(got,expected)
        other=bootstrap(x,protocol=dict(protocol,bootstrap_seed=42))
        self.assertNotEqual(got,other)

    def test_protocol_rejects_silent_invalid_configurations(self):
        protocol=read_json(ROOT/'configs/m4_refinement_v2.json');validate_protocol(protocol)
        for field,value in [('seeds',[1,1]),('cnn_configurations',['unknown']),('confidence',1.1),('ridge_alpha',[0])]:
            with self.subTest(field=field):
                altered=copy.deepcopy(protocol);altered[field]=value
                with self.assertRaises(ValueError):validate_protocol(altered)

    def test_genomic_intervals_and_opposite_strand_overlap(self):
        self.assertEqual(genomic_interval(dict(tss_1based=100,strand='+')),(32,119))
        self.assertEqual(genomic_interval(dict(tss_1based=100,strand='-')),(80,167))
        self.assertEqual(overlaps([(32,119)],[(80,167)]),1)
        self.assertEqual(overlaps([(10,97)],[(97,184)]),0)
        self.assertEqual(overlaps([(10,100),(20,30)],[(90,110)]),1)

    def test_group_partition_prevents_overlap_duplicate_and_test_leakage(self):
        rng=np.random.default_rng(17);rows=[]
        for i in range(120):
            sequence=''.join(rng.choice(list('ACGT'),87));p=200+i*700
            rows.append(dict(sample_id=str(i),tss_1based=p,strand='+' if i%2 else '-',split='val' if i%4 else 'train',sequence=sequence))
        rows[10]['sequence']=rows[0]['sequence'];rows[7]['split']='test'
        rows.append(dict(sample_id='boundary',tss_1based=1020,strand='+',split='train',sequence='A'*87))
        protocol=dict(genomic_block_bp=1000,genome_length=100000,group_split_seed=17,group_evaluation_fraction=.2,group_dev_fraction=.2)
        result=partition(rows,protocol);assignment={r['sample_id']:r['split'] for r in result}
        self.assertEqual(assignment['7'],'excluded_historical_test');self.assertEqual(assignment['boundary'],'excluded_block_boundary')
        self.assertEqual(assignment['0'],assignment['10'])
        intervals={s:[genomic_interval(r) for r in rows if assignment[r['sample_id']]==s] for s in ('train','dev','evaluation')}
        self.assertTrue(all(intervals.values()))
        self.assertTrue(all(overlaps(intervals[a],intervals[b])==0 for a,b in [('train','dev'),('train','evaluation'),('dev','evaluation')]))

    def test_head_selection_cannot_access_evaluation_labels(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(60,8));y=rng.normal(size=60)
        train=np.arange(60)<30;dev=(np.arange(60)>=30)&(np.arange(60)<45)
        first,search=fit_head(x,y,train,dev,[1,10,100,1000])
        changed=y.copy();changed[45:]=1e10
        second,other=fit_head(x,changed,train,dev,[1,10,100,1000])
        self.assertEqual(search,other);np.testing.assert_array_equal(first.predict(x),second.predict(x))

    def test_custom_cnn_seeds_and_candidate_subset_change_real_training(self):
        from pr02.context87.m4_cnn import run
        cfg=read_json(ROOT/'project_config.json');cfg['num_threads']=1
        cfg['cnn']['model'].update(conv_channels=[2,4],hidden_dim=4)
        cfg['cnn']['training'].update(max_epochs=2,patience=1,batch_size=4)
        protocol=read_json(ROOT/'configs/m4_protocol_v1.json')
        protocol.update(output='runs/custom',seeds=[41,42],cnn_configurations=['C2'])
        rng=np.random.default_rng(7)
        rows=[dict(sample_id=str(i),sequence=''.join(rng.choice(list('ACGT'),87)),
                   target_log10=str(i/12),split='train' if i<8 else 'val') for i in range(12)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch('pr02.context87.m4_cnn.ROOT',root),patch('pr02.context87.m4_data_eval.ROOT',root),\
                 patch('pr02.context87.m4_cnn.load',return_value=(rows,dict(min_log10=0.,max_log10=1.))),\
                 patch('pr02.context87.m4_cnn.finish'),patch('pr02.context87.m4_cnn.bundle'),\
                 patch('pr02.context87.m4_cnn.export_predictions'):
                output=run(cfg,protocol)
            from pr02.context87.data import read_table
            search=read_table(output/'search.csv')
            self.assertEqual({(r['condition'],int(r['seed'])) for r in search},{('C2',41),('C2',42)})
            selected=read_json(output/'selected_model.json')
            self.assertEqual(selected['representative_seed'],41)
            for seed in [41,42]:
                actual=read_json(output/f'C2_s{seed}/config.json')
                self.assertEqual(actual['seed'],seed);self.assertEqual(actual['cnn']['model']['pooled_length'],20)
                self.assertEqual(actual['cnn']['model']['dropout'],.1)

    def test_visual_acknowledgement_requires_exact_figure_list(self):
        from pr02.context87.m4_reporting import acknowledge
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);figures=root/'figures';figures.mkdir();image=figures/'a.png';image.write_bytes(b'unit-test image identity')
            cfg={'seed':1};protocol={'output':'runs/test'}
            write_json(figures/'figure_manifest.json',dict(figures=[dict(figure_id='a',png='a.png',sha256=sha(image),visual_review='pending')]))
            write_json(figures/'run_manifest.json',dict(artifacts={'figure_manifest.json':sha(figures/'figure_manifest.json'),'a.png':sha(image)},
                upstream={},input_hashes={},code_hashes={},config_sha256=config_digest(cfg),protocol_sha256=config_digest(protocol)))
            with patch('pr02.context87.m4_reporting.run_root',return_value=root),patch('pr02.context87.m4_reporting.finish'):
                with self.assertRaisesRegex(ValueError,'every figure'):acknowledge(cfg,protocol,[])
                acknowledge(cfg,protocol,['a'])
            self.assertEqual(read_json(figures/'figure_manifest.json')['figures'][0]['visual_review'],'reviewed')

    def test_report_build_is_run_local_even_when_results_change(self):
        from pr02.context87.m4_final_report import build
        import shutil
        old=ROOT/'runs/m4_complete_20261014_v1'
        needed=['evaluation/model_summary.csv','ablation/stage_summary.csv','explanation_analysis/overlap_summary.csv',
                'explanation_analysis/consistency_summary.csv','mutation/mutation_summary.csv','mutation/direction_summary.csv',
                'transfer/transfer_metrics.csv','cnn/selected_model.json']
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'runs/new';root.mkdir(parents=True)
            for name in needed:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/name,p)
            published=Path(folder)/'reports';published.mkdir();sentinel=published/'M5_FINAL_REPORT.md';sentinel.write_text('published version')
            protocol=read_json(ROOT/'configs/m4_protocol_v1.json');protocol['output']='runs/new'
            with patch('pr02.context87.m4_final_report.run_root',return_value=root):build({},protocol)
            self.assertEqual(sentinel.read_text(),'published version')
            self.assertIn('runs/new',(root/'reports/M5_FINAL_REPORT.md').read_text())


if __name__=='__main__':unittest.main()
