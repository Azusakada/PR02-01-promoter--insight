"""Contract regression tests. Only synthetic fixtures; never run actual models."""
from pathlib import Path
import copy,csv,json,shutil,tempfile,unittest
import validate as v
import label_transform as lt

FILES={'dataset':'samples.tsv','splits':'split_manifest.tsv','targets':'targets.tsv','feature_index':'sample_ids.tsv','annotations':'annotations.tsv','calculator_inputs':'calculator_inputs.tsv','predictions':'predictions.csv','attributions':'ig_long.csv','attribution_qc':'ig_qc.csv','mutation_design':'mutation_design.csv','mutations':'mutations.csv','metrics':'metrics.csv'}
class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        shutil.copytree(v.BASE/'assets/examples',self.p,dirs_exist_ok=True)
    def tearDown(self):self.tmp.cleanup()
    def read(self,kind):return v.load_table(kind,self.p/FILES[kind])
    def write(self,kind,rows):
        spec=v.CONTRACTS['tables'][kind]
        with (self.p/FILES[kind]).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(spec['columns']),delimiter=spec['delimiter']);w.writeheader()
            for r in rows:w.writerow({k:('true' if val is True else 'false' if val is False else '' if val is None else val) for k,val in r.items()})
    def fail(self,kind,field,value,row=0):
        rs=self.read(kind);rs[row][field]=value;self.write(kind,rs)
        with self.assertRaises(v.ContractError):self.read(kind)
    def bundle(self,**kw):return v.bundle_check(self.p/'samples.tsv',self.p/'split_manifest.tsv',**{k:self.p/val for k,val in kw.items()})
    def test_all_twelve_tables(self):
        for kind in FILES:
            with self.subTest(kind=kind):self.assertTrue(self.read(kind))
    def test_complete_bundle(self):
        res=self.bundle(transform='label_transform.json',targets='targets.tsv',feature_index='sample_ids.tsv',annotations='annotations.tsv',calculator_inputs='calculator_inputs.tsv',predictions='predictions.csv',attributions='ig_long.csv',attribution_qc='ig_qc.csv',mutation_design='mutation_design.csv',mutations='mutations.csv',requested_ids='requested_ids.tsv')
        self.assertEqual(res['samples'],4)
    def test_old150_sequence_rejected(self):self.fail('dataset','sequence','A'*150)
    def test_old_schema_rejected(self):self.fail('dataset','schema_version','1.1.0')
    def test_nonpositive_label(self):self.fail('dataset','strength',0)
    def test_nan_label(self):self.fail('dataset','strength',float('nan'))
    def test_wrong_log(self):self.fail('dataset','target_log10',99)
    def test_duplicate_source_row(self):self.fail('dataset','source_row',2)
    def test_duplicate_sequence(self):self.fail('dataset','sequence','C'*50)
    def test_repeated_labels_not_duplicates(self):
        rs=self.read('dataset');rs[1]['strength']=rs[0]['strength'];rs[1]['target_log10']=rs[0]['target_log10'];self.write('dataset',rs);self.assertEqual(len(self.read('dataset')),4)
    def test_old_background_strategy(self):self.fail('splits','split_strategy','background_holdout')
    def test_group_strategy_missing_ids(self):
        rs=self.read('splits')
        for r in rs:r['split_strategy']='sequence_group'
        self.write('splits',rs)
        with self.assertRaises(v.ContractError):self.read('splits')
    def test_group_overlap(self):
        rs=self.read('splits')
        for r in rs:r.update(split_strategy='sequence_group',group_id='same')
        self.write('splits',rs)
        with self.assertRaises(v.ContractError):self.read('splits')
    def test_group_valid(self):
        rs=self.read('splits')
        for r in rs:r.update(split_strategy='sequence_group',group_id=r['split'])
        self.write('splits',rs);self.bundle()
    def test_split_coverage(self):
        rs=self.read('splits');rs.pop(0);self.write('splits',rs)
        with self.assertRaises(v.ContractError):self.bundle()
    def test_train_only_scale(self):
        x=lt.compute_label_transform(self.p/'samples.tsv',self.p/'split_manifest.tsv','demo_random42')
        self.assertEqual((x['min_log10'],x['max_log10']),(2.,4.))
        self.assertLess(lt.normalise_strength(10,x),0);self.assertGreater(lt.normalise_strength(1e6,x),1)
    def test_full_data_scaler_rejected(self):
        p=self.p/'label_transform.json';d=json.loads(p.read_text());d['max_log10']=6;p.write_text(json.dumps(d))
        with self.assertRaises(v.ContractError):self.bundle(transform='label_transform.json')
    def test_foreign_fit_ids_rejected(self):
        p=self.p/'label_transform.json';d=json.loads(p.read_text());d['train_ids'].append('demo_ecoli50_r000004');p.write_text(json.dumps(d))
        with self.assertRaises(v.ContractError):self.bundle(transform='label_transform.json')
    def test_clipped_targets_rejected(self):
        rs=self.read('targets');rs[-1]['target_normalized']=1;self.write('targets',rs)
        with self.assertRaises(v.ContractError):self.bundle(transform='label_transform.json',targets='targets.tsv')
    def test_transform_write_no_overwrite(self):
        p=self.p/'new_transform.json'
        lt.fit_label_transform(self.p/'samples.tsv',self.p/'split_manifest.tsv',split_id='demo_random42',output_path=p)
        before=p.read_bytes()
        with self.assertRaises(FileExistsError):lt.fit_label_transform(self.p/'samples.tsv',self.p/'split_manifest.tsv',split_id='demo_random42',output_path=p)
        self.assertEqual(p.read_bytes(),before)
    def test_scale_roundtrip(self):
        d=json.loads((self.p/'label_transform.json').read_text());z,raw=lt.restore_prediction(1.5,d)
        self.assertEqual((z,raw),(5,1e5))
    def test_prediction_requires_transform(self):self.fail('predictions','label_transform_id',None)
    def test_prediction_wrong_inverse(self):
        rs=self.read('predictions');rs[0]['predicted_value_normalized']=.5;self.write('predictions',rs)
        with self.assertRaises(v.ContractError):self.bundle(transform='label_transform.json',predictions='predictions.csv')
    def test_failed_prediction_zero(self):
        rs=self.read('predictions');rs[0].update(prediction_status='failed',error_reason='oops',predicted_value_normalized=0,predicted_value_log10=None,predicted_value=None);self.write('predictions',rs)
        with self.assertRaises(v.ContractError):self.read('predictions')
    def test_true_label_misalignment(self):
        rs=self.read('predictions');rs[0]['true_value']=3;self.write('predictions',rs)
        with self.assertRaises(v.ContractError):self.bundle(predictions='predictions.csv',transform='label_transform.json')
    def test_unknown_annotation_not_absent(self):
        self.assertEqual(self.read('annotations')[0]['annotation_status'],'missing')
        self.assertIsNone(self.read('annotations')[0]['start_0index'])
    def test_unknown_with_fabricated_bounds(self):self.fail('annotations','start_0index',10)
    def test_external_annotation_requires_source(self):
        rs=self.read('annotations');rs[0].update(annotation_status='reliable_external',start_0index=10,end_0index_exclusive=11,strand='forward');self.write('annotations',rs)
        with self.assertRaises(v.ContractError):self.read('annotations')
    def test_fixed_tss_121_rejected(self):
        rs=self.read('calculator_inputs');rs[0].update(tss_mode='fixed_tss',tss_position_1index=121,tss_evidence_ref='old');self.write('calculator_inputs',rs)
        with self.assertRaises(v.ContractError):self.read('calculator_inputs')
    def test_context_requires_source(self):
        rs=self.read('calculator_inputs');rs[0].update(input_origin='recovered_context',sequence='T'*60,sequence_length=60);self.write('calculator_inputs',rs)
        with self.assertRaises(v.ContractError):self.read('calculator_inputs')
    def test_ig_missing_position(self):
        rs=self.read('attributions');rs.pop();self.write('attributions',rs)
        with self.assertRaises(v.ContractError):self.read('attributions')
    def test_ig_old_coordinate(self):self.fail('attributions','coordinate_version','v1_motif_validated')
    def test_ig_wrong_rank(self):self.fail('attributions','rank_abs',50)
    def test_ig_wrong_base_cross_check(self):
        rs=self.read('attributions');rs[0]['nucleotide']='A';self.write('attributions',rs)
        with self.assertRaises(v.ContractError):self.bundle(transform='label_transform.json',attributions='ig_long.csv',attribution_qc='ig_qc.csv')
    def test_qc_mismatch(self):self.fail('attribution_qc','completeness_error',5)
    def test_new_k10_ok(self):self.assertEqual(v.mutation_config_check(v.read_json(v.BASE/'assets/config_templates/mutation.json'))['k'],10)
    def test_old_k20_k29_fail(self):
        d=v.read_json(v.BASE/'assets/config_templates/mutation.json')
        for k in [20,29]:
            with self.subTest(k=k),self.assertRaises(v.ContractError):v.mutation_config_check({**d,'k':k})
    def test_k_bounds_table(self):self.fail('mutation_design','k',20)
    def test_design_overlap(self):
        rs=self.read('mutation_design');r=rs[-1];r.update(position_1index=1,sequence='A'+'T'*49);self.write('mutation_design',rs)
        with self.assertRaises(v.ContractError):self.read('mutation_design')
    def test_multibase_mutation(self):
        rs=self.read('mutation_design');r=rs[0];r['sequence']='AA'+r['sequence'][2:];self.write('mutation_design',rs)
        with self.assertRaises(v.ContractError):self.bundle(mutation_design='mutation_design.csv')
    def test_direction_without_annotation(self):
        rs=self.read('mutation_design');r=rs[0];r.update(replacement_rule='toward_consensus',has_direction_expectation=True,expected_direction='up',direction_basis='guess');self.write('mutation_design',rs)
        with self.assertRaises(v.ContractError):self.read('mutation_design')
    def test_mutant_no_truth(self):
        rs=self.read('predictions');rs[0]['subset']='mutation';self.write('predictions',rs)
        with self.assertRaises(v.ContractError):self.read('predictions')
    def test_negative_r2_legal(self):self.assertLess(self.read('metrics')[0]['value'],0)
    def test_tool_r2_rejected(self):self.fail('metrics','target_scale','tool_rank')
    def test_synthetic_run_manifest(self):self.assertEqual(v.run_check(self.p/'run_manifest.json',self.p)['artifacts_checked'],1)
    def test_run_hash_mismatch(self):
        with (self.p/'predictions.csv').open('a') as f:f.write('\n')
        with self.assertRaises(v.ContractError):v.run_check(self.p/'run_manifest.json',self.p)
    def test_run_path_escape(self):
        d=v.read_json(self.p/'run_manifest.json');d['artifacts'][0]['path']='../outside';(self.p/'run_manifest.json').write_text(json.dumps(d))
        with self.assertRaises(v.ContractError):v.run_check(self.p/'run_manifest.json',self.p)
    def test_test_fit_forbidden(self):
        d=v.read_json(self.p/'run_manifest.json');d['fit_subsets']=['test'];(self.p/'run_manifest.json').write_text(json.dumps(d))
        with self.assertRaises(v.ContractError):v.run_check(self.p/'run_manifest.json',self.p)

if __name__=='__main__':unittest.main(verbosity=2)
