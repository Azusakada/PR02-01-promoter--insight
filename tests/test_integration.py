from __future__ import annotations
import unittest
from pr02.thermo import select_state, fit_calibration, apply_calibration
from pr02.metrics import metric_rows


def state(tss,energy):
    return dict(TSS=tss,dG_total=energy,Tx_rate=100.,UP_position=[0,24],hex35_position=[25,31],
                spacer_position=[31,46],hex10_position=[46,52],disc_position=[52,tss])


class IntegrationTests(unittest.TestCase):
    def test_same_coordinate_retains_forward_candidate(self):
        out = dict(Forward_Predictions_per_TSS={70:state(70,-4)},Reverse_Predictions_per_TSS={70:state(80,-1)})
        strand,tss,selected,boxes,count = select_state(out,150)
        self.assertEqual((strand,tss,count),('forward',70,2))
        self.assertEqual(selected['dG_total'],-4)

    def test_reverse_coordinates_use_mapped_key_and_half_open_boxes(self):
        out = dict(Forward_Predictions_per_TSS={},Reverse_Predictions_per_TSS={70:state(80,-4)})
        strand,tss,selected,boxes,count = select_state(out,150)
        self.assertEqual((strand,tss),('reverse',70))
        self.assertEqual(boxes['hex10_position'],[98,104])

    def test_calibration_excludes_validation_labels(self):
        rows = [dict(sample_id='tr1',prediction_status='ok',predicted_tx_rate=10,true_value=100),
                dict(sample_id='tr2',prediction_status='ok',predicted_tx_rate=100,true_value=1000),
                dict(sample_id='val',prediction_status='ok',predicted_tx_rate=10000,true_value=1e20)]
        a = fit_calibration(rows,{'tr1','tr2'},'r','s','d')
        rows[-1]['true_value'] = 1e-20
        b = fit_calibration(rows,{'tr1','tr2'},'r','s','d')
        self.assertEqual(a,b)
        self.assertAlmostEqual(a['coef_a'],1)
        self.assertAlmostEqual(a['intercept_b'],1)

    def test_failed_request_preserved_in_calibrated_export(self):
        rows = [dict(sample_id='ok',prediction_status='ok',predicted_tx_rate=10,error_reason=None),
                dict(sample_id='bad',prediction_status='failed',predicted_tx_rate=None,error_reason='missing')]
        result = apply_calibration(rows,dict(coef_a=1,intercept_b=1,calibration_id='c'))
        self.assertEqual(len(result),2)
        self.assertEqual(result[1]['prediction_status'],'failed')
        self.assertEqual(result[1]['error_reason'],'missing')
        self.assertEqual(result[1]['calibration_id'],'c')

    def test_metric_counts_are_subset_and_common_cohort_specific(self):
        base = dict(dataset_id='d',run_id='r',method_name='m',split_id='s',data_version='v',
                    target_scale_model='log10',calibration_id=None,predicted_value=100,predicted_value_log10=2,true_value=100)
        rows = [dict(base,sample_id='a',prediction_status='ok'),dict(base,sample_id='b',prediction_status='ok'),
                dict(base,sample_id='c',prediction_status='failed')]
        metrics = metric_rows(rows,'val',comparison_set_id='common',common_ids={'a'})
        self.assertTrue(all((r['n_requested'],r['n_success'],r['n_used'])==(3,2,1) for r in metrics))


if __name__=='__main__': unittest.main()
