"""Release tests: chronological eligibility, portable trees and joint mapping."""
import unittest
import numpy as np
from center_runtime import artifact, correct_draws, predict_trees


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(401)
        self.values=2+np.cumsum(rng.normal(0,.03,(320,2)),axis=0)
        self.draws=self.values[-1]+rng.normal(0,.1,(1000,2,2))
        self.args=(self.draws,self.values,[2,12],['UST_2Y','UST_10Y'],'level')

    def test_cutoff_same_day_and_before_block_all_corrections(self):
        for day in ('2016-01-01','2019-10-23','2020-07-21'):
            adjusted,info=correct_draws(*self.args,day)
            np.testing.assert_array_equal(adjusted,self.draws)
            self.assertEqual(info['width_factor'],1.)
            self.assertFalse(info['center_applied'])

    def test_eligible_correction_is_joint_affine_and_exact_width(self):
        adjusted,info=correct_draws(*self.args,'2026-10-23')
        self.assertTrue(info['center_applied'])
        self.assertEqual(info['width_factor'],1.5)
        median=np.median(self.draws,axis=0)
        expected=median+np.array(info['delta'])+1.5*(self.draws-median)
        np.testing.assert_allclose(adjusted,expected,rtol=0,atol=0)
        np.testing.assert_array_equal(np.argsort(adjusted,axis=0),np.argsort(self.draws,axis=0))
        width=np.diff(np.quantile(adjusted,[.05,.95],axis=0),axis=0)
        old=np.diff(np.quantile(self.draws,[.05,.95],axis=0),axis=0)
        np.testing.assert_allclose(width,1.5*old,atol=1e-12)

    def test_short_monthly_and_transfer_fallbacks_only_skip_centre(self):
        for kwargs in (dict(frequency='monthly'),dict(transfer=True),dict(scales=4)):
            adjusted,info=correct_draws(*self.args,'2026-10-23',**kwargs)
            self.assertFalse(info['center_applied'])
            self.assertEqual(info['width_factor'],1.5)
            np.testing.assert_array_equal(info['delta_sd'],np.zeros((2,2)))
        adjusted,info=correct_draws(self.draws,self.values[-200:],[2,12],
            ['UST_2Y','UST_10Y'],'level','2026-10-23')
        self.assertFalse(info['center_applied'])

    def test_single_asset_shape_and_future_history_cannot_modify_blocked_output(self):
        single=self.draws[:,:,0]
        adjusted,info=correct_draws(single,self.values[:,0],[2,12],['UST_2Y'],
            'level','2026-10-23')
        self.assertEqual(adjusted.shape,single.shape)
        self.assertTrue(info['center_applied'])
        blocked,_=correct_draws(single,self.values[:,0]+1000,[2,12],['UST_2Y'],
            'level','2010-01-01')
        np.testing.assert_array_equal(blocked,single)

    def test_raw_numeric_split_and_pre_scaled_leaf_sum(self):
        model=dict(baseline=.25,trees=[[[0.,0,1.,1,2,False],
            [2.,0,0.,0,0,True],[-2.,0,0.,0,0,True]]])
        X=np.zeros((2,14));X[1,0]=2.
        np.testing.assert_array_equal(predict_trees(model,X),[2.25,-1.75])
        with self.assertRaises(ValueError):predict_trees(model,np.full((2,14),np.nan))

    def test_model_has_no_training_examples_or_answer_index(self):
        model,digest=artifact()
        self.assertEqual(len(digest),64)
        self.assertEqual(model['training_cutoff'],'2019-10-23')
        self.assertEqual(model['selection_cutoff'],'2020-07-21')
        self.assertFalse(model['provenance']['inference_corpus_packaged'])
        self.assertTrue(set(model).isdisjoint({'X','y','labels','unit_ids','panels','forecasts'}))


if __name__=='__main__':unittest.main()
