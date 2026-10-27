"""Regression tests for the frozen Stage18B inference/evaluation conventions."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from inference import (inverse_y_to_q, calc_seasonal_nse, calc_persistence_skill,
                       evaluate_arrays, load_frozen_film_checkpoint, write_summary_json)
from psfilm_reference import SharedFiLMTCN

class InferenceTests(unittest.TestCase):
    def test_inverse_transform_exp_m1_and_clip(self):
        scaler=StandardScaler().fit(np.linspace(0,10,100).reshape(-1,1))
        log_values=np.array([-4.0,0.0,np.log1p(1234.0)])
        z=scaler.transform(log_values[:,None]).reshape(-1)
        q=inverse_y_to_q(z,scaler,50.0)
        np.testing.assert_allclose(q,[0.,0.,1234*50/1000],atol=1e-9)

    def test_season_threshold_and_flood_months(self):
        n=99
        dates=pd.date_range('2021-04-01',periods=n,freq='h')
        a,b=calc_seasonal_nse(np.arange(n),np.arange(n),dates)
        self.assertTrue(np.isnan(a) and np.isnan(b))

    def test_skill_uses_joint_finite_records(self):
        obs=np.array([1.,2.,3.,np.nan])
        model=np.array([2.,3.,4.,4.])
        persist=np.array([1.,3.,4.,5.])
        out=calc_persistence_skill(obs,model,persist)
        expected=1-np.sum(np.array([1.,1.,1.])**2)/(np.sum(np.array([0.,1.,1.])**2)+1e-8)
        self.assertAlmostEqual(out,expected)

    def test_post2022_fallback_n(self):
        dates=pd.date_range('2021-12-31',periods=20,freq='h')
        obs=np.arange(20,dtype=float)
        report=evaluate_arrays(obs,obs+1,obs+2,dates)
        self.assertEqual(report['n_post2022'],0)
        self.assertTrue(np.isnan(report['NSE_post2022']))
        self.assertEqual(report['n'],20)

    def test_untrusted_model_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wrong.pth'
            torch.save({'model_name':'MS-TCN(RS)','state_dict':{}},p)
            with self.assertRaises(ValueError):
                load_frozen_film_checkpoint(SharedFiLMTCN(),p)

    def test_summary_json_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'summary.json'
            write_summary_json(p, {'NSE':float('nan'),'n':np.int64(11)})
            import json
            obj=json.loads(p.read_text(encoding='utf8'))
            self.assertIsNone(obj['NSE'])
            self.assertEqual(obj['n'],11)

if __name__=='__main__':
    unittest.main()
