"""Input adapter shape checks; no real data or stored checkpoints."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepared_fold import PreparedHeldoutDataset,FrozenScaler

class PreparedFoldTests(unittest.TestCase):
    def test_missing_mask_target_and_previous_q_alignment(self):
        n=15
        q=np.arange(n+24,dtype=float)+1
        dataset=PreparedHeldoutDataset(np.zeros((n,24,2)),[0,0,0,0],np.zeros(n),
                                       np.zeros(n),0,q[22:n+22],['2022-01-01']*n,q)
        self.assertEqual(len(dataset),n)
        self.assertEqual(float(dataset[0][5]),float(q[22]))
        self.assertEqual(float(dataset.Q[23]),float(q[23]))

    def test_reject_bad_q_length(self):
        with self.assertRaises(ValueError):
            PreparedHeldoutDataset(np.zeros((10,24,2)),[0,0,0,0],np.zeros(10),
                                   np.zeros(10),0,np.zeros(10),['2022']*10,np.zeros(33))

    def test_frozen_scaler(self):
        scaler=FrozenScaler(2.,3.)
        np.testing.assert_allclose(scaler.inverse_transform([[0.],[1.]]).reshape(-1),[2.,5.])

if __name__=='__main__':
    unittest.main()
