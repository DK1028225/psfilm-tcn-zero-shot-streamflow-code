"""Regression checks for public metrics against formal Stage18 definitions."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psfilm_reference import metrics, summarize_seed_values

class FormalMetricsTests(unittest.TestCase):
    def test_below_ten_valid_pairs(self):
        result = metrics(np.arange(9.), np.arange(9.))
        self.assertEqual(result["n"], 9)
        for k in ("NSE", "KGE", "RMSE", "MAE", "PBIAS"):
            self.assertTrue(np.isnan(result[k]), k)

    def test_excludes_nonfinite_pairs(self):
        obs = np.arange(12., dtype=float)
        pred = obs.copy()
        obs[0] = np.nan
        pred[1] = np.inf
        result = metrics(obs, pred)
        self.assertEqual(result["n"], 10)
        self.assertAlmostEqual(result["NSE"], 1.0, places=12)
        self.assertAlmostEqual(result["KGE"], 1.0, places=6)

    def test_population_kge_and_pbias_sign(self):
        obs = np.arange(1., 21.)
        pred = 1.1 * obs + 0.3
        out = metrics(obs, pred)
        self.assertEqual(out["n"], 20)
        np.testing.assert_allclose(out["RMSE"], np.sqrt(np.mean((pred-obs)**2)), rtol=1e-12)
        self.assertGreater(out["PBIAS"], 0.)
        eps = 1e-8
        r = np.corrcoef(obs, pred)[0, 1]
        alpha = np.std(pred, ddof=0) / (np.std(obs, ddof=0) + eps)
        beta = np.mean(pred) / (np.mean(obs) + eps)
        kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
        np.testing.assert_allclose(out["KGE"], kge, rtol=1e-12)

    def test_constant_observation_kge_undefined(self):
        obs = np.ones(12)
        pred = np.arange(12., dtype=float)
        self.assertTrue(np.isnan(metrics(obs, pred)["KGE"]))

    def test_seed_sample_standard_deviation(self):
        result = summarize_seed_values([1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["sample_sd"], 1.0)

if __name__ == "__main__":
    unittest.main()
