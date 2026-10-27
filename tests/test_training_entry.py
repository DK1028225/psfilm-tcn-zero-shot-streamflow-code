"""Stage2E-2 synthetic-only tests of formal model and orchestration contracts."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from formal_training_models import (
    build_frozen_model, EXPECTED_MS_PARAMETERS, EXPECTED_FILM_PARAMETERS,
)
from experiment_runner import validate_training_request, train_prepared_fold
from psfilm_reference import SharedFiLMTCN
from training import TrainConfig


class ModelContractTests(unittest.TestCase):
    def test_frozen_parameter_counts_and_output_shapes(self):
        ms=build_frozen_model('MS-TCN(RS)').eval()
        film=build_frozen_model('Shared-FiLM').eval()
        self.assertEqual(sum(p.numel() for p in ms.parameters()),EXPECTED_MS_PARAMETERS)
        self.assertEqual(sum(p.numel() for p in film.parameters()),EXPECTED_FILM_PARAMETERS)
        x=torch.randn(2,24,2)
        z=torch.randn(2,4)
        with torch.no_grad():
            self.assertEqual(tuple(ms(x).shape),(2,))
            self.assertEqual(tuple(film(x,z,torch.ones(2)).shape),(2,))

    def test_gate_is_formally_one_and_not_projection_bounds(self):
        f=build_frozen_model('Shared-FiLM')
        x=torch.randn(2,24,2)
        z=torch.randn(2,4)
        with self.assertRaises(ValueError):
            f(x,z,torch.tensor([0.5,1.]))
        with self.assertRaises(TypeError):
            f(x,z,[{'gamma_min':0.}])
        with self.assertRaises(ValueError):
            build_frozen_model('PS-FiLM-TCN')

    def test_checkpoint_keys_and_replay_model_are_identical(self):
        torch.manual_seed(33)
        adapter=build_frozen_model('Shared-FiLM').eval()
        replay=SharedFiLMTCN().eval()
        replay.load_state_dict(adapter.state_dict(),strict=True)
        self.assertEqual(list(adapter.state_dict()),list(replay.state_dict()))
        x=torch.randn(3,24,2)
        z=torch.randn(3,4)
        with torch.no_grad():
            y0=adapter(x,z,torch.ones(3))
            y1=replay(x,z,None)
        torch.testing.assert_close(y0,y1,atol=1e-6,rtol=1e-6)

    def test_frozen_roles_and_seeds(self):
        for seed in [42,123,2024,2025,3407]:
            target,sources=validate_training_request('lobo',seed,'Zengjiang','Shared-FiLM')
            self.assertEqual(target,'增江')
            self.assertEqual(len(sources),5)
        target,sources=validate_training_request('predefined3',42,None,'MS-TCN(RS)')
        self.assertIsNone(target)
        self.assertEqual(sources,('安墩河','密溪水','船塘河'))
        with self.assertRaises(ValueError):
            validate_training_request('lobo',17,'Zengjiang','Shared-FiLM')
        with self.assertRaises(ValueError):
            validate_training_request('predefined3',42,'Zengjiang','Shared-FiLM')
        with self.assertRaises(ValueError):
            validate_training_request('lobo',42,'Zengjiang','PS-FiLM-TCN')
        with self.assertRaises(ValueError):
            validate_training_request('lobo',42,None,'Shared-FiLM')


if __name__=='__main__':
    unittest.main()
