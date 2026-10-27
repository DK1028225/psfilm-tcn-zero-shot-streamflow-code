"""Stage2E-3 synthetic contracts; NEVER access hydrological observations."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from data_pipeline import (BASIN_PHYSICS, LOBO_BASIN_ORDER,
                           ORIGINAL_SOURCE_BASINS, build_fold_data)
from formal_fold_replay import replay_frozen_fold
from formal_training_models import build_frozen_model
from psfilm_reference import SharedFiLMTCN
from prepared_fold import PreparedHeldoutDataset


class FrozenFoldReplayTests(unittest.TestCase):
    def make_fold(self, mode='lobo', heldout='增江', n=32):
        sources = tuple(b for b in LOBO_BASIN_ORDER if b!=heldout) if mode=='lobo' else ORIGINAL_SOURCE_BASINS
        # Only static values are source-fitted. Dynamic targets are synthetic.
        scaler = StandardScaler().fit(np.array([BASIN_PHYSICS[b] for b in sources], dtype=np.float32))
        source_z = scaler.transform(np.array([BASIN_PHYSICS[b] for b in sources],dtype=np.float32)).astype(np.float32)
        target_z = scaler.transform(np.array([BASIN_PHYSICS[heldout]],dtype=np.float32)).astype(np.float32).reshape(-1)
        rng = np.random.default_rng(2024)
        x = rng.normal(size=(n,24,2)).astype(np.float32)
        q = np.arange(n+24,dtype=np.float32) + 11.0
        target = rng.normal(size=n).astype(np.float32)
        mask = np.zeros(n,dtype=np.float32);mask[::7]=1
        dates = [f'2022-05-{(i//24+1):02d} {(i%24):02d}:00:00' for i in range(n)]
        ds = PreparedHeldoutDataset(x, target_z, target, mask, 0, q[22:22+n], dates, q)
        return {
            'mode':mode,'sources':sources,'heldout':heldout,'heldout_loader':DataLoader(ds,batch_size=8,shuffle=False),
            'source_static_scaled':source_z,'static_scaler':scaler,
            'y_scaler':StandardScaler().fit(np.linspace(0.5,3.5,100).reshape(-1,1)),
        }

    def save_checkpoints(self, folder, seed=42):
        torch.manual_seed(11)
        ms = build_frozen_model('MS-TCN(RS)').eval()
        film = build_frozen_model('Shared-FiLM').eval()
        ms_file = Path(folder)/'ms.pth'
        film_file = Path(folder)/'film.pth'
        torch.save({'model_name':'MS-TCN(RS)','seed':seed,'best_epoch':2,'state_dict':ms.state_dict()}, ms_file)
        torch.save({'model_name':'Shared-FiLM','seed':seed,'best_epoch':3,'state_dict':film.state_dict()}, film_file)
        return ms_file, film_file

    def test_lobo_replay_all_four_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            ms, film = self.save_checkpoints(tmp)
            a = replay_frozen_fold(self.make_fold(),heldout_name='Zengjiang',seed=42,
                                   ms_checkpoint=ms,film_checkpoint=film)
            self.assertEqual(tuple(a['metrics']),('MS-TCN(RS)','FiLM-TCN','PS-FiLM-TCN','Persistence'))
            self.assertTrue(a['metadata']['raw_ps_shared_checkpoint'])
            self.assertTrue(a['metadata']['trained_film_state_unchanged'])
            self.assertEqual({m['n'] for m in a['metrics'].values()},{32-len(range(0,32,7))})
            self.assertAlmostEqual(a['metrics']['Persistence']['Skill_vs_Persistence'],0.0)

    def test_three_source_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            ms, film = self.save_checkpoints(tmp,seed=123)
            x = replay_frozen_fold(self.make_fold(mode='predefined3',heldout='秋香江'),
                                   heldout_name='Qiuxiangjiang',seed=123,
                                   ms_checkpoint=ms,film_checkpoint=film)
            self.assertEqual(x['metadata']['experiment'],'predefined3')
            self.assertEqual(len(x['metadata']['source_basins']),3)

    def test_wrong_seed_wrong_model_and_same_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ms, film = self.save_checkpoints(tmp)
            args=dict(heldout_name='Zengjiang',seed=42,ms_checkpoint=ms,film_checkpoint=film)
            with self.assertRaises(ValueError):
                replay_frozen_fold(self.make_fold(),**{**args,'film_checkpoint':ms})
            with self.assertRaises(ValueError):
                replay_frozen_fold(self.make_fold(),**{**args,'ms_checkpoint':film})
            with self.assertRaises(ValueError):
                replay_frozen_fold(self.make_fold(),**{**args,'seed':123})

    def test_heldout_must_not_be_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            ms, film = self.save_checkpoints(tmp)
            fold = self.make_fold()
            fold['sources'] = tuple(fold['sources'])+('增江',)
            with self.assertRaises(ValueError):
                replay_frozen_fold(fold,heldout_name='增江',seed=42,
                                   ms_checkpoint=ms,film_checkpoint=film)

    def test_live_stage2e1_fold_adapter_with_synthetic_source_data(self):
        # Unlike the prepared-NPZ unit tests, exercise the actual Stage2E-1
        # source-training scaler / held-out PhysicsHydroDataset interface.
        n=170
        frames={}
        for k,basin in enumerate(LOBO_BASIN_ORDER):
            q=5.0+0.2*np.arange(n,dtype=float)+k
            area=BASIN_PHYSICS[basin][0]
            log=np.log1p(1000*q/area)
            prev=np.zeros(n);prev[1:]=log[:-1]
            frames[basin]=pd.DataFrame({
                '流域面雨量(mm)':np.sin(np.arange(n)/10.0)+1.0,
                'Log前期径流模数':prev,
                'Log径流模数':log,
                '流量_缺失指示':np.zeros(n,dtype=int),
                '流量(m³/s)':q,
                '时间':pd.date_range('2022-06-01',periods=n,freq='h'),
            })
        fold=build_fold_data(frames,heldout_basin='增江',seed=42)
        self.assertEqual(fold['mode'],'lobo')
        with tempfile.TemporaryDirectory() as tmp:
            ms,film=self.save_checkpoints(tmp)
            out=replay_frozen_fold(fold,heldout_name='Zengjiang',seed=42,
                                   ms_checkpoint=ms,film_checkpoint=film)
            self.assertEqual(out['metrics']['FiLM-TCN']['n'],n-24)
            self.assertEqual(out['metrics']['PS-FiLM-TCN']['n'],n-24)
            self.assertTrue(out['metadata']['source_only_scalers'])

    def test_incorrect_target_scaler_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ms, film = self.save_checkpoints(tmp)
            fold = self.make_fold()
            fold['static_scaler'] = StandardScaler().fit(np.random.default_rng(1).normal(size=(5,4)))
            with self.assertRaises(AssertionError):
                replay_frozen_fold(fold,heldout_name='增江',seed=42,
                                   ms_checkpoint=ms,film_checkpoint=film)

if __name__=='__main__':
    unittest.main()
