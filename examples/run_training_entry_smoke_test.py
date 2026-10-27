"""Synthetic single-fold integration test with both frozen model architectures.

Uses artificial precipitation and discharge and a disposable temporary folder.
NEVER accesses actual observations, full official checkpoints or the private repo.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_pipeline import RAIN_COLUMNS, ORIGINAL_SOURCE_BASINS, preprocess_unified_df, build_fold_data
from experiment_runner import train_prepared_fold
from training import TrainConfig
from psfilm_reference import SharedFiLMTCN


def synth_csv(path: Path, basin: str, n: int, index: int) -> None:
    t=np.arange(n,dtype=float)
    d={'time':pd.date_range('2017-01-01',periods=n,freq='h').astype(str)}
    for k, column in enumerate(RAIN_COLUMNS[basin]):
        d[column]=0.2+0.3*np.sin(t/21.0+index+k)**2
    d['流量(m³/s)']=2.5+0.35*np.sin(t/30.+index)+0.07*np.cos(t/14.+index)
    pd.DataFrame(d).to_csv(path,index=False,encoding='utf-8-sig')


def main():
    torch.set_num_threads(min(torch.get_num_threads(),2))
    with tempfile.TemporaryDirectory(prefix='stage2e2_synthetic_') as temp:
        root=Path(temp)
        data={}
        for j, basin in enumerate(ORIGINAL_SOURCE_BASINS):
            p=root/f'synthetic_source_{j}.csv'
            synth_csv(p,basin,275+10*j,j)
            data[basin]=preprocess_unified_df(p,basin)[0]
        fold=build_fold_data(data,source_basins=ORIGINAL_SOURCE_BASINS,seed=42)
        assert fold['mode']=='predefined3'
        assert fold['train_loader'].sampler.replacement
        assert fold['heldout'] is None
        for model_name,folder in (('MS-TCN(RS)','ms'),('Shared-FiLM','film')):
            run_dir=root/folder
            out=train_prepared_fold(
                fold,42,model_name,run_dir,device='cpu',
                config=TrainConfig(epochs=1,early_stopping_patience=15),
                synthetic=True,
            )
            assert out['best_epoch']==1
            assert (run_dir/'train_log.csv').is_file()
            assert (run_dir/'run_metadata.json').is_file()
            ckpt=torch.load(run_dir/'best_model.pth',map_location='cpu',weights_only=False)
            assert ckpt['model_name']==model_name
            if model_name=='Shared-FiLM':
                replay=SharedFiLMTCN()
                replay.load_state_dict(ckpt['state_dict'],strict=True)
            print(f'[PASS] {model_name}: synthetic training, log, checkpoint, replay keys')
        # Safety proof: existing checkpoint cannot be silently overwritten.
        try:
            train_prepared_fold(fold,42,'Shared-FiLM',root/'film',
                                device='cpu',config=TrainConfig(epochs=1),synthetic=True)
        except FileExistsError:
            print('[PASS] Existing checkpoint overwrite refused')
        else:
            raise AssertionError('Refused overwrite check failed')
    print('[STATUS] STAGE2E2_SYNTHETIC_SINGLE_FOLD_PASS = True')


if __name__=='__main__':
    main()
