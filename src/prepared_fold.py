"""Validation adapter for preprocessed, AUTHORIZED held-out LOBO inputs.

The private .npz used by this adapter is never part of the GitHub repository.
The loader is strict about array shape and target/observation alignment.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FrozenScaler:
    """Univariate source-only target scaler, sufficient for inverse_transform."""
    def __init__(self, mean, scale):
        self.mean_ = np.asarray([float(mean)], dtype=float)
        self.scale_ = np.asarray([float(scale)], dtype=float)
        if not np.isfinite(self.mean_[0]) or not (np.isfinite(self.scale_[0]) and self.scale_[0] > 0):
            raise ValueError('invalid scaler statistics')

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=float).reshape(-1,1)
        return x * self.scale_ + self.mean_


class PreparedHeldoutDataset(Dataset):
    def __init__(self, dynamic, static, target_scaled, missing_mask,
                 basin_id, previous_discharge, dates, q_full):
        self.X = np.asarray(dynamic, dtype=np.float32)
        self.S = np.asarray(static, dtype=np.float32).reshape(-1)
        self.Y = np.asarray(target_scaled, dtype=np.float32).reshape(-1)
        self.M = np.asarray(missing_mask, dtype=np.float32).reshape(-1)
        self.ID = int(basin_id)
        self.QPrev = np.asarray(previous_discharge, dtype=np.float32).reshape(-1)
        self.Dates = np.asarray(dates).astype(str).reshape(-1)
        self.Q = np.asarray(q_full, dtype=float).reshape(-1)
        n = len(self.X)
        if self.X.shape != (n,24,2) or self.S.shape != (4,):
            raise ValueError('Expected N×24×2 dynamic input and four target static descriptors')
        if not all(len(x)==n for x in (self.Y,self.M,self.QPrev,self.Dates)):
            raise ValueError('Inconsistent N across prepared input arrays')
        if len(self.Q) != n+24:
            raise ValueError('Full Q length must equal N+24 (formal N-24 dataset convention)')
        if not np.isin(self.M,[0,1]).all():
            raise ValueError('Missing-target mask must be 0 (valid) or 1 (missing)')
        if not (np.isfinite(self.X).all() and np.isfinite(self.S).all() and np.isfinite(self.QPrev).all()):
            raise ValueError('Nonfinite input arrays in prepared fold')

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        return (torch.from_numpy(self.X[index]),torch.from_numpy(self.S),
                torch.tensor(self.Y[index]),torch.tensor(self.M[index]),
                torch.tensor(self.ID,dtype=torch.long),
                torch.tensor(self.QPrev[index]),self.Dates[index])


def load_prepared_fold(npz_path, batch_size=256):
    npz_path=Path(npz_path)
    with np.load(npz_path,allow_pickle=False) as data:
        required=['dynamic_x','target_scaled','missing_mask','q_previous',
                  'target_dates','q_full','source_static_scaled',
                  'target_static_scaled','y_scaler_mean','y_scaler_scale',
                  'basin_area_km2']
        absent=[key for key in required if key not in data.files]
        if absent:
            raise KeyError(f'Prepared NPZ lacks keys: {absent}')
        arrays={key:np.asarray(data[key]).copy() for key in required}
        basin_id=int(data['basin_id'].item()) if 'basin_id' in data.files else 0
        label=str(data['heldout_name'].item()) if 'heldout_name' in data.files else 'heldout'
        source_names=[str(s) for s in data['source_names']] if 'source_names' in data.files else []
        if source_names and label in source_names:
            raise ValueError('Target basin unexpectedly listed as a training source')
        if source_names and len(set(source_names))!=len(source_names):
            raise ValueError('Duplicate source basin name')
    source_z=arrays['source_static_scaled'].astype(np.float32)
    target_z=arrays['target_static_scaled'].astype(np.float32).reshape(-1)
    if source_z.ndim!=2 or source_z.shape[1]!=4 or target_z.shape!=(4,):
        raise ValueError('Source and target static array shape mismatch')
    area=float(arrays['basin_area_km2'].reshape(-1)[0])
    if not np.isfinite(area) or area<=0:
        raise ValueError('Basin area must be positive and finite')
    ds=PreparedHeldoutDataset(arrays['dynamic_x'],target_z,
                              arrays['target_scaled'],arrays['missing_mask'],
                              basin_id,arrays['q_previous'],arrays['target_dates'],
                              arrays['q_full'])
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0)
    scaler=FrozenScaler(arrays['y_scaler_mean'].reshape(-1)[0],
                        arrays['y_scaler_scale'].reshape(-1)[0])
    return loader, scaler, area, source_z, target_z, label
