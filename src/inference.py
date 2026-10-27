"""Frozen Stage18B-compatible checkpoint replay and held-out evaluation.

This module accepts prepared, authorized data loaders and frozen checkpoints.
It NEVER fits held-out scalers, trains PS-FiLM, or writes time series by default.

The public SharedFiLMTCN forward signature is (x, static, envelopes=None).
The formal Stage18 PS parameter clipping is equivalent to supplying the source
coefficient envelopes. Stage2A previously verified forward compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from psfilm_reference import metrics

POST2022_START = pd.Timestamp('2022-01-01 00:00:00')
FLOOD_MONTHS = (4, 5, 6, 7, 8, 9)
SEQ_LEN = 24


def inverse_y_to_q(pred_scaled, y_scaler, area_km2: float):
    """Stage18B units: standardized log runoff modulus -> discharge (m^3/s)."""
    pred_scaled = np.asarray(pred_scaled, dtype=float).reshape(-1, 1)
    pred_log_mod = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    pred_mod = np.expm1(pred_log_mod)
    pred_mod = np.clip(pred_mod, 0.0, None)
    return pred_mod * float(area_km2) / 1000.0


def calc_seasonal_nse(obs, pred, times):
    """Stage18B seasonal NSE requires at least 100 targets in each subset."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    months = pd.DatetimeIndex(pd.to_datetime(times)).month
    flood = np.isin(months, FLOOD_MONTHS)

    def _nse(mask):
        if np.sum(mask) < 100:
            return float('nan')
        o, p = obs[mask], pred[mask]
        return 1.0 - np.sum((p-o)**2) / (np.sum((o-np.mean(o))**2) + 1e-8)
    return _nse(flood), _nse(~flood)


def calc_persistence_skill(obs, model_pred, persistence_pred):
    """Stage18B persistence-relative SSE reduction; all predictions share a mask."""
    obs = np.asarray(obs, dtype=float)
    model_pred = np.asarray(model_pred, dtype=float)
    persistence_pred = np.asarray(persistence_pred, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(model_pred) & np.isfinite(persistence_pred)
    obs, model_pred, persistence_pred = obs[ok], model_pred[ok], persistence_pred[ok]
    sse_model = np.sum((model_pred - obs)**2)
    sse_pers = np.sum((persistence_pred - obs)**2)
    return float(1.0 - sse_model / (sse_pers + 1e-8))


def evaluate_arrays(obs, pred, persistence, dates):
    """Match every field returned by the formal Stage18B evaluate_arrays()."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    persistence = np.asarray(persistence, dtype=float)
    dates = pd.to_datetime(np.asarray(dates))
    m = metrics(obs, pred)
    nse_flood, nse_dry = calc_seasonal_nse(obs, pred, dates)
    pers_m = metrics(obs, persistence)
    skill = calc_persistence_skill(obs, pred, persistence)
    post = dates >= POST2022_START
    if np.sum(post) >= 10:
        m_post = metrics(obs[post], pred[post])
        pers_post = metrics(obs[post], persistence[post])
        skill_post = calc_persistence_skill(obs[post], pred[post], persistence[post])
    else:
        m_post = {key: float('nan') for key in ['NSE', 'KGE', 'RMSE', 'MAE', 'PBIAS']}
        m_post['n'] = int(np.sum(post))
        pers_post = {'NSE': float('nan')}
        skill_post = float('nan')
    return {
        'NSE': m['NSE'], 'KGE': m['KGE'], 'RMSE': m['RMSE'],
        'MAE': m['MAE'], 'PBIAS': m['PBIAS'], 'n': m['n'],
        'NSE_flood': float(nse_flood), 'NSE_dry': float(nse_dry),
        'Persistence_NSE': pers_m['NSE'], 'Skill_vs_Persistence': skill,
        'NSE_post2022': m_post['NSE'], 'KGE_post2022': m_post['KGE'],
        'RMSE_post2022': m_post['RMSE'], 'MAE_post2022': m_post['MAE'],
        'PBIAS_post2022': m_post['PBIAS'], 'n_post2022': m_post['n'],
        'Persistence_NSE_post2022': pers_post['NSE'],
        'Skill_vs_Persistence_post2022': skill_post,
    }


def compute_source_envelopes(model, source_static_scaled, device='cpu'):
    """Compute per-channel FiLM support from source-only standardized attributes.

    Caller MUST provide standardized static attributes from the training sources
    of the current fold (five for regional LOBO, or three in predefined study).
    """
    model.eval()
    z = torch.as_tensor(np.asarray(source_static_scaled, dtype=np.float32),
                        dtype=torch.float32, device=torch.device(device))
    if z.ndim != 2 or z.shape[1] != 4 or len(z) < 1:
        raise ValueError('source_static_scaled must have shape (source_count, 4)')
    bounds = []
    with torch.no_grad():
        for block in model.film_blocks:
            gamma, beta = block.coefficients(z)
            bounds.append({
                'gamma_min': gamma.min(0).values.view(1,-1),
                'gamma_max': gamma.max(0).values.view(1,-1),
                'beta_min': beta.min(0).values.view(1,-1),
                'beta_max': beta.max(0).values.view(1,-1),
            })
    return bounds


def _predict_single(model, loader, y_scaler, area_km2, static_vector_scaled,
                    *, envelopes=None, device='cpu'):
    """Preserve formal Stage18B seven-field data order and valid-target filtering."""
    model.eval()
    device = torch.device(device)
    z = torch.as_tensor(np.asarray(static_vector_scaled, dtype=np.float32).reshape(1,-1),
                        dtype=torch.float32, device=device)
    if z.shape[1] != 4:
        raise ValueError('target static vector must contain exactly four descriptors')
    pred_scaled, q_prev, dates, valid = [], [], [], []
    with torch.no_grad():
        for bX, bS, bY, bM, bID, bQPrev, bDate in loader:
            bX = bX.to(device, non_blocking=True)
            out = model(bX, z.expand(bX.shape[0], -1), envelopes)
            pred_scaled.append(out.detach().cpu().numpy())
            valid.append((1.0 - bM.numpy()).astype(bool))
            q_prev.append(bQPrev.numpy())
            dates.extend(list(bDate))
    if not pred_scaled:
        raise ValueError('Held-out loader contains no candidate windows')
    pred_scaled = np.concatenate(pred_scaled)
    valid = np.concatenate(valid)
    q_prev = np.concatenate(q_prev)
    # Formal Dataset is N-24; target observations begin at stored row SEQ_LEN-1.
    obs = np.asarray(loader.dataset.Q[SEQ_LEN-1:SEQ_LEN-1+len(pred_scaled)],dtype=float)
    if len(obs) != len(pred_scaled):
        raise ValueError('Dataset Q array is not aligned with N-24 model windows')
    dates = np.asarray(dates)
    pred_q = inverse_y_to_q(pred_scaled, y_scaler, area_km2)
    return {
        'obs_q': obs[valid], 'pred_q': pred_q[valid],
        'persistence_q': q_prev[valid], 'dates': dates[valid],
    }


def replay_raw_ps_pair(model, heldout_loader, y_scaler, area_km2,
                       source_static_scaled, target_static_scaled, *, device='cpu'):
    """Evaluate Raw and PS from ONE frozen FiLM-TCN model; no optimizer or fitting."""
    model = model.to(torch.device(device)).eval()
    envelopes = compute_source_envelopes(model, source_static_scaled, device)
    raw = _predict_single(model, heldout_loader, y_scaler, area_km2,
                          target_static_scaled, envelopes=None, device=device)
    ps = _predict_single(model, heldout_loader, y_scaler, area_km2,
                         target_static_scaled, envelopes=envelopes, device=device)
    for name in ('obs_q', 'persistence_q', 'dates'):
        if not np.array_equal(raw[name], ps[name], equal_nan=True if name != 'dates' else False):
            raise AssertionError(f'Raw and PS not aligned: {name}')
    report = {
        'FiLM-TCN': evaluate_arrays(raw['obs_q'], raw['pred_q'], raw['persistence_q'], raw['dates']),
        'PS-FiLM-TCN': evaluate_arrays(ps['obs_q'], ps['pred_q'], ps['persistence_q'], ps['dates']),
    }
    return report, {'FiLM-TCN': raw, 'PS-FiLM-TCN': ps}


def load_frozen_film_checkpoint(model, checkpoint_path, *, device='cpu'):
    """Load only trusted local PyTorch checkpoints. Never refit a model for PS."""
    # weights_only available in the exact tested PyTorch 2.4.1 environment.
    ckpt = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    if ckpt.get('model_name') not in ('Shared-FiLM', 'FiLM-TCN'):
        raise ValueError(f"Wrong trained checkpoint model_name={ckpt.get('model_name')!r}")
    if 'state_dict' not in ckpt:
        raise KeyError('Checkpoint has no state_dict')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return model.to(device).eval(), {'model_name': ckpt['model_name'], 'seed': ckpt.get('seed'),
                                    'best_epoch': ckpt.get('best_epoch')}


def write_summary_json(path, summary: dict):
    """Write aggregate metrics only. Raw hourly records are never saved here."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def scrub(item):
        if isinstance(item, dict):
            return {str(k):scrub(v) for k,v in item.items()}
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating, float)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, list):
            return [scrub(v) for v in item]
        return item
    path.write_text(json.dumps(scrub(summary), indent=2, ensure_ascii=False), encoding='utf-8')
