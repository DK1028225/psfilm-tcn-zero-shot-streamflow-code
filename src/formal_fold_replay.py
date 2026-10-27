"""Stage2E-3 read-only, frozen single-fold four-model evaluation.

This adapter CONNECTS existing Stage2D inference, Stage2E-1 source-only fold
preparation, and Stage2E-2 independently trained MS/FiLM architectures.
No training, fitted target scaler, checkpoint modification or raw NPZ export.

The two caller-provided checkpoints must be local and trusted; torch.load
(weights_only=True) still should not process arbitrary unaudited weights.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from data_pipeline import (BASIN_PHYSICS, LOBO_BASIN_ORDER,
                           ORIGINAL_SOURCE_BASINS, OFFICIAL_SEEDS,
                           build_fold_data, canonical_basin_name,
                           load_authorized_basin_csvs)
from formal_training_models import MSTCN_RS
from inference import (inverse_y_to_q, evaluate_arrays, replay_raw_ps_pair,
                       load_frozen_film_checkpoint, write_summary_json)
from psfilm_reference import SharedFiLMTCN

FORMAL_REPLAY_MODEL_NAMES = ("MS-TCN(RS)", "FiLM-TCN", "PS-FiLM-TCN", "Persistence")


def _trusted_checkpoint(model: torch.nn.Module, path: str | Path,
                        expected_name: str, expected_seed: int, device: str):
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise FileNotFoundError(p)
    info = torch.load(p, map_location='cpu', weights_only=True)
    if not isinstance(info, dict) or info.get('model_name') != expected_name:
        raise ValueError('Incorrect checkpoint type; expected ' + expected_name)
    if info.get('seed') is None or int(info['seed']) != expected_seed:
        raise ValueError('Checkpoint seed does not match the requested fold/seed')
    if not isinstance(info.get('state_dict'), dict):
        raise ValueError('Missing checkpoint state_dict')
    model.load_state_dict(info['state_dict'], strict=True)
    model = model.to(torch.device(device)).eval()
    return model, {'model_name': expected_name,
                   'seed': int(info['seed']), 'best_epoch': info.get('best_epoch')}


def _single_ms(model, heldout_loader, y_scaler, area_km2, *, device='cpu'):
    """Preserve Stage18/Stage2D seven-field loader order and N-24 target mask."""
    model.eval()
    device = torch.device(device)
    pred_parts, valid_parts, prev_parts, times = [], [], [], []
    with torch.no_grad():
        for x, static, y, mask, basin_id, q_prev, dates in heldout_loader:
            pred_parts.append(model(x.to(device, non_blocking=True)).detach().cpu().numpy())
            valid_parts.append((1.0-mask.numpy()).astype(bool))
            prev_parts.append(q_prev.numpy())
            times.extend(list(dates))
    if not pred_parts:
        raise ValueError('No held-out candidate windows')
    pred = np.concatenate(pred_parts)
    valid = np.concatenate(valid_parts)
    q_prev = np.concatenate(prev_parts)
    full_q = np.asarray(heldout_loader.dataset.Q, dtype=float)
    # Stage18: __len__=N-24, target index i+23; last stored row is unused.
    obs_q = full_q[23:23+len(pred)]
    if len(obs_q) != len(pred) or len(q_prev) != len(pred):
        raise AssertionError('Held-out target/previous discharge alignment mismatch')
    return {'obs_q':obs_q[valid],
            'pred_q':inverse_y_to_q(pred, y_scaler, area_km2)[valid],
            'persistence_q':q_prev[valid],
            'dates':np.asarray(times)[valid]}


def _aligned(reference, other, *, label):
    for key in ('obs_q', 'persistence_q', 'dates'):
        x, y = np.asarray(reference[key]), np.asarray(other[key])
        if not np.array_equal(x, y, equal_nan=(key != 'dates')):
            raise AssertionError(f'{label}: {key} does not align')


def replay_frozen_fold(fold: dict, *, heldout_name: str, seed: int,
                       ms_checkpoint: str | Path,
                       film_checkpoint: str | Path,
                       device: str = 'cpu'):
    """Read checkpoint states, produce four aligned aggregate metrics, no writes.

    All scalers and source modulation bounds MUST come from Stage2E-1 fold, fitted
    only on its source training partitions. PS uses the SAME FiLM state_dict as
    Raw, applying channel-wise source coefficient bounds solely at inference.
    """
    heldout = canonical_basin_name(heldout_name)
    if type(seed) is not int or seed not in OFFICIAL_SEEDS:
        raise ValueError('Official replay requires frozen official seed')
    if fold.get('heldout') != heldout or heldout in fold.get('sources', ()):
        raise ValueError('Fold held-out role is inconsistent with model sources')
    sources = tuple(fold.get('sources', ()))
    expected = tuple(b for b in LOBO_BASIN_ORDER if b != heldout)
    if fold.get('mode') == 'lobo':
        if sources != expected:
            raise ValueError('Unexpected source basin order in official LOBO fold')
    elif fold.get('mode') == 'predefined3':
        if sources != ORIGINAL_SOURCE_BASINS or heldout in sources:
            raise ValueError('Unexpected predefined three-source experiment')
    else:
        raise ValueError('Unknown frozen experiment mode')
    for field in ('heldout_loader','y_scaler','source_static_scaled','static_scaler'):
        if field not in fold or fold[field] is None:
            raise ValueError('Missing source-only preparation element: ' + field)
    z_source = np.asarray(fold['source_static_scaled'], dtype=np.float32)
    if z_source.shape != (len(sources),4):
        raise ValueError('Source static attributes do not match frozen source count')
    loader = fold['heldout_loader']
    if len(loader.dataset) <= 0:
        raise ValueError('No evaluation windows in held-out basin')
    # Stage2E-1 PhysicsHydroDataset uses `static`; Stage2D prepared adapter uses `S`.
    ds = loader.dataset
    if hasattr(ds, 'static'):
        static_raw = ds.static
    elif hasattr(ds, 'S'):
        static_raw = ds.S
    else:
        raise ValueError('Held-out dataset does not expose its target static vector')
    z_target = np.asarray(static_raw, dtype=np.float32).reshape(-1)
    if z_target.shape != (4,) or not np.isfinite(z_target).all():
        raise ValueError('Target static vector malformed')
    # Cross-check target representation using the source-trained static scaler.
    expected_z = fold['static_scaler'].transform(
        np.asarray(BASIN_PHYSICS[heldout], dtype=np.float32).reshape(1,-1)
    ).astype(np.float32).reshape(-1)
    if not np.array_equal(z_target, expected_z):
        raise AssertionError('Target static scaler must come from source-only fold')
    # Reject using the same checkpoint for the independently-trained networks.
    if Path(ms_checkpoint).resolve() == Path(film_checkpoint).resolve():
        raise ValueError('MS and FiLM must use distinct trained checkpoints')
    device_t = torch.device(device)
    if device_t.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA device was requested but is unavailable')
    ms, ms_meta = _trusted_checkpoint(MSTCN_RS(), ms_checkpoint,
                                      'MS-TCN(RS)', seed, device)
    film, film_meta = load_frozen_film_checkpoint(
        SharedFiLMTCN(), film_checkpoint, device=device
    )
    # The existing Stage2D loader permits absent seed; reject it for official E3.
    if film_meta.get('seed') is None or int(film_meta['seed']) != seed:
        raise ValueError('Shared-FiLM checkpoint seed mismatch')
    # Validate PS never mutates weights, not even in memory.
    before = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                     for v in film.state_dict().values())).hexdigest()
    ms_arrays = _single_ms(ms, loader, fold['y_scaler'], BASIN_PHYSICS[heldout][0],device=device)
    film_reports, film_arrays = replay_raw_ps_pair(
        film, loader, fold['y_scaler'], BASIN_PHYSICS[heldout][0],
        z_source, z_target, device=device
    )
    after = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                                    for v in film.state_dict().values())).hexdigest()
    if before != after:
        raise AssertionError('PS inference modified the Shared-FiLM checkpoint parameters')
    _aligned(ms_arrays, film_arrays['FiLM-TCN'], label='MS vs Raw')
    _aligned(ms_arrays, film_arrays['PS-FiLM-TCN'], label='MS vs PS')
    obs, persistence, times = (ms_arrays[k] for k in ('obs_q','persistence_q','dates'))
    result = {
        'MS-TCN(RS)': evaluate_arrays(obs, ms_arrays['pred_q'], persistence, times),
        'FiLM-TCN': film_reports['FiLM-TCN'],
        'PS-FiLM-TCN': film_reports['PS-FiLM-TCN'],
        'Persistence': evaluate_arrays(obs, persistence, persistence, times),
    }
    if len({result[k]['n'] for k in result}) != 1:
        raise AssertionError('Valid-target counts differ across models')
    metadata = {
        'experiment': fold['mode'], 'heldout': heldout, 'seed': seed,
        'source_basins': list(sources),
        'source_only_scalers': True,
        'raw_ps_shared_checkpoint': True,
        'ps_inference_only': True, 'trained_film_state_unchanged': before == after,
        'matched_target_and_persistence_across_models': True,
        'dataset_length_convention': 'N-24, target at i+23',
        'target_hour_observed_precipitation_used': True,
        'checkpoint_metadata': {'MS-TCN(RS)': ms_meta, 'Shared-FiLM': film_meta},
        'n_valid_targets': int(result['FiLM-TCN']['n']),
        'evidence_level': 'checkpoint_replay_with_authorized_data',
        'limitations': 'Four-model replay for a provided single fold only; no new training or model selection.',
    }
    return {'metadata': metadata, 'metrics': result}


def replay_authorized_fold(path_mapping: dict, *, experiment: str,
                           heldout_name: str, seed: int,
                           ms_checkpoint: str | Path,
                           film_checkpoint: str | Path,
                           device='cpu') -> dict:
    """Read user-authorized inputs; fit source TRAIN scalers only. Replay, never train."""
    heldout = canonical_basin_name(heldout_name)
    if experiment not in ('lobo','predefined3'):
        raise ValueError('Only frozen lobo/predefined3 experiments are supported')
    required = LOBO_BASIN_ORDER if experiment == 'lobo' else ORIGINAL_SOURCE_BASINS + (heldout,)
    data, _audit = load_authorized_basin_csvs(path_mapping, basins=required)
    if experiment == 'lobo':
        fold = build_fold_data(data, heldout_basin=heldout, seed=seed)
    else:
        fold = build_fold_data(data, heldout_basin=heldout,
                               source_basins=ORIGINAL_SOURCE_BASINS, seed=seed)
    return replay_frozen_fold(
        fold, heldout_name=heldout, seed=seed,
        ms_checkpoint=ms_checkpoint, film_checkpoint=film_checkpoint, device=device,
    )
