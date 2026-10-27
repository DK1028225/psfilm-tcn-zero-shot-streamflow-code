"""Stage2E-2 frozen single-fold training orchestration (no method selection).

Requires explicitly authorized local CSV paths. It reuses Stage2E-1 fold builders,
Stage2C training core, and the formal-model compatible declarations. Public tests
only run synthetic data in temporary directories; nothing retrains official runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from data_pipeline import (
    OFFICIAL_SEEDS, ORIGINAL_SOURCE_BASINS, LOBO_BASIN_ORDER,
    canonical_basin_name, load_authorized_basin_csvs, build_fold_data,
)
from formal_training_models import build_frozen_model
from training import TrainConfig, train_one_model, set_seed

FORMAL_MODEL_NAMES = ('MS-TCN(RS)', 'Shared-FiLM')


def validate_training_request(
    experiment: str, seed: int, heldout: str | None, model_name: str,
    *, synthetic: bool = False,
) -> tuple[str | None, tuple[str, ...]]:
    """Freeze official trainable model names, seeds and experiment roles."""
    if experiment not in ('lobo', 'predefined3'):
        raise ValueError('Only the historical lobo and predefined3 designs are supported')
    if model_name not in FORMAL_MODEL_NAMES:
        raise ValueError('Only MS-TCN(RS) and Shared-FiLM are independently trainable')
    if not synthetic and (type(seed) is not int or seed not in OFFICIAL_SEEDS):
        raise ValueError(f'Official seeds must be one of {OFFICIAL_SEEDS}')
    if experiment == 'lobo':
        if heldout is None:
            raise ValueError('LOBO training requires one held-out basin')
        target = canonical_basin_name(heldout)
        return target, tuple(b for b in LOBO_BASIN_ORDER if b != target)
    if heldout is not None:
        raise ValueError('Predefined3 training fits one three-source model; no held-out target is needed')
    return None, ORIGINAL_SOURCE_BASINS


def _within_repo(path: Path) -> bool:
    """Guard against placing any training output inside a Git repository."""
    candidate = path.resolve()
    return any((base / '.git').exists() for base in (candidate, *candidate.parents))


def train_prepared_fold(
    fold: dict, seed: int, model_name: str, run_dir: str | Path, *,
    device: str | torch.device = 'cpu',
    config: TrainConfig = TrainConfig(),
    synthetic: bool = False,
) -> dict:
    """Train ONE frozen network with an already source-only prepared fold.

    The calling workflow explicitly chooses the output path and must use a
    fresh checkpoint directory. Raw and PS later share the trained FiLM state.
    """
    if model_name not in FORMAL_MODEL_NAMES:
        raise ValueError('PS-FiLM is inference-only; it cannot have its own training run')
    if not synthetic and (type(seed) is not int or seed not in OFFICIAL_SEEDS):
        raise ValueError('An official training run requires one of five official seeds')
    if not synthetic and config != TrainConfig():
        raise ValueError('Official training hyperparameters are frozen; overrides prohibited')
    run_dir = Path(run_dir).resolve()
    if _within_repo(run_dir):
        raise ValueError('Training outputs must be stored outside the GitHub repository')
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Nonempty run directory; refusing to overwrite: {run_dir}')
    if not all(k in fold for k in ('train_loader','val_loader','mode','sources','heldout')):
        raise ValueError('Incomplete Stage2E-1 fold dictionary')
    source_names = tuple(fold['sources'])
    if fold['heldout'] is not None and fold['heldout'] in source_names:
        raise ValueError('Held-out basin must never be a training source')
    loader = fold['train_loader']
    sampler = loader.sampler
    if not sampler.replacement or sampler.num_samples != len(loader.dataset):
        raise ValueError('Training requires official with-replacement full-length sampler')
    if sampler.generator is None:
        raise ValueError('Frozen sampler must have its own seeded generator')
    # Stage18 builds separate samplers with the same local generator seed for
    # each model. Resetting that generator permits using one frozen fold object
    # for sequential MS and FiLM fits while reproducing the intended sequence.
    sampler.generator.manual_seed(seed)

    # Crucial historical ordering: set_seed BEFORE model initialization, even
    # though Stage2C's frozen train_one_model also calls set_seed internally.
    set_seed(seed)
    model = build_frozen_model(model_name).to(torch.device(device))
    result = train_one_model(
        model_name, model, fold['train_loader'], fold['val_loader'],
        seed=seed, run_dir=run_dir, device=device, config=config,
    )
    audit = {
        'mode': fold['mode'],
        'sources': list(source_names),
        'heldout': fold['heldout'],
        'seed': seed,
        'trained_model': model_name,
        'train_samples_per_epoch': int(sampler.num_samples),
        'sampler_replacement': bool(sampler.replacement),
        'source_only_scaler_fitting': True,
        'independent_ps_training': False,
        'synthetic_test': bool(synthetic),
        'training_settings': 'smoke_only_override' if synthetic else 'frozen_formal',
        'best_epoch': result['best_epoch'],
        'best_val_loss': result['best_val_loss'],
        'n_parameters': result['n_params'],
    }
    (run_dir / 'run_metadata.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return audit


def train_authorized_single_fold(
    path_mapping: dict[str, str | Path], *, experiment: str,
    seed: int, heldout: str | None, model_name: str,
    output_dir: str | Path, device: str | torch.device = 'cpu',
) -> dict:
    """Authorize data via caller-managed mapping; train one official model/fold.

    This command should NOT be used to overwrite or reproduce already finished
    historical runs unless separately agreed. No target basin may fit scalers.
    """
    target, sources = validate_training_request(experiment, seed, heldout, model_name)
    required = tuple(LOBO_BASIN_ORDER) if experiment == 'lobo' else sources
    loaded, _ = load_authorized_basin_csvs(path_mapping, basins=required)
    if experiment == 'lobo':
        fold = build_fold_data(loaded, heldout_basin=target, seed=seed)
    else:
        fold = build_fold_data(loaded, source_basins=sources, seed=seed)
    return train_prepared_fold(fold, seed, model_name, output_dir, device=device)
