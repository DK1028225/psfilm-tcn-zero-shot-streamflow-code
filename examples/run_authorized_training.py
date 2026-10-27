"""Opt-in, single-fold formal training CLI; no real data is bundled.

Example: run with --check first. --execute requires explicit authorization,
uses the frozen training settings, and refuses checkpoint overwrites.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment_runner import validate_training_request, train_authorized_single_fold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--experiment', required=True, choices=['lobo','predefined3'])
    p.add_argument('--heldout', default=None)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--model', required=True, choices=['MS-TCN(RS)','Shared-FiLM'])
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Only inspect requested run and file presence')
    group.add_argument('--execute', action='store_true', help='Start actual official-parameter model training')
    args=p.parse_args()
    if not args.manifest.is_file():
        raise SystemExit('Authorized manifest not found')
    config=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    mapping=config.get('basin_data')
    if not isinstance(mapping,dict):
        raise SystemExit('Expected a JSON object containing basin_data')
    target,sources=validate_training_request(args.experiment,args.seed,args.heldout,args.model)
    import data_pipeline as dp
    needed = dp.LOBO_BASIN_ORDER if args.experiment=='lobo' else sources
    paths={dp.canonical_basin_name(k):Path(v) for k,v in mapping.items()}
    missing=[b for b in needed if b not in paths or not paths[b].is_file()]
    if missing:
        raise SystemExit(f'Authorized CSV paths are missing for {len(missing)} required basins')
    if args.output_dir.resolve() == ROOT.resolve() or ROOT.resolve() in args.output_dir.resolve().parents:
        raise SystemExit('Output must be located outside the public code repository')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Output directory is nonempty; refusing to overwrite.')
    print('Experiment:',args.experiment)
    print('Held-out:',target)
    print('Source basins:',len(sources))
    print('Seed:',args.seed)
    print('Independent network:',args.model)
    if args.check:
        print('[STATUS] STAGE2E2_AUTHORIZED_INPUT_CHECK_PASS = True')
        return
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('CUDA was requested but is unavailable')
    result=train_authorized_single_fold(
        mapping,experiment=args.experiment,seed=args.seed,heldout=target,
        model_name=args.model,output_dir=args.output_dir,device=args.device
    )
    print('Finished official-parameter training. Best epoch:',result['best_epoch'])


if __name__=='__main__':
    main()
