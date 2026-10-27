"""Opt-in Stage2E-3 authorized single-fold replay, never model training.

Check only:
 python examples/run_authorized_fold_replay.py --manifest G:/private/manifest.json \
   --experiment lobo --heldout Zengjiang --seed 42 \
   --ms-checkpoint G:/private/MS-TCNRS/best_model.pth \
   --film-checkpoint G:/private/Shared-FiLM/best_model.pth \
   --summary-out G:/private/results/summary.json --check

To replay only after prior review, use --execute instead of --check.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data_pipeline import (LOBO_BASIN_ORDER, ORIGINAL_SOURCE_BASINS,
                           OFFICIAL_SEEDS, canonical_basin_name)
from formal_fold_replay import replay_authorized_fold
from inference import write_summary_json


def _within_git_repo(path: Path):
    path = path.resolve()
    return any((b/'.git').exists() for b in (path,*path.parents))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--experiment',choices=['lobo','predefined3'],required=True)
    p.add_argument('--heldout',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--ms-checkpoint',type=Path,required=True)
    p.add_argument('--film-checkpoint',type=Path,required=True)
    p.add_argument('--summary-out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.seed not in OFFICIAL_SEEDS:
        p.error('Official replay only: unknown seed')
    heldout=canonical_basin_name(args.heldout)
    if args.experiment=='predefined3' and heldout in ORIGINAL_SOURCE_BASINS:
        p.error('Predefined targets may not be source basins')
    if not args.manifest.is_file():
        p.error('Authorized manifest not found')
    if _within_git_repo(args.summary_out):
        p.error('Aggregate results must be saved outside the Git repository')
    if args.summary_out.exists():
        p.error('Refusing to overwrite an existing summary file')
    if args.ms_checkpoint.resolve()==args.film_checkpoint.resolve():
        p.error('MS and FiLM checkpoints must be separate')
    for ckpt in (args.ms_checkpoint,args.film_checkpoint):
        if not ckpt.is_file():
            p.error(f'Checkpoint not found: {ckpt.name}')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    paths=manifest.get('basin_data')
    if not isinstance(paths,dict):
        p.error('Manifest must contain basin_data mapping')
    paths={canonical_basin_name(k):Path(v) for k,v in paths.items()}
    required=LOBO_BASIN_ORDER if args.experiment=='lobo' else ORIGINAL_SOURCE_BASINS+(heldout,)
    missing=[b for b in required if b not in paths or not paths[b].is_file()]
    if missing:
        p.error(f'Missing authorized input CSVs for {len(missing)} basin(s)')
    if args.device=='cuda':
        import torch
        if not torch.cuda.is_available():
            p.error('Requested CUDA device unavailable')
    print('Mode:',args.experiment,'| held-out:',heldout,'| seed:',args.seed)
    print('Authorized CSVs:',len(required),'| checkpoint files: 2')
    print('No output will overwrite existing results.')
    if args.check:
        print('[STATUS] STAGE2E3_AUTHORIZED_REPLAY_CHECK_PASS = True')
        print('No observations or checkpoint payloads have been loaded in check mode.')
        return
    result=replay_authorized_fold(paths,experiment=args.experiment,
                                  heldout_name=heldout,seed=args.seed,
                                  ms_checkpoint=args.ms_checkpoint,
                                  film_checkpoint=args.film_checkpoint,
                                  device=args.device)
    # No raw observation rows, intermediate predictions, or private paths are written.
    write_summary_json(args.summary_out,result)
    print('Aggregate valid targets:',result['metadata']['n_valid_targets'])
    for model, vals in result['metrics'].items():
        print(model, 'NSE:', vals['NSE'], 'n:', vals['n'])
    print('[STATUS] STAGE2E3_AUTHORIZED_SINGLE_FOLD_REPLAY_PASS = True')
    print('Saved aggregate summary (not hourly records):',args.summary_out)

if __name__=='__main__':
    main()
