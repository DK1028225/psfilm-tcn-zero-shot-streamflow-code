"""Replay one AUTHORIZED, prepared held-out fold with frozen FiLM checkpoint.

Only aggregate metric JSON is written. Restricted arrays are NOT published.

Example:
  python examples/replay_prepared_fold.py --prepared-input G:/private/fold.npz \\
      --checkpoint G:/private/Shared-FiLM/best_model.pth \\
      --summary-out G:/private/outputs/fold_summary.json
"""
import argparse
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from psfilm_reference import SharedFiLMTCN
from inference import load_frozen_film_checkpoint,replay_raw_ps_pair,write_summary_json
from prepared_fold import load_prepared_fold


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared-input',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--summary-out',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=ap.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():
        ap.error('CUDA device requested but not available')
    loader, y_scaler, area, source_z, target_z, heldout = load_prepared_fold(a.prepared_input)
    model,provenance=load_frozen_film_checkpoint(SharedFiLMTCN(),a.checkpoint,device=a.device)
    summary, _arrays = replay_raw_ps_pair(model,loader,y_scaler,area,source_z,target_z,device=a.device)
    report={'heldout':heldout,'checkpoint':provenance,'metrics':summary,
            'notice':'Only aggregate metrics were exported; hourly arrays remain private.'}
    write_summary_json(a.summary_out,report)
    print('Prepared held-out replay PASS')
    print('Basin:',heldout,'| seed:',provenance.get('seed'))
    print('Valid records:',summary['FiLM-TCN']['n'])
    print('Raw NSE:',summary['FiLM-TCN']['NSE'])
    print('PS NSE:',summary['PS-FiLM-TCN']['NSE'])
    print('Aggregate summary saved:',a.summary_out)


if __name__=='__main__':
    main()
