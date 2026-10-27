"""One-command private-adapter smoke test, using temporary SYNTHETIC inputs only."""
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from psfilm_reference import SharedFiLMTCN


def main():
    rng=np.random.default_rng(42)
    n=135
    q=20+10*np.sin(np.arange(n+24)/12.)
    dynamic=rng.normal(size=(n,24,2)).astype(np.float32)
    target_scaled=rng.normal(size=n).astype(np.float32)
    mask=np.zeros(n,dtype=np.float32);mask[::29]=1
    dates=np.asarray([str(i) for i in __import__('pandas').date_range('2022-05-01',periods=n+24,freq='h')][23:n+23])
    sources=rng.normal(size=(5,4)).astype(np.float32)
    target=rng.normal(size=4).astype(np.float32)
    ysc=StandardScaler().fit(np.linspace(1.0,5.0,100).reshape(-1,1))
    torch.manual_seed(123)
    model=SharedFiLMTCN().eval()
    with TemporaryDirectory(prefix='stage2d_adapter_') as tmp:
        base=Path(tmp)
        f=base/'prepared_synthetic.npz'
        ckpt=base/'checkpoint_synthetic.pth'
        out=base/'summary.json'
        np.savez_compressed(f,dynamic_x=dynamic,target_scaled=target_scaled,missing_mask=mask,
                            q_previous=q[22:n+22].astype(np.float32),target_dates=dates,
                            q_full=q,source_static_scaled=sources,target_static_scaled=target,
                            y_scaler_mean=ysc.mean_,y_scaler_scale=ysc.scale_,
                            basin_area_km2=np.array([100.]),heldout_name='SYNTHETIC',
                            source_names=np.array(['A','B','C','D','E']))
        torch.save({'model_name':'Shared-FiLM','state_dict':model.state_dict(),
                    'seed':123,'best_epoch':1},ckpt)
        subprocess.run([sys.executable,str(ROOT/'examples'/'replay_prepared_fold.py'),
                        '--prepared-input',str(f),'--checkpoint',str(ckpt),
                        '--summary-out',str(out)],check=True)
        import json
        result=json.loads(out.read_text(encoding='utf-8'))
        assert result['metrics']['FiLM-TCN']['n']==int(np.sum(mask==0))
        assert result['metrics']['PS-FiLM-TCN']['n']==int(np.sum(mask==0))
    print('Stage2D prepared-fold CLI smoke test PASS')
    print('All synthetic private files and checkpoint were cleaned up.')


if __name__=='__main__':
    main()
