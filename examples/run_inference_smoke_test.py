"""Run frozen Raw/PS-FiLM checkpoint replay on synthetic, non-restricted data."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from psfilm_reference import SharedFiLMTCN
from inference import replay_raw_ps_pair, load_frozen_film_checkpoint, write_summary_json


class SyntheticHeldout(Dataset):
    def __init__(self, n=235):
        g = np.random.default_rng(2024)
        self.Q = 10.0 + 3.0*np.sin(np.arange(n+24)/13.0)
        self.rain = np.maximum(g.normal(0.7, 0.4, n+24), 0)
        self.static = np.array([2., .3, -1., 1.], dtype=np.float32)
        self.mask = np.zeros(n, dtype=np.float32)
        self.mask[::61] = 1
        self.x = g.normal(size=(n,24,2)).astype(np.float32)
        self.y = np.zeros(n, dtype=np.float32)
        self.dates = [str(t) for t in __import__('pandas').date_range('2022-01-01', periods=n+24, freq='h')]

    def __len__(self):
        return len(self.Q)-24

    def __getitem__(self, i):
        t = i+23
        return (torch.from_numpy(self.x[i]), torch.from_numpy(self.static),
                torch.tensor(self.y[i]), torch.tensor(self.mask[i]),
                torch.tensor(0), torch.tensor(self.Q[t-1], dtype=torch.float32),
                self.dates[t])


def main():
    ds = SyntheticHeldout()
    loader = DataLoader(ds, batch_size=50, shuffle=False)
    torch.manual_seed(3407)
    initial_model = SharedFiLMTCN().cpu().eval()
    # Activate projection in the synthetic example with nontrivial generator coefficients.
    with torch.no_grad():
        for b in initial_model.film_blocks:
            b.film_gen[-1].weight.normal_(mean=0, std=.02)
    source_static = np.array([[0.,0.,0.,0.],[.3,.2,-.5,.2],[-.2,.4,.4,-.4],
                              [.1,-.3,.3,.5],[-.4,-.2,-.2,-.1]],dtype=np.float32)
    target_static = np.array([2.,.3,-1.,1.],dtype=np.float32)
    y_scaler = StandardScaler().fit(np.linspace(0.0,8.0,400).reshape(-1,1))
    with TemporaryDirectory(prefix='psfilm_inference_') as tmp:
        chk=Path(tmp)/'toy_checkpoint.pth'
        torch.save({'model_name':'Shared-FiLM','seed':3407,'best_epoch':1,
                    'state_dict': initial_model.state_dict()},chk)
        loaded_model, provenance = load_frozen_film_checkpoint(SharedFiLMTCN(),chk)
        before = {k:v.clone() for k,v in loaded_model.state_dict().items()}
        summary, pred = replay_raw_ps_pair(loaded_model, loader, y_scaler, area_km2=100.0,
                                           source_static_scaled=source_static,
                                           target_static_scaled=target_static)
        assert np.array_equal(pred['FiLM-TCN']['dates'],pred['PS-FiLM-TCN']['dates'])
        assert np.array_equal(pred['FiLM-TCN']['obs_q'],pred['PS-FiLM-TCN']['obs_q'])
        assert summary['FiLM-TCN']['n'] == int((ds.mask==0).sum())
        assert summary['PS-FiLM-TCN']['n'] == summary['FiLM-TCN']['n']
        assert all(torch.equal(before[k], v) for k,v in loaded_model.state_dict().items())
        write_summary_json(Path(tmp)/'summary.json',{'provenance':provenance, 'metrics':summary})
        assert (Path(tmp)/'summary.json').is_file()
    print('Stage2D inference smoke test PASS')
    print('Shared checkpoint and held-out targets aligned; no model fitting performed.')
    print('Synthetic valid target count:', summary['FiLM-TCN']['n'])
    print('Raw synthetic NSE:', round(summary['FiLM-TCN']['NSE'],6))
    print('PS synthetic NSE:',round(summary['PS-FiLM-TCN']['NSE'],6))


if __name__ == '__main__':
    main()
