"""Data-free training smoke test using the published training core.

Creates toy checkpoints/logs only inside a temporary directory, then removes them.
"""
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from training import TrainConfig, train_one_model


class ToyFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(6,1)

    def forward(self, x, static, gate):
        return self.layer(torch.cat((x.mean(1), static * gate[:, None]), dim=1)).view(-1)


def make_loader(n):
    gen=torch.Generator().manual_seed(1122+n)
    x=torch.randn(n,24,2,generator=gen)
    z=torch.randn(n,4,generator=gen)
    y=x.mean(dim=(1,2)) + 0.2*z[:,0]
    missing=torch.zeros(n)
    if n==12:
        missing[4:8]=1  # explicitly exercise an invalid batch
    dataset=TensorDataset(x,z,y,missing,torch.zeros(n,dtype=torch.long),
                          torch.zeros(n),torch.arange(n))
    return DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)


if __name__=='__main__':
    torch.manual_seed(112)
    model=ToyFiLM().to('cpu')
    with tempfile.TemporaryDirectory(prefix='psfilm_training_smoke_') as temp:
        result=train_one_model(
            'FiLM-TCN',model,make_loader(12),make_loader(8),
            seed=42,run_dir=temp,device='cpu',config=TrainConfig(epochs=3),
        )
        assert result['best_epoch'] in (1,2,3)
        assert (Path(temp)/'train_log.csv').is_file()
        assert (Path(temp)/'best_model.pth').is_file()
    print('Training smoke test PASS')
    print(result)
