"""Public unit tests for formal training semantics, using synthetic tensors."""
import sys
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from training import TrainConfig, masked_mse_loss, train_one_model


class ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(torch.tensor(.7))

    def forward(self,x,s,gate):
        return torch.zeros(x.shape[0])+self.bias*0.


def all_missing_loader():
    n=8
    ds=TensorDataset(torch.randn(n,24,2),torch.randn(n,4),torch.zeros(n),
                     torch.ones(n),torch.zeros(n,dtype=torch.long),
                     torch.zeros(n),torch.arange(n))
    return DataLoader(ds,batch_size=4,shuffle=False)


class TrainingTests(unittest.TestCase):
    def test_masked_loss(self):
        pred=torch.tensor([1.,3.,10.],requires_grad=True)
        target=torch.tensor([2.,4.,0.])
        mask=torch.tensor([0.,0.,1.])
        loss=masked_mse_loss(pred,target,mask)
        self.assertAlmostEqual(loss.item(),1.)

    def test_all_invalid_still_backward(self):
        pred=torch.tensor([1.,3.],requires_grad=True)
        loss=masked_mse_loss(pred,torch.zeros(2),torch.ones(2))
        self.assertEqual(loss.item(),0.)
        loss.backward()
        torch.testing.assert_close(pred.grad,torch.zeros_like(pred.grad))

    def test_patience_and_synthetic_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader=all_missing_loader()
            out=train_one_model('FiLM-TCN',ConstantModel(),loader,loader,42,tmp,
                                device='cpu',config=TrainConfig(epochs=20,early_stopping_patience=15))
            self.assertEqual(out['best_epoch'],1)
            import pandas as pd
            history=pd.read_csv(Path(tmp)/'train_log.csv')
            self.assertEqual(len(history),16)
            self.assertTrue((Path(tmp)/'best_model.pth').exists())
            self.assertLess(history['lr'].iloc[-1],5e-4)


if __name__=='__main__':
    unittest.main()
