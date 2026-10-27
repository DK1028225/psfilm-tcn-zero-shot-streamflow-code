"""Frozen model architectures and gate-compatible training adapter.

Only two independent networks are fitted per seed/fold: MS-TCN(RS) and
Shared-FiLM. PS-FiLM uses the saved Shared-FiLM weights at inference only.
The Shared-FiLM adapter accepts the historical training gate and preserves
state_dict names/shapes for replay by the public SharedFiLMTCN reference model.
"""
from __future__ import annotations

import torch
from torch import nn

from psfilm_reference import (
    Chomp1d, SharedFiLMTCN, NUM_CHANNELS, NUM_DYNAMIC, KERNEL_SIZE, DROPOUT
)

EXPECTED_MS_PARAMETERS = 239_169
EXPECTED_FILM_PARAMETERS = 281_729


class FrozenMSTemporalBlock(nn.Module):
    """Stage16/18 dynamic-only block, including the second convolution ReLU."""

    def __init__(self, n_in: int, n_out: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu_res = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_res(out + res)


class MSTCN_RS(nn.Module):
    """Stage16/18 MS-TCN(RS) architecture, matching frozen state_dict keys."""

    def __init__(self):
        super().__init__()
        blocks = []
        for i, out_ch in enumerate(NUM_CHANNELS):
            in_ch = NUM_DYNAMIC if i == 0 else NUM_CHANNELS[i - 1]
            blocks.append(FrozenMSTemporalBlock(
                in_ch, out_ch, KERNEL_SIZE, 2 ** i, DROPOUT
            ))
        self.tcn = nn.ModuleList(blocks)
        self.linear = nn.Linear(NUM_CHANNELS[-1], 1)

    def forward(self, x_dyn: torch.Tensor) -> torch.Tensor:
        out = x_dyn.transpose(1, 2)
        for block in self.tcn:
            out = block(out)
        return self.linear(out[:, :, -1]).view(-1)


class TrainingCompatibleSharedFiLM(SharedFiLMTCN):
    """Formal gate=1 training forward over the unchanged reference parameters.

    The Stage2C training core calls model(x_dyn, static, gate), where gate is
    always a vector of ones. The published SharedFiLMTCN instead interprets
    its third argument as PS source envelopes. This class resolves that
    interface distinction WITHOUT adding/reordering modules or parameters.

    Only training forward is exposed here. For PS/Raw inference, instantiate
    the original SharedFiLMTCN and load this model's checkpoint state_dict.
    """

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        if not isinstance(gate, torch.Tensor):
            raise TypeError('Formal training requires a gate tensor (not projection envelopes).')
        if gate.ndim != 1 or gate.numel() != x_dyn.shape[0]:
            raise ValueError('Gate shape must be (batch_size,)')
        if not torch.all(gate == 1):
            raise ValueError('Frozen official training uses only gate=1.0')
        x = x_dyn.transpose(1, 2)
        x = self.block0_relu(self.block0(x) + self.block0_ds(x))
        for block in self.film_blocks:
            film = block.film_gen(x_static)
            half = film.shape[1] // 2
            gamma = film[:, :half].unsqueeze(-1)
            beta = film[:, half:].unsqueeze(-1)
            c = gate.view(-1, 1, 1)
            gamma_eff = 1.0 + c * (gamma - 1.0)
            beta_eff = c * beta
            out = block.drop1(block.relu1(block.chomp1(block.conv1(x))))
            out = block.drop2(block.chomp2(block.conv2(out)))
            out = block.relu_out(gamma_eff * out + beta_eff)
            res = x if block.downsample is None else block.downsample(x)
            x = block.relu_res(out + res)
        return self.linear(x[:, :, -1]).view(-1)


def build_frozen_model(model_name: str) -> nn.Module:
    """Instantiate one frozen architecture, rejecting any unknown model label."""
    if model_name == 'MS-TCN(RS)':
        model, required = MSTCN_RS(), EXPECTED_MS_PARAMETERS
    elif model_name == 'Shared-FiLM':
        model, required = TrainingCompatibleSharedFiLM(), EXPECTED_FILM_PARAMETERS
    else:
        raise ValueError(f'Unknown trainable model {model_name!r}; PS-FiLM is inference-only')
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != required:
        raise RuntimeError(f'Frozen {model_name} parameter count changed: {actual} != {required}')
    return model
