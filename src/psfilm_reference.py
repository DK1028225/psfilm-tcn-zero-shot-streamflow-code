"""Minimal reference implementation for PS-FiLM-TCN methodology.

This file contains no study observations or trained checkpoints.
It is intended to make the computational intervention reproducible.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import WeightedRandomSampler

SEQ_LEN = 24
NUM_DYNAMIC = 2
NUM_STATIC = 4
NUM_CHANNELS = [64, 128, 128, 64]
KERNEL_SIZE = 3
DROPOUT = 0.20


def runoff_transform(q_m3s: np.ndarray, area_km2: float) -> np.ndarray:
    q = np.maximum(np.asarray(q_m3s, dtype=float), 0.0)
    runoff_modulus = 1000.0 * q / float(area_km2)
    return np.log1p(runoff_modulus)


def make_lagged_state(log_runoff: np.ndarray) -> np.ndarray:
    """One-record lag constructed before partitioning; initial placeholder is numeric zero."""
    x = np.asarray(log_runoff, dtype=float)
    out = np.zeros_like(x)
    out[1:] = x[:-1]
    return out


def build_record_position_sequences(precip, lagged_state, target, seq_len: int = SEQ_LEN):
    """Match the manuscript's stored-record indexing.

    Candidate i uses rows i..i+seq_len-1 as dynamic input and row i+seq_len-1 as target.
    Because lagged_state was shifted by one record, discharge states correspond to one stored
    record earlier than the precipitation row. The last stored row is not used as a target under
    the original Dataset length convention len(df)-seq_len.
    """
    p = np.asarray(precip, dtype=float)
    s = np.asarray(lagged_state, dtype=float)
    y = np.asarray(target, dtype=float)
    if not (len(p) == len(s) == len(y)):
        raise ValueError("Input arrays must have equal length")
    n = len(p) - seq_len
    X, Y = [], []
    for i in range(n):
        t = i + seq_len - 1
        X.append(np.column_stack([p[i:i+seq_len], s[i:i+seq_len]]))
        Y.append(y[t])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def fit_source_scalers(source_dynamic_arrays, source_targets, source_static):
    x_scaler = StandardScaler().fit(np.vstack(source_dynamic_arrays))
    y_scaler = StandardScaler().fit(np.concatenate(source_targets).reshape(-1, 1))
    static_scaler = StandardScaler().fit(np.asarray(source_static, dtype=float))
    return x_scaler, y_scaler, static_scaler


class Chomp1d(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
    def forward(self, x):
        return x[:, :, :-self.n].contiguous() if self.n > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.drop2 = nn.Dropout(dropout)
        self.relu_out = nn.ReLU()
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu_res = nn.ReLU()

    def temporal_features(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return out, res

    def forward(self, x):
        out, res = self.temporal_features(x)
        return self.relu_res(self.relu_out(out) + res)


class FiLMBlock(TemporalBlock):
    def __init__(self, n_in, n_out, n_static=NUM_STATIC, kernel_size=KERNEL_SIZE, dilation=1, dropout=DROPOUT):
        super().__init__(n_in, n_out, kernel_size, dilation, dropout)
        self.film_gen = nn.Sequential(nn.Linear(n_static, 64), nn.ReLU(), nn.Linear(64, 2*n_out))
        nn.init.zeros_(self.film_gen[-1].weight)
        with torch.no_grad():
            self.film_gen[-1].bias.copy_(torch.cat([torch.ones(n_out), torch.zeros(n_out)]))

    def coefficients(self, static):
        film = self.film_gen(static)
        h = film.shape[1] // 2
        return film[:, :h], film[:, h:]

    def forward(self, x, static, bounds=None):
        gamma, beta = self.coefficients(static)
        if bounds is not None:
            gamma = torch.maximum(torch.minimum(gamma, bounds['gamma_max']), bounds['gamma_min'])
            beta = torch.maximum(torch.minimum(beta, bounds['beta_max']), bounds['beta_min'])
        out, res = self.temporal_features(x)
        out = self.relu_out(gamma.unsqueeze(-1) * out + beta.unsqueeze(-1))
        return self.relu_res(out + res)


class SharedFiLMTCN(nn.Module):
    def __init__(self):
        super().__init__()
        pad0 = KERNEL_SIZE - 1
        self.block0 = nn.Sequential(
            nn.Conv1d(NUM_DYNAMIC, NUM_CHANNELS[0], KERNEL_SIZE, padding=pad0),
            Chomp1d(pad0), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Conv1d(NUM_CHANNELS[0], NUM_CHANNELS[0], KERNEL_SIZE, padding=pad0),
            Chomp1d(pad0), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.block0_ds = nn.Conv1d(NUM_DYNAMIC, NUM_CHANNELS[0], 1)
        self.block0_relu = nn.ReLU()
        self.film_blocks = nn.ModuleList([
            FiLMBlock(NUM_CHANNELS[i-1], NUM_CHANNELS[i], dilation=2**i)
            for i in range(1, len(NUM_CHANNELS))
        ])
        self.linear = nn.Linear(NUM_CHANNELS[-1], 1)

    def forward(self, x_dyn, x_static, envelopes=None):
        x = x_dyn.transpose(1, 2)
        x = self.block0_relu(self.block0(x) + self.block0_ds(x))
        for i, block in enumerate(self.film_blocks):
            bounds = None if envelopes is None else envelopes[i]
            x = block(x, x_static, bounds=bounds)
        return self.linear(x[:, :, -1]).view(-1)


def build_source_envelopes(model: SharedFiLMTCN, source_static_scaled: np.ndarray):
    z = torch.as_tensor(source_static_scaled, dtype=torch.float32)
    envelopes = []
    model.eval()
    with torch.no_grad():
        for block in model.film_blocks:
            gamma, beta = block.coefficients(z)
            envelopes.append({
                'gamma_min': gamma.min(0).values.view(1, -1),
                'gamma_max': gamma.max(0).values.view(1, -1),
                'beta_min': beta.min(0).values.view(1, -1),
                'beta_max': beta.max(0).values.view(1, -1),
            })
    return envelopes


def project_coefficients(gamma, beta, envelope):
    g = np.clip(np.asarray(gamma), np.asarray(envelope['gamma_min']), np.asarray(envelope['gamma_max']))
    b = np.clip(np.asarray(beta), np.asarray(envelope['beta_min']), np.asarray(envelope['beta_max']))
    return g, b


def project_to_convex_hull(points, target, tol=1e-9):
    """Exact small-source Euclidean convex-hull projection used by Stage23D v2."""
    P = np.asarray(points, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64).reshape(-1)
    if P.ndim != 2 or P.shape[1] != len(x):
        raise ValueError('Incompatible convex-hull dimensions')
    n = P.shape[0]
    best = None
    for k in range(1, n+1):
        for subset in combinations(range(n), k):
            Q = P[list(subset)]
            if k == 1:
                w = np.array([1.0], dtype=np.float64)
            else:
                q_last = Q[-1]
                D = (Q[:-1] - q_last).T
                u, _, _, _ = np.linalg.lstsq(D, x - q_last, rcond=None)
                w = np.concatenate([np.asarray(u, dtype=np.float64), [1.0 - np.sum(u)]])
            if np.min(w) < -tol:
                continue
            w = np.where(w < 0.0, 0.0, w)
            if w.sum() <= 0:
                continue
            w /= w.sum()
            projected = w @ Q
            d2 = float(np.dot(projected-x, projected-x))
            if best is None or d2 < best['d2'] - 1e-14:
                full = np.zeros(n, dtype=np.float64)
                full[list(subset)] = w
                best = {'d2': d2, 'weights': full, 'projected': projected, 'active': subset}
    if best is None:
        raise RuntimeError('No feasible convex-hull face found')
    d = float(np.sqrt(best['d2']))
    return {
        'weights': best['weights'], 'projected': best['projected'], 'distance': d,
        'normalized_distance': d / max(float(np.linalg.norm(x)), 1.0),
        'active_indices': best['active'],
        'sum_weight_error': abs(float(best['weights'].sum()) - 1.0),
        'min_weight': float(best['weights'].min()),
    }


def metrics(obs, pred):
    """Compute metrics exactly as in the formal Stage18 evaluation.

    A sample is evaluable only when observation and prediction are finite.
    Metrics are undefined for fewer than ten valid pairs. KGE uses
    population standard deviations (NumPy's default, ddof=0).
    PBIAS is positive for overall overprediction.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    n = int(len(obs))

    if n < 10:
        return {
            "NSE": np.nan, "KGE": np.nan, "RMSE": np.nan,
            "MAE": np.nan, "PBIAS": np.nan, "n": n,
        }

    eps = 1e-8
    nse = 1.0 - np.sum((pred - obs) ** 2) / (
        np.sum((obs - np.mean(obs)) ** 2) + eps
    )
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    mae = np.mean(np.abs(pred - obs))

    if np.std(obs) < eps or np.std(pred) < eps:
        r = np.nan
    else:
        r = np.corrcoef(obs, pred)[0, 1]

    alpha = np.std(pred) / (np.std(obs) + eps)
    beta = np.mean(pred) / (np.mean(obs) + eps)
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) if np.isfinite(r) else np.nan
    pbias = 100.0 * np.sum(pred - obs) / (np.sum(obs) + eps)

    return {
        "NSE": float(nse), "KGE": float(kge),
        "RMSE": float(rmse), "MAE": float(mae),
        "PBIAS": float(pbias), "n": n,
    }


def basin_balanced_weights(lengths):
    """LOBO sampler: each basin receives equal total sampling mass."""
    w = []
    for n in lengths:
        w.extend([1.0/max(int(n), 1)] * int(n))
    return torch.tensor(w, dtype=torch.double)



def build_lobo_weighted_sampler(lengths, seed: int):
    """Formal LOBO sampler: equal total source-basin mass, with replacement.

    Each epoch draws exactly the number of samples in the concatenated source-training dataset.
    """
    weights = basin_balanced_weights(lengths)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(sum(int(n) for n in lengths)),
        replacement=True,
        generator=g,
    )


def build_predefined_weighted_sampler(lengths, basin_weights, seed: int):
    """Predefined three-source sampler using fixed per-sample weights 1.0/0.5/1.0."""
    if len(lengths) != len(basin_weights):
        raise ValueError("lengths and basin_weights must have equal length")
    w = []
    for n, bw in zip(lengths, basin_weights):
        w.extend([float(bw)] * int(n))
    g = torch.Generator()
    g.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.tensor(w, dtype=torch.double),
        num_samples=int(sum(int(n) for n in lengths)),
        replacement=True,
        generator=g,
    )


def summarize_seed_values(values):
    """Across-seed summaries: sample SD (ddof=1), matching manuscript reporting."""
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "sample_sd": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }

def smoke_test():
    rng = np.random.default_rng(42)
    q = np.maximum(rng.normal(10, 3, 100), 0)
    p = np.maximum(rng.normal(1, 2, 100), 0)
    y = runoff_transform(q, 100.0)
    lag = make_lagged_state(y)
    X, Y = build_record_position_sequences(p, lag, y)
    assert X.shape == (100-SEQ_LEN, SEQ_LEN, 2)

    model = SharedFiLMTCN().eval()
    static_sources = rng.normal(size=(5, 4)).astype(np.float32)
    env = build_source_envelopes(model, static_sources)
    with torch.no_grad():
        out = model(torch.as_tensor(X[:4]), torch.as_tensor(np.repeat(static_sources[:1], 4, axis=0)), env)
    assert out.shape == (4,)

    hull = project_to_convex_hull(static_sources, rng.normal(size=4))
    assert abs(hull['weights'].sum()-1.0) <= 1e-8
    assert hull['min_weight'] >= -1e-10
    sampler = build_lobo_weighted_sampler([10, 20, 30], seed=42)
    assert sampler.replacement is True and sampler.num_samples == 60
    s = summarize_seed_values([1.0, 2.0, 3.0])
    assert abs(s['sample_sd'] - 1.0) < 1e-12
    print('Smoke test PASS')
    print('Example metrics:', metrics(q[1:50], q[1:50] + rng.normal(0, 0.5, 49)))


if __name__ == '__main__':
    smoke_test()
