"""Formal Stage16/Stage18 data preparation reference, extracted without rewriting.



The implementation preserves the exact frozen official function AST for preprocessing,

chronological split, source-only scalers, dataset indexing and LOBO sampler.

The source experiment sampler is extracted from Stage16.



Important: the 24-row sequence is based on stored record positions, not guaranteed

hourly-continuous timestamps. Target-hour observed precipitation is included.

Only run on observations you are authorized to access.

"""

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler



# Frozen experiment defaults, identical to formal Stage16 and Stage18.

SEQ_LEN = 24

BATCH_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



BASIN_PHYSICS = {'安墩河': [384.4, 52.08, 15.15, 0.5], '密溪水': [37.3, 14.54, 17.12, 0.39], '船塘河': [1267.32, 72.06, 10.4, 0.58], '秋香江': [1080.9, 128.16, 10.59, 0.51], '彭寨河': [43.45, 27.7, 10.42, 0.63], '增江': [1772.99, 201.5, 11.54, 0.83]}



RAIN_COLUMNS = {'安墩河': ['洋潭_雨量(mm)', '仓下_雨量(mm)', '安墩_雨量(mm)', '松坑_雨量(mm)'], '密溪水': ['横岗下_雨量(mm)', '水洋洞_雨量(mm)', '连平(曲塘)_雨量(mm)'], '船塘河': ['大水_雨量(mm)', '大湖_雨量(mm)', '忠信_雨量(mm)', '船塘_雨量(mm)'], '秋香江': ['黄洞_雨量(mm)', '紫金_雨量(mm)', '升车_雨量(mm)', '合水_雨量(mm)', '热水_雨量(mm)', '瓦溪_雨量(mm)'], '彭寨河': ['大陈屋_雨量(mm)'], '增江': ['江湾_雨量(mm)', '左潭_雨量(mm)', '连麻_雨量(mm)', '龙华_雨量(mm)', '密溪_雨量(mm)', '地派_雨量(mm)', '蓝田_雨量(mm)', '铁岗_雨量(mm)']}



THIESSEN_WEIGHTS = {'安墩河': [0.3615, 0.165, 0.2283, 0.2453], '密溪水': [0.3314, 0.3225, 0.3461], '船塘河': [0.318, 0.2433, 0.187, 0.2517], '秋香江': [0.1574, 0.0767, 0.093, 0.2229, 0.3281, 0.1219], '彭寨河': [1.0], '增江': [0.1354, 0.3969, 0.1717, 0.0908, 0.0823, 0.0384, 0.0409, 0.0435]}



ORIGINAL_SOURCE_SAMPLE_WEIGHTS = {'安墩河': 1.0, '密溪水': 0.5, '船塘河': 1.0}



def read_csv_safely(path):
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='gb18030')



def preprocess_unified_df(filepath, basin_name):
    df = read_csv_safely(filepath)
    rain_cols = RAIN_COLUMNS[basin_name]
    weights = THIESSEN_WEIGHTS[basin_name]
    if len(rain_cols) != len(weights):
        raise ValueError(f'{basin_name}雨量列与泰森权重数量不一致')
    missing_cols = [col for col in rain_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'{basin_name}缺少预期雨量列：{missing_cols}\n实际列：{df.columns.tolist()}')
    if '流量(m³/s)' not in df.columns:
        raise ValueError(f'{basin_name}缺少流量(m³/s)列')
    df[rain_cols] = df[rain_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['流域面雨量(mm)'] = sum((df[col] * w for col, w in zip(rain_cols, weights)))
    df['流量(m³/s)'] = pd.to_numeric(df['流量(m³/s)'], errors='coerce')
    df['流量_缺失指示'] = df['流量(m³/s)'].isna().astype(int)
    df['流量_原始负值指示'] = (df['流量(m³/s)'] < 0).fillna(False).astype(int)
    df['流量(m³/s)'] = df['流量(m³/s)'].fillna(0).clip(lower=0)
    area = BASIN_PHYSICS[basin_name][0]
    df['径流模数'] = df['流量(m³/s)'] / area * 1000.0
    df['Log径流模数'] = np.log1p(df['径流模数'])
    df['Log前期径流模数'] = df['Log径流模数'].shift(1).fillna(0)
    time_col = df.columns[0]
    df['时间'] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=['时间']).reset_index(drop=True)
    duplicate_count = int(df['时间'].duplicated(keep=False).sum())
    info = {'basin': basin_name, 'path': str(filepath), 'time_col': str(time_col), 'rain_cols': ' | '.join(rain_cols), 'n_rain': len(rain_cols), 'n_rows': len(df), 'flow_missing_rate': float(df['流量_缺失指示'].mean()), 'raw_negative_flow_count': int(df['流量_原始负值指示'].sum()), 'duplicate_time_rows': duplicate_count, 'start_time': str(df['时间'].min()), 'end_time': str(df['时间'].max())}
    return (df, info)



def split_source_df(df):
    n = len(df)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    return (train_df, val_df, test_df)



def fit_fold_scalers(source_train_dfs, fold_source_names):
    x_all = []
    y_all = []
    for basin in fold_source_names:
        df = source_train_dfs[basin]
        x_all.append(df[['流域面雨量(mm)', 'Log前期径流模数']].values)
        y_all.append(df['Log径流模数'].values.reshape(-1, 1))
    x_scaler = StandardScaler().fit(np.vstack(x_all))
    y_scaler = StandardScaler().fit(np.vstack(y_all))
    source_static_raw = np.array([BASIN_PHYSICS[basin] for basin in fold_source_names], dtype=np.float32)
    static_scaler = StandardScaler().fit(source_static_raw)
    source_static_scaled = static_scaler.transform(source_static_raw).astype(np.float32)
    return (x_scaler, y_scaler, static_scaler, source_static_scaled)



class PhysicsHydroDataset(Dataset):

    def __init__(self, df, basin_name, basin_id, seq_len, x_scaler, y_scaler, static_scaler):
        self.df = df.reset_index(drop=True)
        self.basin_name = basin_name
        self.basin_id = basin_id
        self.seq_len = seq_len
        self.X = x_scaler.transform(self.df[['流域面雨量(mm)', 'Log前期径流模数']].values).astype(np.float32)
        self.Y = y_scaler.transform(self.df['Log径流模数'].values.reshape(-1, 1)).flatten().astype(np.float32)
        raw_s = np.array(BASIN_PHYSICS[basin_name], dtype=np.float32).reshape(1, -1)
        self.static = static_scaler.transform(raw_s).flatten().astype(np.float32)
        self.M = self.df['流量_缺失指示'].values.astype(np.float32)
        self.Q = self.df['流量(m³/s)'].values.astype(np.float32)
        self.T = self.df['时间'].values

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        target_idx = idx + self.seq_len - 1
        previous_idx = target_idx - 1
        x = torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float32)
        s = torch.tensor(self.static, dtype=torch.float32)
        y = torch.tensor(self.Y[target_idx], dtype=torch.float32)
        m = torch.tensor(self.M[target_idx], dtype=torch.float32)
        bid = torch.tensor(self.basin_id, dtype=torch.long)
        q_prev = torch.tensor(self.Q[previous_idx], dtype=torch.float32)
        t = str(pd.to_datetime(self.T[target_idx]))
        return (x, s, y, m, bid, q_prev, t)



def build_train_val_loaders(source_train_dfs, source_val_dfs, fold_source_names, source_basin_ids, x_scaler, y_scaler, static_scaler, seed):
    train_datasets = []
    val_datasets = []
    sample_weights = []
    balance_rows = []
    for basin in fold_source_names:
        bid = source_basin_ids[basin]
        train_ds = PhysicsHydroDataset(df=source_train_dfs[basin], basin_name=basin, basin_id=bid, seq_len=SEQ_LEN, x_scaler=x_scaler, y_scaler=y_scaler, static_scaler=static_scaler)
        val_ds = PhysicsHydroDataset(df=source_val_dfs[basin], basin_name=basin, basin_id=bid, seq_len=SEQ_LEN, x_scaler=x_scaler, y_scaler=y_scaler, static_scaler=static_scaler)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        w_per_sample = 1.0 / max(len(train_ds), 1)
        sample_weights.extend([w_per_sample] * len(train_ds))
        balance_rows.append({'basin': basin, 'train_samples': len(train_ds), 'per_sample_sampler_weight': w_per_sample, 'total_basin_sampler_weight': w_per_sample * len(train_ds)})
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(train_dataset), replacement=True, generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    return (train_loader, val_loader, pd.DataFrame(balance_rows))



def build_main3_train_val_loaders(source_train_dfs, source_val_dfs, source_names, source_basin_ids, x_scaler, y_scaler, static_scaler, seed):
    """
    保持原论文的逐样本采样权重：
      安墩河1.0、密溪水0.5、船塘河1.0

    所有模型共用完全相同的sampler规则。
    """
    train_datasets = []
    val_datasets = []
    sample_weights = []
    audit_rows = []
    weighted_mass_total = 0.0
    temp_rows = []
    for basin in source_names:
        bid = source_basin_ids[basin]
        train_ds = PhysicsHydroDataset(df=source_train_dfs[basin], basin_name=basin, basin_id=bid, seq_len=SEQ_LEN, x_scaler=x_scaler, y_scaler=y_scaler, static_scaler=static_scaler)
        val_ds = PhysicsHydroDataset(df=source_val_dfs[basin], basin_name=basin, basin_id=bid, seq_len=SEQ_LEN, x_scaler=x_scaler, y_scaler=y_scaler, static_scaler=static_scaler)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        nominal_w = float(ORIGINAL_SOURCE_SAMPLE_WEIGHTS[basin])
        sample_weights.extend([nominal_w] * len(train_ds))
        mass = nominal_w * len(train_ds)
        weighted_mass_total += mass
        temp_rows.append({'basin': basin, 'train_samples': len(train_ds), 'val_samples': len(val_ds), 'per_sample_weight': nominal_w, 'weighted_sampling_mass': mass})
    for row in temp_rows:
        row = dict(row)
        row['expected_sampling_probability'] = row['weighted_sampling_mass'] / max(weighted_mass_total, 1e-12)
        audit_rows.append(row)
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(train_dataset), replacement=True, generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    return (train_loader, val_loader, pd.DataFrame(audit_rows))




def collect_heldout_base_arrays(heldout_name, heldout_df, x_scaler, y_scaler, static_scaler):
    ds = PhysicsHydroDataset(df=heldout_df, basin_name=heldout_name, basin_id=0, seq_len=SEQ_LEN, x_scaler=x_scaler, y_scaler=y_scaler, static_scaler=static_scaler)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    return (ds, loader)

# The following are safe orchestration helpers, separate from frozen official functions.
BASIN_ALIASES = {
    "Andunhe": "安墩河", "Mixishui": "密溪水", "Chuantanghe": "船塘河",
    "Qiuxiangjiang": "秋香江", "Pengzhaihe": "彭寨河", "Zengjiang": "增江",
}
OFFICIAL_SEEDS = (42, 123, 2024, 2025, 3407)
DEVELOPMENT_SEEDS = (17, 731, 4099)
ORIGINAL_SOURCE_BASINS = ("安墩河", "密溪水", "船塘河")
FORMAL_TARGET_BASINS = ("秋香江", "彭寨河", "增江")
LOBO_BASIN_ORDER = tuple(BASIN_PHYSICS)


def canonical_basin_name(name):
    """Resolve an English alias or the original data label, without altering columns."""
    if name in BASIN_PHYSICS:
        return name
    if name in BASIN_ALIASES:
        return BASIN_ALIASES[name]
    raise KeyError(f"Unknown basin: {name!r}")


def load_authorized_basin_csvs(path_mapping, basins=None):
    """Read only caller-provided authorized CSV paths, preserving input record order.

    Parameters
    ----------
    path_mapping : dict[str, str | Path]
        Mapping from Chinese basin names or English aliases to local CSV file paths.
    basins : sequence, optional
        Selected basins. Defaults to the six-basin formal protocol.

    Returns
    -------
    data : dict[str, pd.DataFrame]
    audits : pd.DataFrame
        Data provenance summaries only. The input observations are not exported.
    """
    normalized = {canonical_basin_name(k): Path(v) for k, v in path_mapping.items()}
    selected = LOBO_BASIN_ORDER if basins is None else [canonical_basin_name(b) for b in basins]
    result = {}
    audits = []
    for basin in selected:
        if basin not in normalized:
            raise KeyError(f"No authorized input CSV was specified for {basin}")
        path = normalized[basin]
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found for basin {basin}")
        df, audit = preprocess_unified_df(path, basin)
        result[basin] = df
        audit = dict(audit)
        # Avoid printing/writing private absolute paths in publication materials.
        audit.pop('path', None)
        audits.append(audit)
    return result, pd.DataFrame(audits)


def build_fold_data(data, heldout_basin=None, *, source_basins=None, seed=42):
    """Construct an official source-only fold without training or target fitting.

    For LOBO, set `heldout_basin` and leave `source_basins=None`.
    For the predefined 3-to-3 experiment, set `source_basins=ORIGINAL_SOURCE_BASINS`
    and leave `heldout_basin=None`.

    DataFrames must already be processed by `preprocess_unified_df`.
    Returns a dictionary of the unmodified official datasets, loaders, scalers,
    train/validation partitions and sampler audit table.
    """
    if source_basins is None:
        if heldout_basin is None:
            raise ValueError("LOBO requires heldout_basin")
        heldout = canonical_basin_name(heldout_basin)
        names = [b for b in LOBO_BASIN_ORDER if b != heldout]
        mode = 'lobo'
    else:
        names = [canonical_basin_name(b) for b in source_basins]
        if tuple(names) != ORIGINAL_SOURCE_BASINS:
            raise ValueError("The predefined 3-source protocol requires the frozen source order")
        heldout = None if heldout_basin is None else canonical_basin_name(heldout_basin)
        if heldout in names:
            raise ValueError("Target basin cannot appear among source basins")
        mode = 'predefined3'
    missing = [b for b in names if b not in data]
    if missing:
        raise KeyError(f"Missing source-basin data: {missing}")
    # Preserve the frozen global identifiers and train/validation segment lengths.
    source_ids = {b: i for i,b in enumerate(names)}
    split = {b: split_source_df(data[b]) for b in names}
    trains = {b: split[b][0] for b in names}
    vals = {b: split[b][1] for b in names}
    x, y, static, source_static = fit_fold_scalers(trains, names)
    if mode == 'lobo':
        train_loader, val_loader, sampler_audit = build_train_val_loaders(
            trains, vals, names, source_ids, x, y, static, seed)
    else:
        train_loader, val_loader, sampler_audit = build_main3_train_val_loaders(
            trains, vals, names, source_ids, x, y, static, seed)
    if heldout is not None:
        if heldout not in data:
            raise KeyError(f"Missing authorized held-out basin: {heldout}")
        heldout_dataset, heldout_loader = collect_heldout_base_arrays(
            heldout, data[heldout], x, y, static
        )
    else:
        heldout_dataset, heldout_loader = None, None
    return {
        'heldout_dataset': heldout_dataset, 'heldout_loader': heldout_loader,
        'mode':mode, 'sources':tuple(names), 'heldout':heldout,
        'train_dfs':trains, 'val_dfs':vals, 'source_ids':source_ids,
        'x_scaler':x, 'y_scaler':y, 'static_scaler':static,
        'source_static_scaled':source_static,
        'train_loader':train_loader, 'val_loader':val_loader,
        'sampling_audit':sampler_audit,
    }
