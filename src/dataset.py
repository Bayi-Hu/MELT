import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PKL = os.path.join(PROJECT_ROOT, "data", "feat", "feature.pkl")
LABEL_CSV = os.path.join(PROJECT_ROOT, "data", "label", "label.csv")

TS_KEYS = ("ts", "ts_len")
LABEL_COLS = ["mint_address", "label", "min_ratio", "return_ratio", "manipulated"]


class TSDataset(Dataset):
    def __init__(self, X, lengths, y):
        self.X = X
        self.lengths = lengths
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.tensor(int(self.lengths[idx]), dtype=torch.long),
            torch.tensor(float(self.y[idx]), dtype=torch.float32),
        )


def ts_collate(batch):
    xs, lens, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(lens, 0), torch.stack(ys, 0)


def _load_feat_list():
    with open(FEATURE_PKL, "rb") as f:
        return pkl.load(f)


def _load_labels():
    df = pd.read_csv(LABEL_CSV, usecols=LABEL_COLS)
    return df


def _build_tabular_and_ts(feat_list):
    tab_rows = []
    ts_list = []
    ts_len_list = []
    for item in feat_list:
        tab_rows.append({k: v for k, v in item.items() if k not in TS_KEYS})
        ts_list.append(item["ts"])
        ts_len_list.append(item["ts_len"])
    tab_df = pd.DataFrame(tab_rows)
    ts_arr = np.stack(ts_list, axis=0).astype(np.float32)
    ts_len = np.asarray(ts_len_list, dtype=np.int32)
    return tab_df, ts_arr, ts_len


def _normalize_ts(ts_train, ts_test):
    """Price channels (0..3) share a single mean/std; volume (4) uses StandardScaler.
    Stats are fit on the train split only."""
    ts_train = ts_train.copy()
    ts_test = ts_test.copy()

    price_train = ts_train[:, :, :4]
    price_mean = price_train.mean()
    price_std = price_train.std() + 1e-8
    ts_train[:, :, :4] = (ts_train[:, :, :4] - price_mean) / price_std
    ts_test[:, :, :4] = (ts_test[:, :, :4] - price_mean) / price_std

    vol_scaler = StandardScaler()
    N_tr, T, _ = ts_train.shape
    N_te = ts_test.shape[0]
    vol_train = ts_train[:, :, 4].reshape(-1, 1)
    vol_test = ts_test[:, :, 4].reshape(-1, 1)
    vol_scaler.fit(vol_train)
    ts_train[:, :, 4] = vol_scaler.transform(vol_train).reshape(N_tr, T)
    ts_test[:, :, 4] = vol_scaler.transform(vol_test).reshape(N_te, T)

    return ts_train.astype(np.float32), ts_test.astype(np.float32)


def load_dataset(
    filter_valid: bool = True,
    train_ratio: float = 0.7,
    shuffle_seed: int = 42,
    scale_tabular: bool = True,
    scale_ts: bool = True,
):
    """
    Load tabular features, time-series features, and labels; merge by mint_address;
    chronologically split (train ratio first, test ratio last); shuffle train portion.

    Returns dict with:
        X_train_tab, X_test_tab : (N, F) float32 standardized tabular features
        ts_train, ts_test       : (N, T, C) float32 time-series tensors
        ts_len_train, ts_len_test : (N,) int32 valid sequence lengths
        y_train, y_test         : (N,) int64 labels (0 if "high", 1 otherwise)
        mint_train, mint_test   : (N,) mint_address arrays
        mint2label_info         : dict {mint -> {label, min_ratio, return_ratio}}
        feature_cols            : list of tabular feature column names
    """
    feat_list = _load_feat_list()
    tab_df, ts_arr, ts_len = _build_tabular_and_ts(feat_list)

    labels_df = _load_labels()
    merged = tab_df.merge(labels_df, on="mint_address", how="inner")

    addr_to_idx = {addr: i for i, addr in enumerate(tab_df["mint_address"].values)}
    align_idx = merged["mint_address"].map(addr_to_idx).values.astype(np.int64)
    ts_arr = ts_arr[align_idx]
    ts_len = ts_len[align_idx]

    if filter_valid:
        mask = (merged["group3_time_span_valid"] >= 60) & (merged["group3_holder_num"] >= 100)
        mask_np = mask.values
        merged = merged.loc[mask].reset_index(drop=True)
        ts_arr = ts_arr[mask_np]
        ts_len = ts_len[mask_np]

    order = np.argsort(merged["mint_ts"].values, kind="stable")
    merged = merged.iloc[order].reset_index(drop=True)
    ts_arr = ts_arr[order]
    ts_len = ts_len[order]

    split_idx = int(len(merged) * train_ratio)

    rng = np.random.default_rng(shuffle_seed)
    train_perm = rng.permutation(split_idx)
    reorder = np.concatenate([train_perm, np.arange(split_idx, len(merged))])
    merged = merged.iloc[reorder].reset_index(drop=True)
    ts_arr = ts_arr[reorder]
    ts_len = ts_len[reorder]

    mint2label_info = (
        merged[["mint_address", "label", "min_ratio", "return_ratio"]]
        .set_index("mint_address")
        .to_dict(orient="index")
    )

    feature_cols = [c for c in merged.columns if c.startswith("group")]
    X_tab = merged[feature_cols].to_numpy(dtype=np.float32)
    X_train_tab, X_test_tab = X_tab[:split_idx], X_tab[split_idx:]
    if scale_tabular:
        scaler = StandardScaler()
        X_train_tab = scaler.fit_transform(X_train_tab).astype(np.float32)
        X_test_tab = scaler.transform(X_test_tab).astype(np.float32)

    ts_train, ts_test = ts_arr[:split_idx], ts_arr[split_idx:]
    if scale_ts:
        ts_train, ts_test = _normalize_ts(ts_train, ts_test)

    y = np.where(merged["label"].values == "high", 0, 1).astype(np.int64)
    mints = merged["mint_address"].to_numpy()

    return {
        "X_train_tab": X_train_tab,
        "X_test_tab": X_test_tab,
        "ts_train": ts_train,
        "ts_test": ts_test,
        "ts_len_train": ts_len[:split_idx],
        "ts_len_test": ts_len[split_idx:],
        "y_train": y[:split_idx],
        "y_test": y[split_idx:],
        "mint_train": mints[:split_idx],
        "mint_test": mints[split_idx:],
        "mint2label_info": mint2label_info,
        "feature_cols": feature_cols,
    }
