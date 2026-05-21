import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 512, 256), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        B, _, H = out.shape
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, H)
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last).squeeze(1)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        B, _, H = out.shape
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, H)
        last = out.gather(1, idx).squeeze(1)
        return self.fc(last).squeeze(1)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        use_last_timestep=True,
    ):
        super().__init__()
        self.use_last_timestep = use_last_timestep
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        B, T, _ = x.shape
        device = x.device
        lengths = lengths.to(device)
        seq_range = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        key_padding_mask = seq_range >= lengths.unsqueeze(1)
        enc_out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if self.use_last_timestep:
            idx = (lengths - 1).view(B, 1, 1).expand(B, 1, enc_out.size(-1))
            feat = enc_out.gather(1, idx).squeeze(1)
        else:
            mask_float = (~key_padding_mask).unsqueeze(-1).float()
            feat = (enc_out * mask_float).sum(dim=1) / lengths.unsqueeze(1).float()
        return self.fc(feat).squeeze(1)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        )

    def forward(self, x):
        residual = x if self.residual_conv is None else self.residual_conv(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_channels=128, num_blocks=6, kernel_size=9, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_blocks):
            layers.append(
                TCNBlock(in_ch, hidden_channels, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = hidden_channels
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x).squeeze(1)


def build_ts_model(name: str, input_dim: int) -> nn.Module:
    if name == "lstm":
        return LSTMClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2)
    if name == "gru":
        return GRUClassifier(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2)
    if name == "transformer":
        return TransformerClassifier(
            input_dim=input_dim, d_model=64, nhead=1, num_layers=2, dim_feedforward=64, dropout=0.2,
        )
    if name == "tcn":
        return TCNClassifier(input_dim=input_dim, hidden_channels=128, num_blocks=6, kernel_size=9, dropout=0.2)
    raise ValueError(f"Unknown TS model: {name}")


def build_sklearn_model(name: str, y_train: np.ndarray, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=1000,
            max_depth=16,
            min_samples_leaf=3,
            min_samples_split=6,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
            max_features="sqrt",
        )
    if name == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            min_samples_split=20,
            subsample=0.8,
            max_features="sqrt",
            random_state=seed,
        )
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    if name == "lr":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgb":
        pos_ratio = y_train.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio
        print("scale_pos_weight:", scale_pos_weight)
        return XGBClassifier(
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.02,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            early_stopping_rounds=100,
        )
    raise ValueError(f"Unknown sklearn model: {name}")
