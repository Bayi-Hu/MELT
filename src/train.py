import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import average_precision_score

from dataset import PROJECT_ROOT, TSDataset, load_dataset, ts_collate
from model import MLP, build_sklearn_model, build_ts_model

ML_MODELS = {"rf", "xgb", "lgbm", "gbdt", "lr"}
TAB_DL_MODELS = {"mlp"}
TS_DL_MODELS = {"tcn", "lstm", "gru", "transformer"}
ALL_MODELS = ML_MODELS | TAB_DL_MODELS | TS_DL_MODELS

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def set_seed(seed: int):
    """Seed python `random`, numpy, torch (CPU + CUDA) so every RNG that
    downstream code touches is deterministic for this run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_generator(seed: int) -> torch.Generator:
    """Generator passed to DataLoader so its shuffle order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


EARLY_STOPPING_MODELS = {"xgb", "lgbm"}
VAL_FRAC = 0.1


def train_sklearn(args, data, seed: int):
    X_train, X_test = data["X_train_tab"], data["X_test_tab"]
    y_train, y_test = data["y_train"], data["y_test"]

    model = build_sklearn_model(args.model, y_train, seed)

    if args.model in EARLY_STOPPING_MODELS:
        val_size = max(1, int(len(X_train) * VAL_FRAC))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        print(f"early stopping: train={len(X_tr)} val={len(X_val)}")
        if args.model == "lgbm":
            from lightgbm import early_stopping, log_evaluation
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(100), log_evaluation(0)],
            )
        else:  # xgb (early_stopping_rounds set in constructor)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    auprc = average_precision_score(y_test, y_proba)
    print("AUPRC:", auprc)

    save_predictions(args.model, data["mint_test"], y_test, y_proba, epoch=None, auprc=auprc)
    return model


def train_mlp(args, data, seed: int):
    X_train, X_test = data["X_train_tab"], data["X_test_tab"]
    y_train, y_test = data["y_train"], data["y_test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=_loader_generator(seed),
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MLP(input_dim=X_train.shape[1]).to(device)
    pos_weight_val = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    print("pos_weight:", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        probs_chunks = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                probs_chunks.append(torch.sigmoid(model(batch_X.to(device))).cpu())
        y_proba = torch.cat(probs_chunks).numpy().reshape(-1)

        auprc = average_precision_score(y_test, y_proba)
        print(f"Epoch:{epoch}, AUPRC:", auprc)

        save_predictions(args.model, data["mint_test"], y_test, y_proba, epoch=epoch, auprc=auprc)
    return model


def train_ts(args, data, seed: int):
    X_train, X_test = data["ts_train"], data["ts_test"]
    len_train, len_test = data["ts_len_train"], data["ts_len_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader = DataLoader(
        TSDataset(X_train, len_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ts_collate,
        generator=_loader_generator(seed),
    )
    test_loader = DataLoader(
        TSDataset(X_test, len_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ts_collate,
    )

    model = build_ts_model(args.model, input_dim=X_train.shape[2]).to(device)
    print(model)
    pos_weight_val = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    print("pos_weight:", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=80)
        for batch_x, batch_len, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_len = batch_len.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x, batch_len)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        probs_chunks = []
        with torch.no_grad():
            for batch_x, batch_len, _ in test_loader:
                batch_x = batch_x.to(device)
                batch_len = batch_len.to(device)
                probs_chunks.append(torch.sigmoid(model(batch_x, batch_len)).cpu())
        y_proba = torch.cat(probs_chunks).numpy().reshape(-1)

        auprc = average_precision_score(y_test, y_proba)
        print(f"Epoch:{epoch}, AUPRC:", auprc)

        save_predictions(args.model, data["mint_test"], y_test, y_proba, epoch=epoch, auprc=auprc)
    return model


def save_predictions(model_name, mints, y_true, y_proba, epoch, auprc):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"{epoch}_" if epoch is not None else ""
    path = os.path.join(RESULTS_DIR, f"{model_name}_pred_{suffix}{auprc:.6f}.csv")
    pd.DataFrame({"mint": mints, "label": y_true, "prob": y_proba}).to_csv(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="xgb",
        choices=sorted(ALL_MODELS),
        help="Model type to train.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs (MLP only).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (MLP only).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (MLP only).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (Python, numpy, torch, sklearn, dataset shuffle, DataLoader).")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    print(f"model={args.model} seed={seed}")
    set_seed(seed)

    data = load_dataset(shuffle_seed=seed)

    if args.model in ML_MODELS:
        train_sklearn(args, data, seed)
    elif args.model in TAB_DL_MODELS:
        train_mlp(args, data, seed)
    elif args.model in TS_DL_MODELS:
        train_ts(args, data, seed)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
