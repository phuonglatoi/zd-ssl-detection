"""Huấn luyện 2 baseline: Autoencoder và Isolation Forest.

Chạy:
    python src/train_baseline.py --epochs 30
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import joblib

from models import Autoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = 'data/processed'
OUT = 'outputs'


def train_ae(args, X):
    print('\n========= Autoencoder =========')
    in_dim = X.shape[1]
    ae = Autoencoder(in_dim).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    for ep in range(1, args.epochs + 1):
        ae.train(); total = 0.0; n = 0
        for (xb,) in tqdm(dl, desc=f'AE ep {ep}/{args.epochs}', leave=False):
            xb = xb.to(DEVICE)
            x_hat = ae(xb)
            loss = crit(x_hat, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0); n += xb.size(0)
        print(f'[AE ep {ep:03d}] loss = {total / n:.6f}')

    # Threshold = percentile 95 của reconstruction error trên benign
    ae.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i:i + 4096]).to(DEVICE)
            err = ((ae(xb) - xb) ** 2).mean(dim=1).cpu().numpy()
            errs.append(err)
    errs = np.concatenate(errs)
    th = float(np.percentile(errs, 95))

    torch.save(ae.state_dict(), os.path.join(OUT, 'autoencoder.pt'))
    np.save(os.path.join(OUT, 'ae_threshold.npy'), np.array(th))
    print(f'[OK] AE đã lưu, threshold = {th:.6f}')


def train_iforest(X, max_samples=200_000):
    print('\n========= Isolation Forest =========')
    if len(X) > max_samples:
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X
    print(f'[INFO] Fit IF trên {X_fit.shape}')
    iforest = IsolationForest(
        n_estimators=200, max_samples='auto',
        contamination=0.05, random_state=42, n_jobs=-1
    ).fit(X_fit)
    joblib.dump(iforest, os.path.join(OUT, 'iforest.joblib'))
    print('[OK] IsolationForest đã lưu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--max-train', type=int, default=400_000)
    args = ap.parse_args()

    X = np.load(os.path.join(DATA, 'X_train_benign.npy'))
    if len(X) > args.max_train:
        X = X[np.random.RandomState(42).choice(len(X), args.max_train, replace=False)]
    print(f'[INFO] Device: {DEVICE} | X_train_benign: {X.shape}')

    train_ae(args, X)
    train_iforest(X)


if __name__ == '__main__':
    main()
