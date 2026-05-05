"""Huấn luyện encoder bằng Self-Supervised Contrastive Learning (InfoNCE).

Chạy:
    python src/train_contrastive.py --epochs 50 --batch 512 --tau 0.5
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import Encoder, Projector, info_nce, augment

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = 'data/processed'
OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-train', type=int, default=400_000,
                    help='Subsample benign train (CICIDS có > 1M benign, dùng tất cả sẽ chậm).')
    args = ap.parse_args()

    print(f'[INFO] Device: {DEVICE}')
    X = np.load(os.path.join(DATA, 'X_train_benign.npy'))
    if len(X) > args.max_train:
        idx = np.random.RandomState(42).choice(len(X), args.max_train, replace=False)
        X = X[idx]
    print(f'[INFO] Train benign shape: {X.shape}')

    in_dim = X.shape[1]
    encoder = Encoder(in_dim).to(DEVICE)
    projector = Projector(64, 32).to(DEVICE)

    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)

    history = []
    best_loss = float('inf')

    for ep in range(1, args.epochs + 1):
        encoder.train(); projector.train()
        total = 0.0
        n = 0
        for (xb,) in tqdm(dl, desc=f'Epoch {ep}/{args.epochs}', leave=False):
            xb = xb.to(DEVICE)
            x1 = augment(xb)
            x2 = augment(xb)
            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            loss = info_nce(z1, z2, tau=args.tau)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        avg = total / max(n, 1)
        history.append(avg)
        print(f'[EP {ep:03d}] loss = {avg:.4f}   lr = {sched.get_last_lr()[0]:.2e}')

        if avg < best_loss:
            best_loss = avg
            torch.save(encoder.state_dict(), os.path.join(OUT, 'encoder_best.pt'))
            print(f'         -> saved best (loss {avg:.4f})')

    # Tính centroid của benign train trong không gian embedding
    encoder.load_state_dict(torch.load(os.path.join(OUT, 'encoder_best.pt')))
    encoder.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i:i + 4096]).to(DEVICE)
            embs.append(encoder(xb).cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    centroid = embs.mean(axis=0)
    np.save(os.path.join(OUT, 'centroid.npy'), centroid)
    np.save(os.path.join(OUT, 'train_loss.npy'), np.array(history))

    # Threshold = percentile-95 của khoảng cách cosine
    centroid_n = centroid / (np.linalg.norm(centroid) + 1e-8)
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    dist = 1 - embs_n @ centroid_n
    th = float(np.percentile(dist, 95))
    np.save(os.path.join(OUT, 'threshold.npy'), np.array(th))
    print(f'\n[OK] Centroid lưu tại outputs/centroid.npy')
    print(f'[OK] Threshold (P95) = {th:.4f}')


if __name__ == '__main__':
    main()
