"""Mô phỏng pipeline streaming đa luồng (Capture / Feature / Scoring).

Vì laptop khó setup sniff thật, demo này replay tập test theo dạng "stream":
- Thread 1 (Producer): sinh từng flow (giả lập Capture+Feature) -> Q1
- Thread 2 (Consumer): lấy batch flow từ Q1 -> Encoder -> Anomaly Score -> ALERT

Chạy:
    python src/stream_demo.py --rate 200 --batch 32
"""
import os
import time
import argparse
import threading
import queue
import numpy as np
import torch

from models import Encoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = 'data/processed'
OUT = 'outputs'

RED = '\033[91m'; GRN = '\033[92m'; YLW = '\033[93m'; CYN = '\033[96m'; RST = '\033[0m'


def producer(q, X, cat, rate):
    """Sinh flow theo tốc độ rate flows/sec."""
    interval = 1.0 / rate
    for i, (x, c) in enumerate(zip(X, cat)):
        q.put((time.time(), i, x, c))
        time.sleep(interval)
    q.put(None)  # sentinel


def consumer(q, encoder, centroid_n, threshold, batch_size):
    pending = []
    n_total = n_alert = n_tp = n_fp = 0
    while True:
        item = q.get()
        if item is None:
            break
        pending.append(item)
        if len(pending) >= batch_size:
            flush(pending, encoder, centroid_n, threshold,
                  stats=(n_total, n_alert, n_tp, n_fp))
            n_total += len(pending)
            pending.clear()
    if pending:
        flush(pending, encoder, centroid_n, threshold)


def flush(batch, encoder, centroid_n, threshold, stats=None):
    ts0 = batch[0][0]
    xs = np.stack([b[2] for b in batch], axis=0)
    cats = [b[3] for b in batch]
    with torch.no_grad():
        z = encoder(torch.from_numpy(xs).to(DEVICE)).cpu().numpy()
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    dist = 1 - z @ centroid_n
    for (ts, idx, _, c), s in zip(batch, dist):
        is_alert = s > threshold
        latency_ms = (time.time() - ts) * 1000
        if is_alert:
            label = 'ATTACK' if c != 'BENIGN' else 'BENIGN'
            color = RED if c != 'BENIGN' else YLW
            tag = 'TRUE+' if c != 'BENIGN' else 'FALSE+'
            print(f'{color}[ALERT]{RST} idx={idx:>6}  score={s:.4f}  '
                  f'latency={latency_ms:5.1f}ms  cat={c:<14}  ({tag})')
        else:
            if idx % 200 == 0:
                print(f'{CYN}[ INFO]{RST} idx={idx:>6}  score={s:.4f}  '
                      f'latency={latency_ms:5.1f}ms  cat={c:<14}  (OK)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=int, default=200,
                    help='flows/sec (mô phỏng tốc độ stream)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--n', type=int, default=2000,
                    help='số flow chạy demo (lấy ngẫu nhiên từ test)')
    args = ap.parse_args()

    X = np.load(os.path.join(DATA, 'X_test.npy'))
    cat = np.load(os.path.join(DATA, 'y_test_cat.npy'), allow_pickle=True)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(X), size=min(args.n, len(X)), replace=False)
    X = X[idx]; cat = cat[idx]
    print(f'[INFO] Replay {len(X)} flow @ {args.rate} flows/s')

    in_dim = X.shape[1]
    encoder = Encoder(in_dim).to(DEVICE)
    encoder.load_state_dict(torch.load(os.path.join(OUT, 'encoder_best.pt'), map_location=DEVICE))
    encoder.eval()
    centroid = np.load(os.path.join(OUT, 'centroid.npy'))
    centroid_n = centroid / (np.linalg.norm(centroid) + 1e-8)
    threshold = float(np.load(os.path.join(OUT, 'threshold.npy')))
    print(f'[INFO] threshold = {threshold:.4f}\n')

    q = queue.Queue(maxsize=512)
    t1 = threading.Thread(target=producer, args=(q, X, cat, args.rate), daemon=True)
    t2 = threading.Thread(target=consumer,
                          args=(q, encoder, centroid_n, threshold, args.batch),
                          daemon=True)
    t0 = time.time()
    t1.start(); t2.start()
    t1.join(); t2.join()
    print(f'\n[OK] Demo kết thúc sau {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
