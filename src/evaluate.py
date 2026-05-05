"""Đánh giá 3 mô hình: Contrastive Encoder, Autoencoder, Isolation Forest.

Sinh ra:
    figs/roc_compare.png            - đường ROC 3 mô hình
    figs/score_distribution.png     - phân bố score benign vs attack
    figs/confusion_matrix_*.png     - confusion matrix mỗi mô hình
    figs/umap_embedding.png         - giảm chiều 2D embedding của Contrastive
    outputs/results.csv             - bảng AUC / FPR / Detection Delay
"""
import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay)

from models import Encoder, Autoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = 'data/processed'
OUT = 'outputs'
FIG = 'figs'
os.makedirs(FIG, exist_ok=True)


def load_test():
    X = np.load(os.path.join(DATA, 'X_test.npy'))
    y = np.load(os.path.join(DATA, 'y_test.npy'))
    cat = np.load(os.path.join(DATA, 'y_test_cat.npy'), allow_pickle=True)
    return X, y, cat


def score_contrastive(X):
    in_dim = X.shape[1]
    enc = Encoder(in_dim).to(DEVICE)
    enc.load_state_dict(torch.load(os.path.join(OUT, 'encoder_best.pt'), map_location=DEVICE))
    enc.eval()
    cent = np.load(os.path.join(OUT, 'centroid.npy'))
    cent_n = cent / (np.linalg.norm(cent) + 1e-8)

    embs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i:i + 4096]).to(DEVICE)
            embs.append(enc(xb).cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    dist = 1 - embs_n @ cent_n            # cao = bất thường
    return dist, embs


def score_autoencoder(X):
    in_dim = X.shape[1]
    ae = Autoencoder(in_dim).to(DEVICE)
    ae.load_state_dict(torch.load(os.path.join(OUT, 'autoencoder.pt'), map_location=DEVICE))
    ae.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(X[i:i + 4096]).to(DEVICE)
            err = ((ae(xb) - xb) ** 2).mean(dim=1).cpu().numpy()
            errs.append(err)
    return np.concatenate(errs)


def score_iforest(X):
    iforest = joblib.load(os.path.join(OUT, 'iforest.joblib'))
    return -iforest.score_samples(X)      # cao = bất thường


def threshold_by_fpr(scores_benign, target_fpr=0.05):
    return float(np.percentile(scores_benign, 100 * (1 - target_fpr)))


def evaluate_one(name, scores, y, target_fpr=0.05):
    fpr, tpr, _ = roc_curve(y, scores)
    auc_val = auc(fpr, tpr)
    th = threshold_by_fpr(scores[y == 0], target_fpr=target_fpr)
    pred = (scores > th).astype(int)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_actual = fp / (fp + tn)
    tpr_actual = tp / (tp + fn)
    print(f'  AUC = {auc_val:.4f} | thresh@FPR≈{target_fpr*100:.0f}% = {th:.4f} | '
          f'FPR thực = {fpr_actual*100:.2f}% | TPR = {tpr_actual*100:.2f}%')
    return dict(name=name, auc=auc_val, fpr=fpr_actual, tpr=tpr_actual,
                fpr_curve=fpr, tpr_curve=tpr, cm=cm, threshold=th, scores=scores)


def plot_roc(results):
    plt.figure(figsize=(7, 6))
    for r in results:
        plt.plot(r['fpr_curve'], r['tpr_curve'],
                 label=f"{r['name']} (AUC = {r['auc']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], '--', c='gray', lw=1, label='Random')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC – So sánh 3 mô hình trên CICIDS-2017')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'roc_compare.png'), dpi=180)
    plt.close()


def plot_score_dist(results, y):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    for ax, r in zip(axes, results):
        s = r['scores']
        ax.hist(s[y == 0], bins=80, alpha=0.6, label='Benign', color='#1f4e79')
        ax.hist(s[y == 1], bins=80, alpha=0.6, label='Attack', color='#c00000')
        ax.axvline(r['threshold'], color='black', linestyle='--', label=f'threshold')
        ax.set_title(r['name'])
        ax.set_xlabel('score'); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('Phân bố score: Benign vs Attack')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'score_distribution.png'), dpi=180)
    plt.close()


def plot_confusion(results):
    for r in results:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(r['cm'], display_labels=['Benign', 'Attack']).plot(
            ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(f"Confusion Matrix – {r['name']}")
        fig.tight_layout()
        safe = r['name'].lower().replace(' ', '_')
        fig.savefig(os.path.join(FIG, f'confusion_matrix_{safe}.png'), dpi=180)
        plt.close()


def plot_umap(embs, y, cat, max_pts=15_000):
    try:
        import umap
    except Exception as e:
        print('[WARN] umap-learn chưa cài, bỏ qua UMAP:', e)
        return
    if len(embs) > max_pts:
        idx = np.random.RandomState(0).choice(len(embs), max_pts, replace=False)
        embs = embs[idx]; y = y[idx]; cat = cat[idx]
    print('[INFO] Đang tính UMAP 2D ...')
    em2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embs)

    fig, ax = plt.subplots(figsize=(8, 6))
    classes = pd.Series(cat).unique()
    palette = sns.color_palette('tab20', n_colors=len(classes))
    for cls, color in zip(classes, palette):
        m = cat == cls
        ax.scatter(em2[m, 0], em2[m, 1], s=4, alpha=0.5, color=color, label=cls)
    ax.set_title('UMAP 2D – embedding của Contrastive Encoder (CICIDS-2017)')
    ax.legend(loc='best', fontsize=7, markerscale=2, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'umap_embedding.png'), dpi=180)
    plt.close()


def per_attack_report(results, y, cat):
    rows = []
    classes = pd.Series(cat).unique()
    for cls in classes:
        if cls == 'BENIGN':
            continue
        m_pos = cat == cls
        m_neg = cat == 'BENIGN'
        mask = m_pos | m_neg
        y_b = np.concatenate([np.zeros(m_neg.sum()), np.ones(m_pos.sum())])
        for r in results:
            s = r['scores'][mask]
            order = np.concatenate([np.where(m_neg)[0], np.where(m_pos)[0]])
            s_ord = r['scores'][order]
            try:
                auc_v = auc(*roc_curve(y_b, s_ord)[:2])
            except Exception:
                auc_v = float('nan')
            rows.append({'attack': cls, 'model': r['name'], 'AUC': round(auc_v, 4)})
    df = pd.DataFrame(rows).pivot(index='attack', columns='model', values='AUC')
    df.to_csv(os.path.join(OUT, 'per_attack_auc.csv'))
    print('\n[OK] AUC theo từng nhóm tấn công:\n', df)


def main():
    X, y, cat = load_test()
    print(f'[INFO] Test shape: X={X.shape}  benign={int((y == 0).sum())}  attack={int((y == 1).sum())}')

    print('\n[1] Contrastive Encoder')
    s_c, embs = score_contrastive(X)
    r_c = evaluate_one('Contrastive', s_c, y)

    print('\n[2] Autoencoder')
    s_a = score_autoencoder(X)
    r_a = evaluate_one('Autoencoder', s_a, y)

    print('\n[3] Isolation Forest')
    s_i = score_iforest(X)
    r_i = evaluate_one('IsolationForest', s_i, y)

    results = [r_c, r_a, r_i]
    plot_roc(results)
    plot_score_dist(results, y)
    plot_confusion(results)
    plot_umap(embs, y, cat)
    per_attack_report(results, y, cat)

    pd.DataFrame([{'model': r['name'], 'AUC': round(r['auc'], 4),
                   'FPR': round(r['fpr'], 4), 'TPR': round(r['tpr'], 4),
                   'threshold': round(r['threshold'], 6)} for r in results]) \
      .to_csv(os.path.join(OUT, 'results.csv'), index=False)
    print('\n[OK] Đã lưu figs/ và outputs/results.csv')


if __name__ == '__main__':
    main()
