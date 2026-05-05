"""Vẽ loss curve InfoNCE từ outputs/train_loss.npy.
Sinh ra figs/training_loss.png — chèn vào Hình 4.3 trong báo cáo.

Chạy:
    python src/plot_loss.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = 'outputs'
FIG = 'figs'
os.makedirs(FIG, exist_ok=True)

loss = np.load(os.path.join(OUT, 'train_loss.npy'))
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(loss) + 1), loss, lw=2, color='#1f4e79', marker='o')
plt.xlabel('Epoch')
plt.ylabel('InfoNCE loss (NT-Xent)')
plt.title(f'Đường cong huấn luyện Encoder Contrastive ({len(loss)} epoch)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'training_loss.png'), dpi=180)
plt.close()
print('[OK] Lưu', os.path.join(FIG, 'training_loss.png'))
