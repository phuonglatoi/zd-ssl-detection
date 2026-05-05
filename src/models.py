"""Mô hình PyTorch: Encoder Contrastive + Autoencoder baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Contrastive ----------------
class Encoder(nn.Module):
    """Backbone f(.) - dùng cho cả huấn luyện và inference."""
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.BatchNorm1d(hidden),  nn.ReLU(),
            nn.Linear(hidden, 128),     nn.BatchNorm1d(128),     nn.ReLU(),
            nn.Linear(128, out_dim),    nn.BatchNorm1d(out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Projector(nn.Module):
    """g(.) - chỉ dùng khi huấn luyện."""
    def __init__(self, in_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """InfoNCE / NT-Xent loss như trong SimCLR."""
    z = torch.cat([z1, z2], dim=0)                # (2N, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau                         # (2N, 2N)
    N = z1.size(0)

    # Mask self-similarity = -inf
    mask_self = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask_self, float('-inf'))

    # Positive index: i <-> i+N
    targets = torch.arange(2 * N, device=z.device)
    targets = (targets + N) % (2 * N)

    return F.cross_entropy(sim, targets)


# ---------------- Augmentation ----------------
def augment(x: torch.Tensor, mask_p: float = 0.15, jitter_sigma: float = 0.05) -> torch.Tensor:
    """Stochastic augment cho dữ liệu dạng vector số."""
    # 1) Feature masking: random zero-out mask_p% chiều
    mask = (torch.rand_like(x) > mask_p).float()
    x = x * mask
    # 2) Gaussian jitter
    x = x + torch.randn_like(x) * jitter_sigma
    return x


# ---------------- Autoencoder baseline ----------------
class Autoencoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, 32),     nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 128),    nn.ReLU(),
            nn.Linear(128, 256),   nn.ReLU(),
            nn.Linear(256, in_dim),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
