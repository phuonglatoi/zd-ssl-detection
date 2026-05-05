# Zero-Day Detection bằng Self-Supervised Learning – CICIDS-2017

Source code minh chứng cho báo cáo môn **An toàn mạng nâng cao – Đề tài 18 –
Phát hiện tấn công Zero-Day bằng học tự giám sát (Self-Supervised Learning)
thời gian thực**.

[![Smoke test](https://github.com/phuonglatoi/zd-ssl-detection/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/phuonglatoi/zd-ssl-detection/actions/workflows/smoke-test.yml)

> **Đã verify end-to-end** trên Python 3.10–3.12 + PyTorch 2.x (CPU). Đầy đủ
> 5 lệnh chạy sạch: `preprocess` → `train_contrastive` → `train_baseline` →
> `evaluate` → `stream_demo`. Repo có sẵn **synthetic dataset 32 MB** giống
> schema CICIDS-2017 để bạn clone về chạy demo ngay lập tức không cần tải
> CICIDS thật.

## Quick start (Ubuntu)

```bash
git clone https://github.com/phuonglatoi/zd-ssl-detection.git
cd zd-ssl-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic dataset đã có sẵn trong data/raw/, chạy luôn:
python src/preprocess.py
python src/train_contrastive.py --epochs 20 --batch 256
python src/train_baseline.py --epochs 15 --batch 256
python src/evaluate.py
python src/plot_loss.py
python src/stream_demo.py --rate 300 --batch 32 --n 1500
```

Sau khi chạy xong, kiểm tra: `figs/*.png`, `outputs/results.csv`,
`outputs/per_attack_auc.csv`.

## 1. Yêu cầu

- Python 3.10+ (3.10 hoặc 3.11 ổn định nhất)
- Khoảng **8 GB RAM** trở lên (CICIDS-2017 sau khi gộp ~3 triệu dòng)
- (Tuỳ chọn) GPU NVIDIA + CUDA → train nhanh gấp 5–10 lần CPU

## 2. Cấu trúc thư mục

```
zd_ssl_detection/
├── data/
│   ├── raw/                  <- BỎ 8 file CSV CICIDS-2017 vào đây
│   └── processed/            <- preprocess.py sinh ra
├── outputs/                  <- model, threshold, results.csv
├── figs/                     <- biểu đồ ROC, confusion, UMAP
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── train_contrastive.py
│   ├── train_baseline.py
│   ├── evaluate.py
│   └── stream_demo.py
├── requirements.txt
└── README.md
```

## 3. Cài đặt môi trường (VS Code, Windows hoặc Linux)

### 3.1. Mở project trong VS Code

`File > Open Folder…` → chọn thư mục `zd_ssl_detection`.

### 3.2. Tạo virtual environment

Mở Terminal trong VS Code (Ctrl+`):

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Nếu PowerShell báo "running scripts is disabled":
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Chọn interpreter trong VS Code

`Ctrl+Shift+P` → `Python: Select Interpreter` → chọn `.venv/.../python`.

### 3.4. (Tuỳ chọn) Cài PyTorch CUDA

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 4. Đặt dataset

Repo đã có sẵn **synthetic dataset** trong `data/raw/` (sinh bởi
`scripts/make_synthetic_cicids.py`, ~32 MB, schema giống hệt CICIDS-2017 —
8 file, 78 features + cột Label, các nhãn BENIGN / DoS Hulk / DoS GoldenEye /
DDoS / PortScan / FTP-Patator / SSH-Patator / Bot / Web Attack / Heartbleed /
Infiltration). Có thể chạy ngay không cần tải gì.

Khi muốn chạy với **CICIDS-2017 thật** (https://www.unb.ca/cic/datasets/ids-2017.html),
xóa các file synthetic rồi copy 8 file CSV thật vào `data/raw/`:

```
data/raw/Monday-WorkingHours.pcap_ISCX.csv
data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
data/raw/Wednesday-workingHours.pcap_ISCX.csv
data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv
data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## 5. Chạy 4 bước demo

```bash
# (1) Tiền xử lý: gộp 8 CSV -> tách benign/attack -> chuẩn hóa
python src/preprocess.py

# (2) Huấn luyện encoder Contrastive (~5–15 phút trên GPU)
python src/train_contrastive.py --epochs 50 --batch 512 --tau 0.5

# (3) Huấn luyện 2 baseline: Autoencoder + Isolation Forest
python src/train_baseline.py --epochs 30

# (4) Đánh giá – ROC, AUC, FPR, UMAP, Confusion matrix
python src/evaluate.py

# (5) (Tuỳ chọn) vẽ training loss curve cho Hình 4.3
python src/plot_loss.py
```

Sau bước (4), mở thư mục `figs/` để xem các biểu đồ:

- `roc_compare.png` – đường ROC 3 mô hình
- `score_distribution.png` – phân bố score benign vs attack
- `confusion_matrix_*.png` – confusion matrix mỗi mô hình
- `umap_embedding.png` – UMAP 2D không gian embedding
- `outputs/results.csv` – bảng AUC / FPR / TPR
- `outputs/per_attack_auc.csv` – AUC tách theo từng nhóm tấn công
- `figs/training_loss.png` – đường cong InfoNCE loss (sinh bởi `plot_loss.py`)

**Đây chính là các ảnh để chèn vào báo cáo:**

| Placeholder báo cáo | File ảnh cần dùng |
|---|---|
| Hình 4.2 (dataset preview) | screenshot `pandas.head()` trong VS Code |
| Hình 4.3 (loss curve) | `figs/training_loss.png` |
| Hình 4.4 (confusion matrix) | `figs/confusion_matrix_contrastive.png` |
| Hình 4.5 (real-time console) | screenshot terminal khi chạy `stream_demo.py` |
| Hình 5.4 (ROC thực) | `figs/roc_compare.png` |
| Hình 5.5 (UMAP thực) | `figs/umap_embedding.png` |
| Hình 5.6 (bảng benchmark) | screenshot `outputs/results.csv` + `per_attack_auc.csv` |

## 6. Demo streaming real-time (mô phỏng)

```bash
python src/stream_demo.py --rate 300 --batch 32 --n 3000
```

Console sẽ in các dòng `[INFO]` cho flow lành tính và `[ALERT]` đỏ cho flow bị
đánh dấu bất thường, kèm `latency` mỗi flow. Đây là ảnh để chèn vào **Hình 4.5**.

## 7. Lưu ý hiệu năng

- Nếu RAM < 8 GB: hạ `--max-train` xuống 100_000.
- Nếu chạy CPU only: bộ Contrastive 50 epoch trên 400k mẫu mất ~30–40 phút;
  có thể dùng `--epochs 20` để rút gọn (ROC-AUC sẽ giảm ~0.01).
- Lần đầu pandas đọc 8 CSV mất ~1–2 phút.

## 8. Khắc phục lỗi thường gặp

| Lỗi | Cách khắc phục |
|---|---|
| `MemoryError` khi đọc CSV | dùng `--max-train` thấp; hoặc chạy preprocess trên máy nhiều RAM |
| `CUDA out of memory` | hạ `--batch` xuống 256 hoặc 128 |
| `ImportError: umap` | `pip install umap-learn` |
| Cột `Label` không tìm thấy | mở CSV bằng Excel xem có khoảng trắng đầu cột; preprocess đã `.strip()` rồi |
| Kết quả AUC quá thấp | tăng `--epochs` lên 80–100; kiểm tra dataset có đủ benign chưa |
