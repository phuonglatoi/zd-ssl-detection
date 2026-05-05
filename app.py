"""Streamlit dashboard cho demo Zero-Day SSL Detection.

Chạy 1 lệnh:
    streamlit run app.py

Dashboard có:
- 5 nút chạy 5 bước pipeline (Preprocess / Train SSL / Train Baseline /
  Evaluate / Stream Demo)
- Live log streaming (subprocess pipe stdout)
- Tabs hiển thị các biểu đồ: ROC, UMAP, Confusion matrix, training loss
- Bảng results.csv + per_attack_auc.csv
- Metric chính (AUC, FPR, TPR) cho 3 mô hình
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / 'src'
FIGS = ROOT / 'figs'
OUT = ROOT / 'outputs'
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROC = ROOT / 'data' / 'processed'

st.set_page_config(
    page_title='Zero-Day SSL Detection — CICIDS-2017',
    page_icon='🛡️',
    layout='wide',
)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    # 🛡️ Zero-Day Detection bằng Self-Supervised Learning
    ### Đề tài 18 — An toàn mạng nâng cao (CICIDS-2017)
    Pipeline: **Preprocess → Train Contrastive Encoder → Train AE/IF Baseline → Evaluate → Streaming Demo**
    """
)
st.divider()


# ----------------------------------------------------------------------------
# Sidebar: hiển thị trạng thái dataset & model
# ----------------------------------------------------------------------------
def _file_status(path: Path, label: str) -> str:
    return f'✅ {label}' if path.exists() else f'⏳ {label}'


with st.sidebar:
    st.header('📊 Trạng thái pipeline')
    csvs = list(DATA_RAW.glob('*.csv')) if DATA_RAW.exists() else []
    st.markdown(f"**Dataset (`data/raw/`):** {len(csvs)} CSV")
    for c in sorted(csvs)[:8]:
        st.caption(f'• {c.name} ({c.stat().st_size / 1e6:.1f} MB)')

    st.markdown('---')
    st.markdown('**Artefacts đã sinh:**')
    st.markdown(_file_status(DATA_PROC / 'X_train_benign.npy', 'Preprocessed'))
    st.markdown(_file_status(OUT / 'encoder_best.pt', 'Encoder Contrastive'))
    st.markdown(_file_status(OUT / 'autoencoder.pt', 'Autoencoder'))
    st.markdown(_file_status(OUT / 'iforest.joblib', 'Isolation Forest'))
    st.markdown(_file_status(OUT / 'results.csv', 'Evaluation results'))

    st.markdown('---')
    st.header('⚙️ Tham số huấn luyện')
    epochs_ssl = st.slider('SSL epochs', 5, 100, 20, step=5)
    batch_ssl = st.select_slider('SSL batch', [128, 256, 512, 1024], value=256)
    tau = st.slider('Temperature τ', 0.1, 1.0, 0.5, step=0.05)
    epochs_ae = st.slider('AE epochs', 5, 50, 15, step=5)
    st.markdown('---')
    st.header('🌊 Tham số streaming')
    rate = st.slider('Flow rate (flows/s)', 50, 1000, 300, step=50)
    n_stream = st.slider('Số flow replay', 200, 5000, 1500, step=100)
    batch_stream = st.select_slider('Batch consumer', [8, 16, 32, 64], value=32)


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_run, tab_results, tab_figs, tab_stream, tab_about = st.tabs(
    ['🚀 Chạy pipeline', '📈 Kết quả', '📊 Biểu đồ', '🌊 Streaming', 'ℹ️ Thông tin']
)


def stream_subprocess(cmd: list[str], log_box) -> int:
    """Chạy subprocess và stream stdout về log_box realtime."""
    log_lines: list[str] = []
    log_lines.append(f'$ {" ".join(cmd)}')
    log_box.code('\n'.join(log_lines), language='bash')

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ''):
        log_lines.append(line.rstrip())
        # Giới hạn 200 dòng cuối để UI không lag
        log_box.code('\n'.join(log_lines[-200:]), language='bash')
    proc.wait()
    return proc.returncode


# ============================================================================
# Tab 1: Chạy pipeline
# ============================================================================
with tab_run:
    st.subheader('Chạy từng bước (hoặc tất cả)')

    col1, col2, col3, col4, col5 = st.columns(5)
    btn_pre = col1.button('1️⃣ Preprocess', use_container_width=True)
    btn_ssl = col2.button('2️⃣ Train SSL', use_container_width=True)
    btn_ae = col3.button('3️⃣ Train AE+IF', use_container_width=True)
    btn_eval = col4.button('4️⃣ Evaluate', use_container_width=True)
    btn_loss = col5.button('5️⃣ Plot loss', use_container_width=True)

    btn_all = st.button('▶️ Chạy toàn bộ pipeline (1→5)', type='primary', use_container_width=True)

    log_box = st.empty()

    py = sys.executable
    if btn_pre or btn_all:
        rc = stream_subprocess([py, str(SRC / 'preprocess.py')], log_box)
        st.success('✅ Preprocess xong') if rc == 0 else st.error(f'❌ Preprocess lỗi (rc={rc})')
    if btn_ssl or btn_all:
        rc = stream_subprocess(
            [py, str(SRC / 'train_contrastive.py'),
             '--epochs', str(epochs_ssl), '--batch', str(batch_ssl), '--tau', str(tau)],
            log_box,
        )
        st.success('✅ Train SSL xong') if rc == 0 else st.error(f'❌ Train SSL lỗi (rc={rc})')
    if btn_ae or btn_all:
        rc = stream_subprocess(
            [py, str(SRC / 'train_baseline.py'), '--epochs', str(epochs_ae), '--batch', str(batch_ssl)],
            log_box,
        )
        st.success('✅ Train AE+IF xong') if rc == 0 else st.error(f'❌ Train baseline lỗi (rc={rc})')
    if btn_eval or btn_all:
        rc = stream_subprocess([py, str(SRC / 'evaluate.py')], log_box)
        st.success('✅ Evaluate xong') if rc == 0 else st.error(f'❌ Evaluate lỗi (rc={rc})')
    if btn_loss or btn_all:
        rc = stream_subprocess([py, str(SRC / 'plot_loss.py')], log_box)
        st.success('✅ Plot loss xong') if rc == 0 else st.error(f'❌ Plot loss lỗi (rc={rc})')


# ============================================================================
# Tab 2: Kết quả định lượng
# ============================================================================
with tab_results:
    st.subheader('Bảng kết quả AUC / FPR / TPR')
    res_path = OUT / 'results.csv'
    per_path = OUT / 'per_attack_auc.csv'

    if res_path.exists():
        df = pd.read_csv(res_path)
        # 3 metric card lớn
        cols = st.columns(len(df))
        for col, (_, row) in zip(cols, df.iterrows()):
            col.metric(
                label=row['model'],
                value=f"AUC = {row['AUC']:.3f}",
                delta=f"FPR={row['FPR']:.2%} | TPR={row['TPR']:.2%}",
            )
        st.dataframe(df.style.format({
            'AUC': '{:.4f}', 'FPR': '{:.4f}', 'TPR': '{:.4f}', 'threshold': '{:.4f}'
        }), use_container_width=True)
    else:
        st.info('Chưa có results.csv. Chạy bước 4 (Evaluate) trước.')

    st.divider()
    st.subheader('AUC theo từng nhóm tấn công (per-attack)')
    if per_path.exists():
        dfp = pd.read_csv(per_path)
        st.dataframe(
            dfp.style.background_gradient(cmap='RdYlGn', subset=dfp.select_dtypes('number').columns),
            use_container_width=True,
        )
    else:
        st.info('Chưa có per_attack_auc.csv.')


# ============================================================================
# Tab 3: Biểu đồ
# ============================================================================
with tab_figs:
    st.subheader('Các biểu đồ được sinh từ evaluate.py')

    fig_specs = [
        ('roc_compare.png', 'ROC – So sánh 3 mô hình'),
        ('training_loss.png', 'InfoNCE Training loss'),
        ('umap_embedding.png', 'UMAP 2D — embedding của Encoder'),
        ('score_distribution.png', 'Phân bố anomaly score'),
        ('confusion_matrix_contrastive.png', 'Confusion matrix — Contrastive'),
        ('confusion_matrix_autoencoder.png', 'Confusion matrix — Autoencoder'),
        ('confusion_matrix_isolationforest.png', 'Confusion matrix — Isolation Forest'),
    ]

    cols = st.columns(2)
    for i, (fname, title) in enumerate(fig_specs):
        with cols[i % 2]:
            p = FIGS / fname
            st.markdown(f'**{title}**')
            if p.exists():
                st.image(str(p), use_container_width=True)
            else:
                st.info(f'Chưa có `{fname}`. Chạy evaluate / plot_loss để sinh.')


# ============================================================================
# Tab 4: Streaming demo realtime
# ============================================================================
with tab_stream:
    st.subheader('Mô phỏng streaming real-time')
    st.caption(
        'Pipeline producer/consumer: capture flow → encode → cosine distance → '
        'so sánh với threshold → in [INFO]/[ALERT]'
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    rate_box = col_a.empty()
    seen_box = col_b.empty()
    alert_box = col_c.empty()
    avg_lat_box = col_d.empty()

    log_stream = st.empty()
    btn_stream_run = st.button('🌊 Chạy streaming', type='primary', use_container_width=True)

    if btn_stream_run:
        py = sys.executable
        cmd = [py, str(SRC / 'stream_demo.py'),
               '--rate', str(rate), '--batch', str(batch_stream), '--n', str(n_stream)]
        log_stream.code(f'$ {" ".join(cmd)}\n', language='bash')

        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        n_seen = 0
        n_alert = 0
        lat_sum = 0.0
        lat_cnt = 0
        log_buf: list[str] = []
        t_start = time.time()
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ''):
            log_buf.append(line.rstrip())
            if 'idx=' in line:
                n_seen += 1
                if '[ALERT]' in line:
                    n_alert += 1
                # parse latency (e.g. "latency= 73.5ms")
                if 'latency=' in line:
                    try:
                        lat_str = line.split('latency=')[1].split('ms')[0].strip()
                        lat_sum += float(lat_str)
                        lat_cnt += 1
                    except Exception:
                        pass
            elapsed = max(time.time() - t_start, 0.01)
            rate_box.metric('Throughput', f'{n_seen / elapsed:.0f} flow/s')
            seen_box.metric('Đã xử lý', f'{n_seen}')
            alert_box.metric('Số ALERT', f'{n_alert}')
            avg_lat_box.metric('Latency TB', f'{lat_sum / max(lat_cnt, 1):.1f} ms')
            log_stream.code('\n'.join(log_buf[-150:]), language='text')
        proc.wait()
        st.success(f'✅ Streaming xong sau {time.time() - t_start:.1f}s')


# ============================================================================
# Tab 5: Thông tin
# ============================================================================
with tab_about:
    st.markdown(
        """
### Về dự án

- **Đề tài:** 18 — Phát hiện tấn công Zero-Day bằng Self-Supervised Learning thời gian thực
- **Môn:** An toàn mạng nâng cao
- **Repo:** [github.com/phuonglatoi/zd-ssl-detection](https://github.com/phuonglatoi/zd-ssl-detection)
- **Dataset:** CICIDS-2017 (https://www.unb.ca/cic/datasets/ids-2017.html)
- **Mô hình so sánh:** Contrastive Encoder (SSL) — Autoencoder — Isolation Forest

### Kiến trúc

- Encoder MLP `[in → 256 → 128 → 64]` với BatchNorm + ReLU
- Projector `[64 → 64 → 32]` (chỉ dùng khi train, bỏ khi inference)
- InfoNCE / NT-Xent loss với τ = 0.5
- Augment: feature masking (15%) + Gaussian jitter (σ = 0.05)
- Anomaly score: cosine distance đến centroid của benign trong embedding space
- Threshold: phân vị 95 (P95) của cosine distance trên tập train benign

### Tham khảo

[1] Chen et al. *SimCLR* — ICML 2020
[2] He et al. *MoCo* — CVPR 2020
[3] van den Oord et al. *InfoNCE* — arXiv:1807.03748
[4] Sharafaldin et al. *CICIDS-2017* — ICISSP 2018
        """
    )
