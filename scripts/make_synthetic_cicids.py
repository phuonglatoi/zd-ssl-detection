"""Sinh synthetic CSV mô phỏng schema của CICIDS-2017 (CICFlowMeter ~78 features).

Chỉ phục vụ TEST pipeline khi chưa có dữ liệu thật. KHÔNG sử dụng cho báo cáo.

Sinh 8 file CSV trong data/raw/ với phân bố nhãn giống ngày thật của CICIDS-2017.
"""
import os
import numpy as np
import pandas as pd

OUT = 'data/raw'
os.makedirs(OUT, exist_ok=True)

# 78 cột feature CIC + 7 cột metadata + Label, gần với schema thật
FEATURE_COLS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]


def gen(n: int, label: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURE_COLS)
    if label == 'BENIGN':
        mu = rng.uniform(0, 0.5, size=n_feat)
        sigma = rng.uniform(0.3, 0.6, size=n_feat)
    elif 'DoS' in label or 'DDoS' in label:
        mu = rng.uniform(2.5, 4.0, size=n_feat)
        sigma = rng.uniform(0.5, 1.0, size=n_feat)
    elif 'PortScan' in label:
        mu = rng.uniform(-2.5, -1.0, size=n_feat)
        sigma = rng.uniform(0.4, 0.7, size=n_feat)
    elif 'Brute Force' in label or 'FTP' in label or 'SSH' in label:
        mu = rng.uniform(1.0, 2.5, size=n_feat)
        sigma = rng.uniform(0.4, 0.8, size=n_feat)
    elif 'Web' in label or 'XSS' in label or 'SQL' in label:
        mu = rng.uniform(-1.0, 0.5, size=n_feat)
        sigma = rng.uniform(0.6, 1.0, size=n_feat)
    elif 'Bot' in label:
        mu = rng.uniform(-0.5, 1.5, size=n_feat)
        sigma = rng.uniform(0.5, 0.9, size=n_feat)
    elif 'Infilt' in label:
        mu = rng.uniform(0.5, 1.8, size=n_feat)
        sigma = rng.uniform(0.7, 1.1, size=n_feat)
    else:
        mu = rng.uniform(0, 1, size=n_feat)
        sigma = rng.uniform(0.5, 0.9, size=n_feat)

    X = rng.normal(loc=mu, scale=sigma, size=(n, n_feat)).astype(np.float32)
    # Inject vài giá trị infinity và NaN để test cleaning
    if rng.random() < 0.3:
        n_bad = max(1, n // 200)
        rows = rng.integers(0, n, size=n_bad)
        cols = rng.integers(0, n_feat, size=n_bad)
        X[rows, cols] = np.inf

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    # Thêm cột metadata để mô phỏng đúng CICIDS (preprocess sẽ drop)
    df.insert(0, 'Flow ID', [f'10.0.0.{i % 200}-10.0.0.{(i + 1) % 200}-{rng.integers(1024, 60000)}-80-6' for i in range(n)])
    df.insert(1, 'Source IP', [f'10.0.0.{i % 200}' for i in range(n)])
    df.insert(2, 'Source Port', rng.integers(1024, 60000, size=n))
    df.insert(3, 'Destination IP', [f'10.0.0.{(i + 1) % 200}' for i in range(n)])
    df.insert(4, 'Destination Port', rng.choice([22, 80, 443, 21, 23, 53, 8080], size=n))
    df.insert(5, 'Protocol', rng.choice([6, 17], size=n))
    df.insert(6, 'Timestamp', pd.date_range('2017-07-03', periods=n, freq='s').strftime('%d/%m/%Y %H:%M:%S'))
    # CICIDS thật cột Label đôi khi có khoảng trắng đầu - mô phỏng:
    df[' Label'] = label
    return df


# Phân bố giống CICIDS-2017 thật, scale nhỏ ~ 1/100 để chạy nhanh
files = {
    'Monday-WorkingHours.pcap_ISCX.csv': [('BENIGN', 5000)],
    'Tuesday-WorkingHours.pcap_ISCX.csv': [('BENIGN', 4000), ('FTP-Patator', 400), ('SSH-Patator', 300)],
    'Wednesday-workingHours.pcap_ISCX.csv': [('BENIGN', 4500), ('DoS Hulk', 1500), ('DoS GoldenEye', 300),
                                             ('DoS slowloris', 200), ('DoS Slowhttptest', 200), ('Heartbleed', 30)],
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv': [
        ('BENIGN', 3500), ('Web Attack \xe2\x80\x93 Brute Force', 200),
        ('Web Attack \xe2\x80\x93 XSS', 100), ('Web Attack \xe2\x80\x93 Sql Injection', 30),
    ],
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv': [
        ('BENIGN', 3500), ('Infiltration', 60),
    ],
    'Friday-WorkingHours-Morning.pcap_ISCX.csv': [('BENIGN', 3000), ('Bot', 300)],
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv': [('BENIGN', 2500), ('PortScan', 1500)],
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv': [('BENIGN', 2500), ('DDoS', 2000)],
}

seed = 0
for fname, parts in files.items():
    dfs = []
    for label, n in parts:
        seed += 1
        dfs.append(gen(n, label, seed))
    full = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    path = os.path.join(OUT, fname)
    full.to_csv(path, index=False)
    print(f'[OK] {path}  shape={full.shape}  labels={full[" Label"].unique().tolist()}')
