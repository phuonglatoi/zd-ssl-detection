"""
preprocess.py - Tiền xử lý 8 file CSV của CICIDS-2017.

Chạy 1 lần đầu:
    python src/preprocess.py

Output:
    data/processed/X_train_benign.npy   (~80 features, chỉ benign, để train SSL)
    data/processed/X_test.npy           (benign + tất cả attack, để test)
    data/processed/y_test.npy           (0 = benign, 1 = attack)
    data/processed/y_test_cat.npy       (nhãn chi tiết, để phân tích từng nhóm)
    data/processed/scaler.joblib        (StandardScaler đã fit trên benign train)
    data/processed/feature_names.json
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------------------------------------------
# Cấu hình - thay nếu folder dataset của bạn nằm ở chỗ khác
# -----------------------------------------------------------------
DATA_DIR = os.environ.get('CICIDS_DIR', 'data/raw')
OUT_DIR = 'data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

# Kiểu nhãn benign trong CICIDS-2017 (có vài biến thể chữ hoa/thường)
BENIGN_TAGS = {'BENIGN', 'Benign', 'benign'}


def load_all_csv(data_dir: str) -> pd.DataFrame:
    csvs = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csvs:
        raise FileNotFoundError(
            f'Không tìm thấy file CSV nào trong "{data_dir}". '
            f'Hãy copy 8 file CSV của CICIDS-2017 vào {data_dir}/'
        )
    print(f'[INFO] Tìm thấy {len(csvs)} file CSV:')
    for c in csvs:
        print('   -', os.path.basename(c))

    dfs = []
    for c in csvs:
        # CICIDS-2017 CSV thường mã hóa UTF-8 (có ký tự en-dash trong "Web Attack – Brute Force").
        # Một số file lại là cp1252/latin-1, ta thử lần lượt.
        df = None
        for enc in ('utf-8', 'cp1252', 'latin-1'):
            try:
                df = pd.read_csv(c, low_memory=False, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise UnicodeDecodeError('utf-8', b'', 0, 1, f'Không đọc được {c}')
        # Cột Label trong CICIDS đôi khi có khoảng trắng đầu " Label" -> strip()
        df.columns = [c_.strip() for c_ in df.columns]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    print(f'[INFO] Tổng số dòng: {len(full):,}')
    return full


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Đảm bảo cột Label tồn tại
    if 'Label' not in df.columns:
        raise KeyError('Không thấy cột Label trong CSV. Cấu trúc CSV không đúng?')

    # Loại các cột bị infinity/NaN sau khi convert numeric
    df = df.replace([np.inf, -np.inf], np.nan)

    # Loại các cột không phải feature số (Flow ID, IPs, timestamps)
    drop_candidates = ['Flow ID', 'Source IP', 'Destination IP',
                       'Timestamp', 'Source Port', 'Destination Port',
                       'Fwd Header Length.1']  # cột trùng phổ biến trong CICIDS
    drop_cols = [c for c in drop_candidates if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Drop NaN
    before = len(df)
    df = df.dropna()
    print(f'[INFO] Drop NaN: {before:,} -> {len(df):,}')

    return df


def main():
    print(f'[INFO] Đang đọc CICIDS-2017 từ {DATA_DIR} ...')
    df = load_all_csv(DATA_DIR)
    df = clean(df)

    # In phân bố nhãn
    print('\n[INFO] Phân bố nhãn:')
    print(df['Label'].value_counts())

    # Tách feature/label - strip ký tự rác trong nhãn (đôi khi CSV có khoảng trắng/en-dash hỏng)
    df['Label'] = df['Label'].astype(str).str.strip()
    y_cat = df['Label'].copy()
    y = (~y_cat.isin(BENIGN_TAGS)).astype(int)  # 0 = benign, 1 = attack
    X = df.drop(columns=['Label'])

    # Chuyển categorical Protocol nếu là string -> code
    for col in X.select_dtypes(include='object').columns:
        X[col] = pd.Categorical(X[col]).codes

    feature_names = X.columns.tolist()
    X = X.astype('float32').values

    print(f'[INFO] Số đặc trưng: {X.shape[1]}')
    print(f'[INFO] Số mẫu benign: {(y == 0).sum():,}  | attack: {(y == 1).sum():,}')

    # ----------------------------------------------------------------
    # Tách train/test
    #   - Lấy 60% benign làm train SSL
    #   - 40% benign + toàn bộ attack làm test
    # ----------------------------------------------------------------
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]

    train_idx, test_b_idx = train_test_split(benign_idx, test_size=0.4, random_state=42)

    X_train = X[train_idx]
    X_test = np.concatenate([X[test_b_idx], X[attack_idx]], axis=0)
    y_test = np.concatenate([np.zeros(len(test_b_idx)), np.ones(len(attack_idx))])
    y_test_cat = np.concatenate([
        np.array(['BENIGN'] * len(test_b_idx)),
        y_cat.iloc[attack_idx].values
    ])

    # Chuẩn hóa: fit chỉ trên benign train
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype('float32')
    X_test = scaler.transform(X_test).astype('float32')

    # Lưu
    np.save(os.path.join(OUT_DIR, 'X_train_benign.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test.astype('int8'))
    np.save(os.path.join(OUT_DIR, 'y_test_cat.npy'), y_test_cat)
    joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.joblib'))
    with open(os.path.join(OUT_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print('\n[OK] Đã lưu artefact vào', OUT_DIR)
    print('   X_train_benign:', X_train.shape)
    print('   X_test       :', X_test.shape)
    print('   y_test (0/1) :', np.bincount(y_test.astype(int)))


if __name__ == '__main__':
    main()
