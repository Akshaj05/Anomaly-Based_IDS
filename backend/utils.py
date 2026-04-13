"""
utils.py
--------
Handles all data preprocessing for the UNSW-NB15 dataset.

UNSW-NB15 Feature Categories:
- Flow features: duration, proto, service, state
- Basic features: sbytes, dbytes, sttl, dttl, sloss, dloss
- Content features: sload, dload, spkts, dpkts
- Time features: stcpb, dtcpb, tcprtt, synack, ackdat
- Additional features: smean, dmean, trans_depth, res_bdy_len, etc.
- Connection features: ct_srv_src, ct_state_ttl, ct_dst_ltm, etc.
- Label: attack_cat (attack category), label (0=normal, 1=attack)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# These are the 42 numerical/categorical features we use from UNSW-NB15.
# We drop 'id', 'attack_cat' (target leakage), and 'label' (ground truth).
FEATURE_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload',
    'spkts', 'dpkts', 'smeansz', 'dmeansz', 'trans_depth',
    'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
    'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
]

# Categorical columns that need label encoding
CATEGORICAL_COLS = ['proto', 'service', 'state']

# Numerical columns that need scaling
NUMERICAL_COLS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLS]


def load_unsw_nb15(filepath: str) -> pd.DataFrame:
    # Official UNSW-NB15 column names (49 features + label)
    columns = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
        'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
        'sload', 'dload', 'spkts', 'dpkts', 'smeansz', 'dmeansz',
        'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime',
        'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
        'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
    ]
    df = pd.read_csv(filepath, names=columns, low_memory=False)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def preprocess(df: pd.DataFrame, scaler=None, encoders=None, fit=True):
    """
    Full preprocessing pipeline.

    Steps:
    1. Drop irrelevant identifier columns (IP addresses, ports as IDs)
    2. Handle missing values - UNSW-NB15 has some NaN in service/attack_cat
    3. Encode categorical features using LabelEncoder
    4. Scale numerical features using StandardScaler
    5. Return processed feature matrix X and label vector y

    Why each step:
    - Step 1: IP/port as raw strings carry no statistical meaning for ML
    - Step 2: NaN rows would cause sklearn to throw errors
    - Step 3: ML algorithms require numeric input; label encoding assigns
              integer codes to protocol/service/state strings
    - Step 4: Isolation Forest and SVM are distance-sensitive;
              unscaled features like sbytes (millions) dominate sjit (microseconds)
    """

    # Step 1: Keep only our defined feature columns + label
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    label_col = 'label' if 'label' in df.columns else None

    X = df[available_features].copy()
    y = df[label_col].copy() if label_col else None

    # Step 2: Handle missing values
    # Fill numeric NaN with column median (robust to outliers)
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Fill categorical NaN with mode (most frequent value)
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')

    # Step 3: Encode categorical columns
    if encoders is None:
        encoders = {}

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in X.columns]
    for col in cat_cols_present:
        X[col] = X[col].astype(str)
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                # Handle unseen labels gracefully
                known_classes = set(le.classes_)
                X[col] = X[col].apply(
                    lambda v: le.transform([v])[0] if v in known_classes else -1
                )
            else:
                X[col] = 0

    # Step 4: Scale numerical features
    num_cols_present = [c for c in X.columns if c not in cat_cols_present]
    if fit:
        if scaler is None:
            scaler = StandardScaler()
        X[num_cols_present] = scaler.fit_transform(X[num_cols_present])
    else:
        if scaler:
            # Only scale columns the scaler was trained on
            cols_to_scale = [c for c in num_cols_present if c in X.columns]
            X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    # Convert label: UNSW-NB15 label is already 0/1
    if y is not None:
        y = y.astype(int)

    return X, y, scaler, encoders


def get_feature_names():
    """Return the expected feature columns for API input validation."""
    return FEATURE_COLUMNS


def make_sample_record():
    """
    Generate a synthetic sample record for testing the API.
    Values are typical of a normal TCP connection.
    """
    return {
        'dur': 0.121478,
        'proto': 'tcp',
        'service': 'http',
        'state': 'FIN',
        'spkts': 6,
        'dpkts': 4,
        'sbytes': 258,
        'dbytes': 172,
        'sttl': 64,
        'dttl': 252,
        'sloss': 0,
        'dloss': 0,
        'sload': 16980.33,
        'dload': 11320.22,
        'smeansz': 43,
        'dmeansz': 43,
        'trans_depth': 1,
        'res_bdy_len': 0,
        'sjit': 0.0,
        'djit': 0.0,
        'stime': 1421927011,
        'ltime': 1421927011,
        'sintpkt': 24.2956,
        'dintpkt': 0.0,
        'tcprtt': 0.0,
        'synack': 0.0,
        'ackdat': 0.0,
        'is_sm_ips_ports': 0,
        'ct_state_ttl': 2,
        'ct_flw_http_mthd': 0,
        'is_ftp_login': 0,
        'ct_ftp_cmd': 0,
        'ct_srv_src': 1,
        'ct_srv_dst': 1,
        'ct_dst_ltm': 1,
        'ct_src_ltm': 1,
        'ct_src_dport_ltm': 1,
        'ct_dst_sport_ltm': 1,
        'ct_dst_src_ltm': 1
    }