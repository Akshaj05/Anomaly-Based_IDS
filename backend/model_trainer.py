"""
WHY ISOLATION FOREST for anomaly detection:
Standard classifiers need labelled attack examples to learn from.
In real networks, attack traffic is rare and constantly evolving —
you cannot label everything. Isolation Forest is unsupervised:
it learns what "normal" looks like, then flags anything that
deviates significantly.

Mathematical Intuition (Isolation Forest):\
The algorithm builds an ensemble of random binary trees.
For each sample, it randomly selects a feature, then randomly
selects a split value between that feature's min and max.

Anomaly score formula:
    s(x, n) = 2^(-E(h(x)) / c(n))

Where:
    h(x)   = average path length to isolate point x across all trees
    c(n)   = expected path length for n samples (normalization constant)
    c(n)   = 2*H(n-1) - (2*(n-1)/n)  where H is harmonic number

    s close to 1.0  -> likely anomaly (isolated quickly)
    s close to 0.0  -> likely normal (took many splits to isolate)
    s close to 0.5  -> ambiguous

WHY ONE-CLASS SVM:
------------------
One-Class SVM learns a hypersphere in feature space (via kernel trick)
that encloses the majority of normal training data.
At inference, points outside the hypersphere are anomalies.
Kernel: RBF (Radial Basis Function) transforms input into infinite-
dimensional space where a linear boundary becomes non-linear in input space.

    K(x, x') = exp(-gamma * ||x - x'||^2)

One-Class SVM is more sensitive to hyperparameters and slower on large
datasets, which is why Isolation Forest is the primary model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import joblib
import os
import json
from utils import load_unsw_nb15, preprocess

# -------------------------
# CONFIGURATION
# -------------------------
DATA_PATH = '../data/'
MODEL_DIR = './models/'
METRICS_PATH = './models/metrics.json'

# Isolation Forest is trained ONLY on normal traffic (semi-supervised).
# This mimics real deployment: you train on your known-good baseline.
TRAIN_NORMAL_SAMPLES = 50000

# contamination: the expected proportion of anomalies in the dataset.
# UNSW-NB15 has ~32% attacks, but we train on normal-only, so
# contamination is a prior for the decision boundary. 0.05 is conservative.
CONTAMINATION = 0.2

# n_estimators: number of isolation trees. Higher = more stable scores.
# 200 gives good stability without being slow.
N_ESTIMATORS = 200

# 256 is the paper's recommended value — small enough for diverse trees.
MAX_SAMPLES = 'auto'

# One-Class SVM: nu is an upper bound on the fraction of outliers.
# kernel='rbf' handles non-linear boundaries.
# gamma='scale' sets gamma=1/(n_features * X.var()) automatically.
OCSVM_NU = 0.05
OCSVM_KERNEL = 'rbf'
OCSVM_GAMMA = 'scale'


def load_train_test_data(data_path: str):

    train_path = os.path.join(data_path, 'UNSW_NB15_training-set.csv')
    test_path = os.path.join(data_path, 'UNSW_NB15_testing-set.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Training or Testing dataset not found in /data folder"
        )

    print(f"Loading training data: {train_path}")
    df_train = pd.read_csv(train_path)

    print(f"Loading testing data: {test_path}")
    df_test = pd.read_csv(test_path)

    print(f"Training samples: {len(df_train)}")
    print(f"Testing samples: {len(df_test)}")

    return df_train, df_test


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("="*60)
    print("LOADING UNSW-NB15 TRAIN & TEST DATASETS")
    print("="*60)

    # -------------------------
    # LOAD TRAIN + TEST DATA
    # -------------------------
    df_train, df_test = load_train_test_data(DATA_PATH)

    print(f"\nTraining samples: {len(df_train)}")
    print(f"Testing samples: {len(df_test)}")

    print("\nTraining label distribution:")
    print(df_train['label'].value_counts())

    print("\nTesting label distribution:")
    print(df_test['label'].value_counts())

    # -------------------------
    # STEP 1: PREPROCESS TRAIN DATA
    # -------------------------
    print("\nPreprocessing training data...")
    X_train_full, y_train_full, scaler, encoders = preprocess(df_train, fit=True)

    # Train ONLY on normal traffic (IMPORTANT)
    X_train = X_train_full[y_train_full == 0]

    print(f"\nTraining on NORMAL samples only: {len(X_train)}")

    # Save scaler and encoders
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'encoders.pkl'))
    print("Scaler and encoders saved.")

    # -------------------------
    # STEP 2: PREPROCESS TEST DATA
    # -------------------------
    print("\nPreprocessing testing data...")
    X_test, y_test, _, _ = preprocess(
        df_test,
        scaler=scaler,
        encoders=encoders,
        fit=False
    )

    print(f"Test set size: {len(X_test)}")

    # -------------------------
    # STEP 3: TRAIN ISOLATION FOREST
    # -------------------------
    print("\n" + "="*60)
    print("TRAINING ISOLATION FOREST")
    print("="*60)

    iforest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    iforest.fit(X_train)
    print("Isolation Forest trained.")

    # -------------------------
    # STEP 4: TRAIN ONE-CLASS SVM
    # -------------------------
    print("\n" + "="*60)
    print("TRAINING ONE-CLASS SVM")
    print("="*60)

    svm_train_size = min(10000, len(X_train))
    X_train_svm = X_train.sample(n=svm_train_size, random_state=42)

    ocsvm = OneClassSVM(
        nu=OCSVM_NU,
        kernel=OCSVM_KERNEL,
        gamma=OCSVM_GAMMA
    )
    ocsvm.fit(X_train_svm)
    print("One-Class SVM trained.")

    # -------------------------
    # STEP 5: EVALUATION
    # -------------------------
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    # Subsample test for speed
    eval_size = min(20000, len(X_test))
    X_eval = X_test.sample(n=eval_size, random_state=42)
    y_eval = y_test.loc[X_eval.index]

    metrics = {}

    # ---- Isolation Forest ----
    if_preds_raw = iforest.predict(X_eval)
    if_preds = (if_preds_raw == -1).astype(int)

    print("\n--- Isolation Forest ---")
    print(confusion_matrix(y_eval, if_preds))
    print(classification_report(y_eval, if_preds,
                                target_names=['Normal', 'Attack']))

    tn, fp, fn, tp = confusion_matrix(y_eval, if_preds).ravel()
    metrics['isolation_forest'] = {
        'precision': float(precision_score(y_eval, if_preds, zero_division=0)),
        'recall': float(recall_score(y_eval, if_preds, zero_division=0)),
        'f1': float(f1_score(y_eval, if_preds, zero_division=0)),
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    }

    # ---- One-Class SVM ----
    svm_eval_size = min(5000, len(X_eval))
    X_eval_svm = X_eval.sample(n=svm_eval_size, random_state=42)
    y_eval_svm = y_eval.loc[X_eval_svm.index]

    svm_preds_raw = ocsvm.predict(X_eval_svm)
    svm_preds = (svm_preds_raw == -1).astype(int)

    print("\n--- One-Class SVM ---")
    print(confusion_matrix(y_eval_svm, svm_preds))
    print(classification_report(y_eval_svm, svm_preds,
                                target_names=['Normal', 'Attack']))

    tn2, fp2, fn2, tp2 = confusion_matrix(y_eval_svm, svm_preds).ravel()
    metrics['one_class_svm'] = {
        'precision': float(precision_score(y_eval_svm, svm_preds, zero_division=0)),
        'recall': float(recall_score(y_eval_svm, svm_preds, zero_division=0)),
        'f1': float(f1_score(y_eval_svm, svm_preds, zero_division=0)),
        'false_positive_rate': float(fp2 / (fp2 + tn2)) if (fp2 + tn2) > 0 else 0.0
    }

    # -------------------------
    # STEP 6: SAVE MODELS
    # -------------------------
    joblib.dump(iforest, os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    joblib.dump(ocsvm, os.path.join(MODEL_DIR, 'one_class_svm.pkl'))

    # Save feature list
    feature_names = list(X_train.columns)
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nALL MODELS SAVED SUCCESSFULLY")
if __name__ == '__main__':
    train()