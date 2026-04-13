"""
main.py
-------
FastAPI backend for the Anomaly-Based Intrusion Detection System.

API DESIGN RATIONALE:
---------------------
/predict        - Single record inference. Real-time use case: a network
                  sensor sends one connection's features and gets an
                  immediate verdict. Low latency, immediate response.

/batch_predict  - Bulk CSV upload. Offline analysis use case: a network
                  administrator uploads yesterday's PCAP-derived CSV and
                  gets a report of all suspicious connections.

/metrics        - Returns stored model performance metrics. Useful for
                  monitoring model drift over time.

/history        - Returns recent prediction logs. Enables the frontend
                  trend visualization without a database.

/simulate       - Generates synthetic traffic and runs predictions.
                  Used for frontend live-mode demo.

/threshold      - Allows tuning the anomaly score cutoff without retraining.
                  Critical for reducing false positives in production.
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import io
import webbrowser
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from utils import preprocess, FEATURE_COLUMNS, make_sample_record
from logger import init_logger, log_prediction, get_recent_logs

# -------------------------
# APP INITIALIZATION
# -------------------------
app = FastAPI(
    title="AIDS - Anomaly-Based Intrusion Detection System",
    description="Classical ML-powered network intrusion detection using Isolation Forest and One-Class SVM",
    version="1.0.0"
)

# -------------------------
# SERVE FRONTEND
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend")

# Mount static files (for JS/CSS if added later)
app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_PATH, "index.html"))

# CORS: Allow the frontend (any origin in dev) to call the API.
# In production, replace "*" with your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# MODEL LOADING
# -------------------------
MODEL_DIR = './models/'

def load_artifact(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# Load models at startup (not per-request — expensive operation)
iforest = load_artifact('isolation_forest.pkl')
ocsvm = load_artifact('one_class_svm.pkl')
scaler = load_artifact('scaler.pkl')
encoders = load_artifact('encoders.pkl')

# Load saved feature list
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.json')
if os.path.exists(feature_names_path):
    with open(feature_names_path) as f:
        TRAINED_FEATURES = json.load(f)
else:
    TRAINED_FEATURES = None

# Load metrics
metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
STORED_METRICS = {}
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        STORED_METRICS = json.load(f)

# Global threshold for anomaly score (tunable at runtime)
# Isolation Forest score_samples returns negative values:
# more negative = more anomalous. Default threshold: -0.1
# Scores below this are classified as anomalies.
ANOMALY_THRESHOLD = -0.1

init_logger()

# -------------------------
# PYDANTIC MODELS (Request/Response Schemas)
# -------------------------
class TrafficRecord(BaseModel):
    """
    Schema for a single network traffic record.
    All UNSW-NB15 features are included.
    Example values represent a typical HTTP connection.
    """
    dur: float = Field(0.121, description="Duration of connection in seconds")
    proto: str = Field("tcp", description="Protocol: tcp, udp, icmp, etc.")
    service: str = Field("http", description="Service: http, ftp, dns, etc.")
    state: str = Field("FIN", description="State: FIN, INT, CON, REQ, etc.")
    spkts: int = Field(6, description="Source-to-dest packet count")
    dpkts: int = Field(4, description="Dest-to-source packet count")
    sbytes: int = Field(258, description="Source-to-dest bytes")
    dbytes: int = Field(172, description="Dest-to-source bytes")
    sttl: int = Field(64, description="Source TTL value")
    dttl: int = Field(252, description="Destination TTL value")
    sloss: int = Field(0, description="Source packet retransmission count")
    dloss: int = Field(0, description="Dest packet retransmission count")
    sload: float = Field(16980.0, description="Source bits per second")
    dload: float = Field(11320.0, description="Dest bits per second")
    smeansz: float = Field(43.0, description="Mean packet size src->dst")
    dmeansz: float = Field(43.0, description="Mean packet size dst->src")
    trans_depth: int = Field(1, description="HTTP transaction depth")
    res_bdy_len: int = Field(0, description="Response body length")
    sjit: float = Field(0.0, description="Source jitter (ms)")
    djit: float = Field(0.0, description="Dest jitter (ms)")
    stime: float = Field(1421927011.0, description="Start time (epoch)")
    ltime: float = Field(1421927011.0, description="Last time (epoch)")
    sintpkt: float = Field(24.3, description="Source inter-packet time")
    dintpkt: float = Field(0.0, description="Dest inter-packet time")
    tcprtt: float = Field(0.0, description="TCP round-trip time")
    synack: float = Field(0.0, description="Time between SYN and SYN-ACK")
    ackdat: float = Field(0.0, description="Time between SYN-ACK and ACK")
    is_sm_ips_ports: int = Field(0, description="1 if src/dst IP and port are equal")
    ct_state_ttl: int = Field(2, description="State-TTL combination count")
    ct_flw_http_mthd: int = Field(0, description="HTTP method count")
    is_ftp_login: int = Field(0, description="1 if FTP login attempted")
    ct_ftp_cmd: int = Field(0, description="FTP command count")
    ct_srv_src: int = Field(1, description="Connections with same service from source")
    ct_srv_dst: int = Field(1, description="Connections with same service to dest")
    ct_dst_ltm: int = Field(1, description="Connections to same dest in last time")
    ct_src_ltm: int = Field(1, description="Connections from same source in last time")
    ct_src_dport_ltm: int = Field(1, description="Same src and dst port in last time")
    ct_dst_sport_ltm: int = Field(1, description="Same dst and src port in last time")
    ct_dst_src_ltm: int = Field(1, description="Same src/dst pair in last time")


class PredictionResponse(BaseModel):
    prediction: str          # "normal" or "anomaly"
    anomaly_score: float     # Raw score from Isolation Forest
    confidence: float        # Normalized confidence 0-1
    model_used: str
    threshold: float
    timestamp: str


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=-1.0, le=0.0,
                              description="Anomaly score threshold (-1.0 to 0.0). "
                                          "More negative = stricter (fewer false positives, more false negatives).")


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def record_to_dataframe(record: TrafficRecord) -> pd.DataFrame:
    """Convert a Pydantic TrafficRecord to a single-row DataFrame."""
    data = record.dict()
    return pd.DataFrame([data])


def run_prediction(df_raw: pd.DataFrame, model_name: str = 'isolation_forest'):
    """
    Preprocess a DataFrame and run prediction.
    Returns (prediction_label, raw_score, confidence).
    """
    global ANOMALY_THRESHOLD

    if iforest is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run model_trainer.py first."
        )

    # Preprocess using saved scaler and encoders (fit=False for inference)
    X, _, _, _ = preprocess(df_raw, scaler=scaler, encoders=encoders, fit=False)

    # Align columns to training feature order
    if TRAINED_FEATURES:
        for col in TRAINED_FEATURES:
            if col not in X.columns:
                X[col] = 0  # Fill missing features with 0
        X = X[TRAINED_FEATURES]

    if model_name == 'isolation_forest':
        model = iforest
    elif model_name == 'one_class_svm':
        model = ocsvm
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    # score_samples: lower score = more anomalous
    scores = model.score_samples(X)
    score = float(scores[0])

    # Decision: if score is below threshold, it's an anomaly
    is_anomaly = score < ANOMALY_THRESHOLD
    prediction = "anomaly" if is_anomaly else "normal"

    # Normalize score to a 0-1 confidence
    # Typical IF scores range roughly from -0.6 to 0.1
    # We map this range to [0,1] for the frontend
    score_min, score_max = -0.6, 0.1
    normalized = (score - score_min) / (score_max - score_min)
    confidence = float(np.clip(normalized, 0.0, 1.0))
    # Confidence = 1.0 means very likely normal; 0.0 means very likely anomaly
    # For display, flip if anomaly so confidence represents certainty of label
    if is_anomaly:
        confidence = 1.0 - confidence

    return prediction, score, confidence


# -------------------------
# ENDPOINTS
# -------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "service": "AIDS - Anomaly-Based Intrusion Detection System",
        "status": "running",
        "models_loaded": iforest is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    record: TrafficRecord,
    model: str = Query(default="isolation_forest",
                       enum=["isolation_forest", "one_class_svm"])
):
    """
    Predict whether a single network traffic record is normal or an anomaly.

    Input: JSON object with network flow features.
    Output: prediction label, anomaly score, and confidence.

    The anomaly_score is the raw Isolation Forest score_samples value.
    More negative = more anomalous. Threshold is configurable via /threshold.
    """
    df_raw = record_to_dataframe(record)
    prediction, score, confidence = run_prediction(df_raw, model)

    # Log the prediction for trend analysis
    log_prediction(
        model=model,
        prediction=prediction,
        anomaly_score=score,
        threshold=ANOMALY_THRESHOLD,
        source='single'
    )

    return PredictionResponse(
        prediction=prediction,
        anomaly_score=round(score, 6),
        confidence=round(confidence, 4),
        model_used=model,
        threshold=ANOMALY_THRESHOLD,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(
    file: UploadFile = File(...),
    model: str = Query(default="isolation_forest",
                       enum=["isolation_forest", "one_class_svm"])
):
    """
    Batch prediction from an uploaded CSV file.

    Input: CSV file with the same columns as the /predict endpoint.
    Output: List of predictions with anomaly scores.

    Use case: Upload a day's worth of network flow logs and get
    a full report of suspicious connections.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    contents = await file.read()
    df_raw = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    if len(df_raw) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit is 10,000 records per request."
        )

    results = []
    for i, row in df_raw.iterrows():
        try:
            row_df = pd.DataFrame([row.to_dict()])
            prediction, score, confidence = run_prediction(row_df, model)
            results.append({
                'index': int(i),
                'prediction': prediction,
                'anomaly_score': round(score, 6),
                'confidence': round(confidence, 4)
            })
            log_prediction(model, prediction, score, ANOMALY_THRESHOLD, 'batch')
        except Exception as e:
            results.append({
                'index': int(i),
                'error': str(e)
            })

    anomaly_count = sum(1 for r in results if r.get('prediction') == 'anomaly')
    normal_count = sum(1 for r in results if r.get('prediction') == 'normal')

    return {
        'total_records': len(results),
        'anomaly_count': anomaly_count,
        'normal_count': normal_count,
        'anomaly_rate': round(anomaly_count / len(results), 4) if results else 0,
        'model_used': model,
        'predictions': results
    }


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """
    Return stored model evaluation metrics from training.
    Includes precision, recall, F1, and false positive rate
    for both Isolation Forest and One-Class SVM.
    """
    if not STORED_METRICS:
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Run model_trainer.py first."
        )
    return STORED_METRICS


@app.get("/history", tags=["Monitoring"])
def get_history(n: int = Query(default=100, ge=1, le=1000)):
    """
    Return the last n prediction log entries.
    Used by the frontend to render the anomaly trend chart.
    """
    logs = get_recent_logs(n)
    return {'count': len(logs), 'logs': logs}


@app.post("/threshold", tags=["Configuration"])
def update_threshold(update: ThresholdUpdate):
    """
    Update the anomaly score threshold without retraining.

    This is basic threshold tuning: the boundary between normal and anomaly
    is shifted. Lower (more negative) threshold = more conservative,
    fewer false positives but more false negatives (missed attacks).
    Higher threshold = more sensitive, more false positives.

    Typical tuning workflow:
    1. Deploy model with default threshold
    2. Review false positives in /history logs
    3. Adjust threshold until FPR meets your SLA
    """
    global ANOMALY_THRESHOLD
    old = ANOMALY_THRESHOLD
    ANOMALY_THRESHOLD = update.threshold
    return {
        'previous_threshold': old,
        'new_threshold': ANOMALY_THRESHOLD,
        'effect': 'More negative = stricter (fewer FP, more FN). '
                  'Less negative = more sensitive (more FP, fewer FN).'
    }


@app.get("/simulate", tags=["Simulation"])
def simulate(
    n: int = Query(default=20, ge=1, le=200),
    attack_ratio: float = Query(default=0.3, ge=0.0, le=1.0),
    model: str = Query(default="isolation_forest",
                       enum=["isolation_forest", "one_class_svm"])
):
    """
    Simulate n network traffic records with a given attack_ratio and predict.

    Used by the frontend's live simulation mode.

    attack_ratio: fraction of generated records that simulate attack patterns.
    Attack simulation: spikes in packet counts, extreme byte counts,
    unusual TTL values, or ports typically associated with scans.
    """
    results = []
    for _ in range(n):
        record = make_sample_record()

        # With probability attack_ratio, distort the record to look attack-like
        if random.random() < attack_ratio:
            # Port scan signature: many packets, little data, low TTL
            record['spkts'] = random.randint(500, 5000)
            record['sbytes'] = random.randint(0, 100)
            record['sttl'] = random.randint(1, 10)
            record['dur'] = random.uniform(0.0, 0.001)
            record['ct_srv_src'] = random.randint(50, 200)
            record['sloss'] = random.randint(100, 500)
            record['sjit'] = random.uniform(100.0, 999.0)
            is_attack = True
        else:
            # Normal: slight random variation around typical values
            record['spkts'] = random.randint(2, 20)
            record['sbytes'] = random.randint(100, 1000)
            record['sttl'] = random.randint(60, 128)
            record['dur'] = random.uniform(0.01, 2.0)
            is_attack = False

        row_df = pd.DataFrame([record])
        try:
            prediction, score, confidence = run_prediction(row_df, model)
        except Exception:
            prediction, score, confidence = 'normal', 0.0, 0.5

        results.append({
            'simulated_as': 'attack' if is_attack else 'normal',
            'prediction': prediction,
            'anomaly_score': round(score, 6),
            'confidence': round(confidence, 4),
            'timestamp': datetime.utcnow().isoformat()
        })

        log_prediction(model, prediction, score, ANOMALY_THRESHOLD, 'simulation')
        time.sleep(0.01)  # Slight delay to create realistic timestamps in logs

    anomaly_count = sum(1 for r in results if r['prediction'] == 'anomaly')
    correct = sum(
        1 for r in results
        if (r['simulated_as'] == 'attack') == (r['prediction'] == 'anomaly')
    )

    return {
        'total': n,
        'anomalies_detected': anomaly_count,
        'simulation_accuracy': round(correct / n, 4),
        'model_used': model,
        'records': results
    }


@app.get("/sample_record", tags=["Utilities"])
def sample_record():
    """Return a sample input record for testing the /predict endpoint."""
    return make_sample_record()


# -------------------------
# RUN
# -------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)