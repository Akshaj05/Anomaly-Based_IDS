"""
logger.py
---------
Logs every prediction to a CSV file for audit, review,
threshold tuning, and trend analysis.
"""

import csv
import os
from datetime import datetime

LOG_FILE = './logs/predictions.csv'
LOG_HEADERS = [
    'timestamp', 'model', 'prediction', 'anomaly_score',
    'threshold_used', 'source'
]


def init_logger():
    """Create log directory and file with headers if not exists."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
            writer.writeheader()


def log_prediction(model: str, prediction: str, anomaly_score: float,
                   threshold: float, source: str = 'api'):
    """Append a single prediction record to the log file."""
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        writer.writerow({
            'timestamp': datetime.utcnow().isoformat(),
            'model': model,
            'prediction': prediction,
            'anomaly_score': round(anomaly_score, 6),
            'threshold_used': threshold,
            'source': source
        })


def get_recent_logs(n: int = 100) -> list:
    """Return the last n log entries as a list of dicts."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-n:]