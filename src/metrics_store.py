"""
Thread-safe JSON persistence layer for federated-learning training metrics.

Schema of server_storage/metrics.json
--------------------------------------
{
  "baseline": {
      "accuracy":  float,
      "loss":      float,
      "samples":   int,
      "timestamp": int (Unix epoch),
      "label":     "Baseline"
  } | null,

  "rounds": [
      {
          "round_num":     int,
          "round_id":      str,
          "accuracy":      float,
          "loss":          float,
          "num_hospitals": int,
          "total_samples": int,
          "timestamp":     int
      },
      ...
  ]
}
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT

METRICS_PATH = PROJECT_ROOT / "server_storage" / "metrics.json"

_lock = threading.Lock()


# ── internal helpers ──────────────────────────────────────────────────────────

def _load_raw() -> dict:
    """Load the metrics file; return an empty skeleton on any error."""
    if not METRICS_PATH.exists():
        return {"baseline": None, "rounds": []}
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"baseline": None, "rounds": []}


def _save_raw(data: dict) -> None:
    """Atomically write *data*: write to a .tmp file then rename over target."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = METRICS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(METRICS_PATH)


# ── public API ────────────────────────────────────────────────────────────────

def get_all_metrics() -> dict:
    """Return the full metrics dict (thread-safe read)."""
    with _lock:
        return _load_raw()


def save_baseline_metrics(
    accuracy: float,
    loss: float,
    samples: int = 0,
) -> dict:
    """
    Persist *accuracy* and *loss* as the baseline (initial model) entry.
    Overwrites any previous baseline.  Returns the saved entry.
    """
    with _lock:
        data = _load_raw()
        entry: dict = {
            "accuracy": round(float(accuracy), 6),
            "loss": round(float(loss), 6),
            "samples": int(samples),
            "timestamp": int(time.time()),
            "label": "Baseline",
        }
        data["baseline"] = entry
        _save_raw(data)
        return entry


def save_round_metrics(
    round_id: str,
    accuracy: float,
    loss: float,
    num_hospitals: int,
    total_samples: int,
    timestamp: Optional[int] = None,
) -> dict:
    """
    Append one federated round's evaluation results to the rounds list.
    ``round_num`` is auto-assigned as (current length + 1).
    Returns the saved entry.
    """
    with _lock:
        data = _load_raw()
        if "rounds" not in data:
            data["rounds"] = []
        round_num = len(data["rounds"]) + 1
        entry: dict = {
            "round_num": round_num,
            "round_id": round_id,
            "accuracy": round(float(accuracy), 6),
            "loss": round(float(loss), 6),
            "num_hospitals": int(num_hospitals),
            "total_samples": int(total_samples),
            "timestamp": int(timestamp or time.time()),
        }
        data["rounds"].append(entry)
        _save_raw(data)
        return entry


def clear_metrics() -> None:
    """Reset everything (useful when starting a fresh experiment)."""
    with _lock:
        _save_raw({"baseline": None, "rounds": []})
