"""
Full model evaluation pipeline using scikit-learn.

Computes:
  - Accuracy, Loss
  - Precision, Recall, F1-score (macro + per-class)
  - Confusion matrix (2×2 for NORMAL / PNEUMONIA)
  - Full sklearn classification_report

Also handles persistence of the evaluation report to
server_storage/evaluation_report.json.
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "server_storage" / "evaluation_report.json"

_lock = threading.Lock()


# ── Report persistence ────────────────────────────────────────────────────────

def _save_report(data: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(REPORT_PATH)


def load_report() -> Optional[dict]:
    """Load and return the last saved evaluation report, or None."""
    with _lock:
        if not REPORT_PATH.exists():
            return None
        try:
            with REPORT_PATH.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None


# ── Core evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model,
    loader,
    criterion,
    device,
    class_names: list,
    label: str = "current",
    save: bool = True,
) -> dict:
    """
    Run a full evaluation pass and return a rich report dict.

    Parameters
    ----------
    model       : trained PyTorch model (already .eval())
    loader      : DataLoader over the evaluation dataset
    criterion   : loss function (e.g. nn.CrossEntropyLoss())
    device      : torch.device
    class_names : ordered list of class names, e.g. ["NORMAL","PNEUMONIA"]
    label       : human-readable label stored in the report ("baseline"/"current")
    save        : if True, write report to REPORT_PATH

    Returns
    -------
    dict with keys: accuracy, loss, samples, precision, recall, f1,
                    confusion_matrix, per_class, report_text, label, timestamp
    """
    model.eval()

    all_preds  = []
    all_labels = []
    total_loss = 0.0
    total      = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total += labels.size(0)

    # ── sklearn metrics ───────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds).tolist()

    precision = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    recall    = float(recall_score   (all_labels, all_preds, average="macro", zero_division=0))
    f1        = float(f1_score       (all_labels, all_preds, average="macro", zero_division=0))
    accuracy  = sum(p == l for p, l in zip(all_preds, all_labels)) / max(total, 1)

    # Per-class metrics
    per_class_f1        = f1_score       (all_labels, all_preds, average=None, zero_division=0).tolist()
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    per_class_recall    = recall_score   (all_labels, all_preds, average=None, zero_division=0).tolist()

    per_class = [
        {
            "class":     class_names[i] if i < len(class_names) else str(i),
            "precision": round(per_class_precision[i], 6),
            "recall":    round(per_class_recall[i],    6),
            "f1":        round(per_class_f1[i],        6),
        }
        for i in range(len(per_class_f1))
    ]

    report_text = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
    )

    report: dict = {
        "label":            label,
        "accuracy":         round(accuracy, 6),
        "loss":             round(total_loss / max(total, 1), 6),
        "samples":          total,
        "precision":        round(precision, 6),
        "recall":           round(recall, 6),
        "f1":               round(f1, 6),
        "confusion_matrix": cm,
        "class_names":      class_names,
        "per_class":        per_class,
        "report_text":      report_text,
        "timestamp":        int(time.time()),
    }

    if save:
        with _lock:
            _save_report(report)

    return report
