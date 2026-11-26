import copy
import sys
from collections import OrderedDict

import torch
from torch import nn


def get_progress_bar(loader, desc=""):
    """Returns tqdm progress bar if a valid stderr stream is available, otherwise raw loader."""
    if sys.stderr is None or getattr(sys, "frozen", False):
        return loader
    try:
        from tqdm import tqdm
        return tqdm(loader, desc=desc, leave=False)
    except Exception:
        return loader


def safe_print(msg):
    if sys.stdout is not None:
        try:
            print(msg)
        except Exception:
            pass


def train_one_epoch(model, loader, optimizer, criterion, device, on_batch=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(get_progress_bar(loader, desc="train")):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if on_batch is not None:
            on_batch(
                batch=batch_idx + 1,
                total_batches=total_batches,
                loss=total_loss / max(total, 1),
                accuracy=correct / max(total, 1),
            )

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "samples": total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in get_progress_bar(loader, desc="eval"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "samples": total,
    }



def train_model(model, train_loader, device, epochs, lr, on_epoch=None, on_batch=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, on_batch=on_batch
        )
        history.append(metrics)
        safe_print(
            f"epoch={epoch} loss={metrics['loss']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} samples={metrics['samples']}"
        )
        if on_epoch is not None:
            on_epoch(epoch=epoch, total_epochs=epochs, metrics=metrics)

    return history


def copy_model_weights(model):
    return copy.deepcopy(model.state_dict())


def fedavg(state_dicts, sample_counts):
    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError("FedAvg requires at least one sample.")

    averaged = OrderedDict()
    for key in state_dicts[0].keys():
        if not state_dicts[0][key].is_floating_point():
            averaged[key] = state_dicts[0][key]
            continue

        weighted_sum = None
        for state_dict, count in zip(state_dicts, sample_counts):
            value = state_dict[key].float() * (count / total_samples)
            weighted_sum = value if weighted_sum is None else weighted_sum + value
        averaged[key] = weighted_sum

    return averaged
