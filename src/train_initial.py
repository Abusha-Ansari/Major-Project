import argparse
from pathlib import Path

import torch
from torch import nn

from config import DEFAULT_BATCH_SIZE, DEFAULT_LR, MODELS_DIR
from data_utils import make_loader
from model import build_model, save_checkpoint
from train_utils import evaluate, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the initial X-ray pneumonia model.")
    parser.add_argument("--train-dir", required=True, help="Training directory in ImageFolder format.")
    parser.add_argument("--test-dir", required=True, help="Test directory in ImageFolder format.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--output", default=str(MODELS_DIR / "xray_global.pt"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    train_loader, train_dataset = make_loader(args.train_dir, batch_size=args.batch_size, train=True)
    test_loader, _ = make_loader(args.test_dir, batch_size=args.batch_size, train=False)
    print(f"Classes: {train_dataset.classes}")

    model = build_model(pretrained=True).to(device)
    train_model(model, train_loader, device=device, epochs=args.epochs, lr=args.lr)

    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"test_loss={metrics['loss']:.4f} test_accuracy={metrics['accuracy']:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, str(output), class_names=train_dataset.classes)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
