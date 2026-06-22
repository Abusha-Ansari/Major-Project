import argparse
from pathlib import Path

import torch
from torch import nn

from config import DEFAULT_BATCH_SIZE, DEFAULT_LR, MODELS_DIR
from data_utils import make_loader
from model import build_model, load_checkpoint, save_checkpoint
from train_utils import evaluate, fedavg, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run local-folder federated learning for X-ray pneumonia.")
    parser.add_argument("--hospitals-dir", required=True, help="Directory containing hospital_* folders.")
    parser.add_argument("--test-dir", required=True, help="Global test directory in ImageFolder format.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--global-model", default=str(MODELS_DIR / "xray_global.pt"))
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def list_hospitals(hospitals_dir):
    hospitals = sorted([path for path in Path(hospitals_dir).iterdir() if path.is_dir()])
    if not hospitals:
        raise FileNotFoundError(f"No hospital folders found in {hospitals_dir}")
    return hospitals


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    global_model = build_model(pretrained=True).to(device)
    global_model_path = Path(args.global_model)
    if global_model_path.exists():
        load_checkpoint(global_model, str(global_model_path), device)
        print(f"Loaded existing global model: {global_model_path}")
    else:
        print("No global model found. Starting from ImageNet-pretrained ResNet18.")

    hospitals = list_hospitals(args.hospitals_dir)
    test_loader, _ = make_loader(args.test_dir, batch_size=args.batch_size, train=False)

    for round_idx in range(1, args.rounds + 1):
        print(f"\nFederated round {round_idx}/{args.rounds}")
        client_states = []
        sample_counts = []

        for hospital_dir in hospitals:
            print(f"Training client: {hospital_dir.name}")
            local_model = build_model(pretrained=False).to(device)
            local_model.load_state_dict(global_model.state_dict())

            train_loader, train_dataset = make_loader(hospital_dir, batch_size=args.batch_size, train=True)
            train_model(local_model, train_loader, device=device, epochs=args.local_epochs, lr=args.lr)

            client_states.append({k: v.detach().cpu() for k, v in local_model.state_dict().items()})
            sample_counts.append(len(train_dataset))
            print(f"{hospital_dir.name} contributed {len(train_dataset)} samples")

        averaged_state = fedavg(client_states, sample_counts)
        global_model.load_state_dict(averaged_state)
        global_model.to(device)

        metrics = evaluate(global_model, test_loader, nn.CrossEntropyLoss(), device)
        print(
            f"round={round_idx} test_loss={metrics['loss']:.4f} "
            f"test_accuracy={metrics['accuracy']:.4f}"
        )

        global_model_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(global_model, str(global_model_path))
        print(f"Saved global model to {global_model_path}")


if __name__ == "__main__":
    main()
