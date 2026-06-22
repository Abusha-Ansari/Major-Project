import torch
from torch import nn
from torchvision import models

from config import CLASS_NAMES


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def save_checkpoint(model: nn.Module, path: str, class_names=None) -> None:
    class_names = class_names or CLASS_NAMES
    checkpoint = {
        "model_state": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    return model
