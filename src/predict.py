import argparse

import torch
from PIL import Image

from config import CLASS_NAMES
from data_utils import eval_transforms
from model import build_model, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pneumonia from one X-ray image.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = build_model(pretrained=False).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    tensor = eval_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())

    print(f"Prediction: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.4f}")
    for class_name, probability in zip(CLASS_NAMES, probabilities.tolist()):
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
