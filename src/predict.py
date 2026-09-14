import argparse
from pathlib import Path

import torch
from PIL import Image

from config import CLASS_NAMES
from data_utils import eval_transforms
from model import build_model, load_checkpoint
from xai import explain_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict pneumonia from one X-ray image with Grad-CAM XAI.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--xai-out", default=None, help="Optional output path to save the Grad-CAM heatmap overlay image.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap blend alpha transparency (default 0.5).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = build_model(pretrained=False).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")

    if args.xai_out:
        result = explain_image(model, image, device, alpha=args.alpha)
        predicted_class = result["prediction"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        out_path = Path(args.xai_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result["overlay_pil"].save(out_path)
        print(f"Saved Grad-CAM XAI visualization to: {out_path}")
    else:
        tensor = eval_transforms()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())

        predicted_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()
        probabilities = [
            {"class_name": cname, "probability": float(probs[i])}
            for i, cname in enumerate(CLASS_NAMES)
        ]

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    for item in probabilities:
        print(f"{item['class_name']}: {item['probability']:.4f}")


if __name__ == "__main__":
    main()

