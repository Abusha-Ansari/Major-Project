"""
Explainable AI (XAI) Module — Grad-CAM for Medical Chest Radiography
────────────────────────────────────────────────────────────────────
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
for visualizing region importance in CNN architectures (e.g., ResNet-18).
"""

import base64
import io
from typing import Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLASS_NAMES
from data_utils import eval_transforms


class GradCAM:
    """Grad-CAM extractor for PyTorch convolutional neural networks."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Default to layer4 for standard ResNet architectures
        if target_layer is None:
            if hasattr(model, "layer4"):
                self.target_layer = model.layer4[-1]
            else:
                raise ValueError("Target layer must be specified for non-ResNet architectures.")
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where the first element is the gradient w.r.t layer output
            self.gradients = grad_out[0].detach()

        self._handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self._handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handler in self._handlers:
            handler.remove()
        self._handlers = []

    def __call__(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, torch.Tensor]:
        """
        Generate the 2D Grad-CAM heatmap mask normalized between 0.0 and 1.0.
        
        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_class: Specific class index to explain. If None, uses argmax class.

        Returns:
            heatmap (np.ndarray): 2D array of shape (H, W) in range [0, 1]
            predicted_class (int): Predicted or target class index
            confidence (float): Probability score of the class
            probabilities (torch.Tensor): Softmax probabilities across all classes
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)[0]

        if target_class is None:
            target_class = int(probabilities.argmax().item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        # gradients shape: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        confidence = float(probabilities[target_class].item())
        return heatmap, target_class, confidence, probabilities.detach().cpu()


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Tuple[Image.Image, Image.Image]:
    """
    Overlay Grad-CAM heatmap onto the original PIL Image.

    Args:
        original_image: PIL Image (RGB or Grayscale)
        heatmap: 2D numpy array [0, 1]
        alpha: Blend ratio for heatmap (0.0 = original, 1.0 = heatmap only)
        colormap_name: Matplotlib colormap ('jet', 'inferno', 'turbo', etc.)

    Returns:
        blended_image: PIL Image with overlaid heatmap
        standalone_heatmap: PIL Image of the colorized heatmap alone
    """
    orig_rgb = original_image.convert("RGB")
    orig_np = np.array(orig_rgb)
    orig_h, orig_w = orig_np.shape[:2]

    # Resize heatmap to match original image dimensions
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), resample=Image.Resampling.BILINEAR
    )
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Apply colormap
    try:
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
    except Exception:
        cmap = getattr(cm, "get_cmap", lambda name: cm.jet)(colormap_name)

    colored_cam = cmap(heatmap_resized)[:, :, :3]  # drop alpha channel -> shape (H, W, 3) in [0, 1]
    colored_cam_uint8 = (colored_cam * 255).astype(np.uint8)

    # Blend original and colored heatmap
    blended = (1.0 - alpha) * orig_np.astype(np.float32) + alpha * colored_cam_uint8.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended), Image.fromarray(colored_cam_uint8)


def pil_to_base64(pil_img: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64 encoded data URI."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def explain_image(
    model: torch.nn.Module,
    image_input: Union[str, Image.Image],
    device: torch.device,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> dict:
    """
    High-level explainability pipeline for a single chest X-ray image.

    Returns dictionary containing:
        - prediction: str
        - confidence: float
        - probabilities: list of {class_name, probability}
        - overlay_base64: str (data URI)
        - heatmap_base64: str (data URI)
        - overlay_pil: Image.Image
        - heatmap_pil: Image.Image
    """
    if isinstance(image_input, str):
        orig_image = Image.open(image_input).convert("RGB")
    else:
        orig_image = image_input.convert("RGB")

    transform = eval_transforms()
    tensor = transform(orig_image).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    try:
        heatmap, pred_idx, confidence, probs = gradcam(tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()

    overlay_img, heatmap_img = overlay_heatmap(
        orig_image, heatmap, alpha=alpha, colormap_name=colormap_name
    )

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": [
            {"class_name": cname, "probability": float(probs[i])}
            for i, cname in enumerate(CLASS_NAMES)
        ],
        "overlay_base64": pil_to_base64(overlay_img),
        "heatmap_base64": pil_to_base64(heatmap_img),
        "overlay_pil": overlay_img,
        "heatmap_pil": heatmap_img,
    }
