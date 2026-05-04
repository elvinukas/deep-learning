from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


CLASSES = ["Tree", "Tomato", "Pizza"]
DEFAULT_THRESHOLD = 0.8
IMAGE_SIZE = (128, 128)

class ClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.steps = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.steps(x)


def build_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])


def load_model(checkpoint_path: str | Path | None = None, device: str | torch.device | None = None):
    device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = ClassificationModel(num_classes=len(CLASSES)).to(device)

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
        model.load_state_dict(state)
    else:
        print("[warning] No checkpoint provided. The model is randomly initialized, so predictions will not be meaningful.")

    model.eval()
    return model, device


def predict_image(image: Image.Image, model: nn.Module, device: torch.device, threshold: float = DEFAULT_THRESHOLD):
    transform = build_transform()
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    results = []
    for label, score in zip(CLASSES, probs):
        if score >= threshold:
            results.append({"label": label, "score": float(score)})

    if not results:
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        results = [{"label": CLASSES[best_idx], "score": float(probs[best_idx])}]

    results.sort(key=lambda x: x["score"], reverse=True)
    return results, probs


def labels_to_text(results: Iterable[dict]) -> str:
    return ", ".join(f'{item["label"]} ({item["score"]:.2f})' for item in results)
