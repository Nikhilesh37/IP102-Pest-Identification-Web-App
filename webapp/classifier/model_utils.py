from pathlib import Path
from typing import BinaryIO

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
CLASS_FILE = PROJECT_ROOT / "data" / "dataset" / "classes.txt"
WEIGHTS_PATH = PROJECT_ROOT / "artifacts" / "results" / "pest_resnet50_weights.pth"
FULL_MODEL_PATH = PROJECT_ROOT / "artifacts" / "results" / "pest_resnet50_full_model.pth"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


class PestPredictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names(CLASS_FILE)
        self.model = self._build_model(len(self.class_names))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    def _build_model(self, num_classes: int) -> nn.Module:
        if FULL_MODEL_PATH.exists():
            model = torch.load(FULL_MODEL_PATH, map_location="cpu", weights_only=False)
            if isinstance(model, nn.Module):
                return model.to(self.device)

        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=True)

        return model.to(self.device)

    @staticmethod
    def _load_class_names(class_file: Path) -> list[str]:
        class_names: list[str] = []
        with class_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    class_names.append(parts[1].strip())
                else:
                    class_names.append(parts[0].strip())
        return class_names

    def predict(self, file_obj: BinaryIO, top_k: int = 5) -> dict:
        image = Image.open(file_obj).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probs, k=k)

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            class_name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            results.append({"class_name": class_name, "probability": prob})

        return {"top1": results[0], "topk": results}


_predictor: PestPredictor | None = None


def get_predictor() -> PestPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PestPredictor()
    return _predictor
