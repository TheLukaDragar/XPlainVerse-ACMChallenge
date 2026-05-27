#!/usr/bin/env python3
"""Unified evaluation script for the 7 VFM baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from models import LOADERS, MODEL_SPECS, canonical_model_name, default_checkpoint_path, load_model

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


class BinaryFolderDataset(Dataset):
    def __init__(self, real_dir: str, fake_dir: str, transform, max_samples: int | None = None):
        self.transform = transform
        real_paths = self._get_image_files(real_dir)
        fake_paths = self._get_image_files(fake_dir)
        if max_samples is not None:
            real_paths = real_paths[:max_samples]
            fake_paths = fake_paths[:max_samples]
        self.image_paths = real_paths + fake_paths
        self.labels = [0] * len(real_paths) + [1] * len(fake_paths)

    @staticmethod
    def _get_image_files(folder: str):
        folder = Path(folder)
        images = []
        for extension in IMAGE_EXTENSIONS:
            images.extend(folder.rglob(f"*{extension}"))
        return sorted(images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), self.labels[index], str(image_path)


def evaluate(model, transform, real_dir: str, fake_dir: str, batch_size: int, num_workers: int, max_samples: int | None):
    dataset = BinaryFolderDataset(real_dir, fake_dir, transform, max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = next(model.parameters()).device
    y_true = []
    y_prob = []
    y_pred = []
    paths = []

    with torch.no_grad():
        for images, labels, batch_paths in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.numpy().tolist())
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            paths.extend(batch_paths)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "real_accuracy": float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])),
        "fake_accuracy": float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["ap"] = float(average_precision_score(y_true, y_prob))

    samples = [
        {
            "path": path,
            "label": int(label),
            "prob_fake": float(prob),
            "pred": int(pred),
        }
        for path, label, prob, pred in zip(paths, y_true, y_prob, y_pred)
    ]
    return {"metrics": metrics, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="One of: all, metacliplin, metaclip2lin, sigliplin, siglip2lin, pelin, dinov2lin, dinov3lin")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path for single-model evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    model_names = list(LOADERS.keys()) if args.model == "all" else [canonical_model_name(args.model)]
    results = {}

    for model_name in model_names:
        checkpoint = args.checkpoint if args.model != "all" and args.checkpoint else default_checkpoint_path(model_name)
        checkpoint = Path(checkpoint)
        try:
            checkpoint_for_output = str(checkpoint.relative_to(Path(__file__).resolve().parent))
        except ValueError:
            checkpoint_for_output = str(checkpoint)
        model, transform = load_model(model_name, checkpoint_path=checkpoint, device=args.device)
        result = evaluate(
            model=model,
            transform=transform,
            real_dir=args.real_dir,
            fake_dir=args.fake_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )
        results[model_name] = {
            "paper_name": MODEL_SPECS[model_name]["paper_name"],
            "checkpoint": checkpoint_for_output,
            **result,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = json.dumps(results, indent=2, ensure_ascii=False)
    print(output)

    if args.save_json:
        Path(args.save_json).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
