"""Quality-agnostic augmentations from Bombek1 ai-image-detector README."""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class QualityAgnosticAugment:
    """JPEG / blur / noise / resize / color jitter / flip (train only)."""

    def __init__(self, p_jpeg: float = 0.5, p_blur: float = 0.3, p_noise: float = 0.3, p_resize: float = 0.3):
        self.p_jpeg = p_jpeg
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.p_resize = p_resize

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")

        if random.random() < self.p_jpeg:
            quality = random.randint(30, 95)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        if random.random() < self.p_blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))

        if random.random() < self.p_noise:
            arr = np.asarray(image).astype(np.float32) / 255.0
            sigma = random.uniform(0.0, 0.05)
            arr = np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0.0, 1.0)
            image = Image.fromarray((arr * 255).astype(np.uint8))

        if random.random() < self.p_resize:
            scale = random.uniform(0.5, 1.0)
            w, h = image.size
            small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
            image = small.resize((w, h), Image.BICUBIC)

        if random.random() < 0.8:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image
