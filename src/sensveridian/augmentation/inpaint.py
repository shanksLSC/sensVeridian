from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import cv2


def mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()


class LaMAInpainter:
    def __init__(self, device: str = "cuda", model_id: str = "lama"):
        self.device = device
        self.model_id = model_id
        self.pipe = None

    def load(self) -> None:
        from simple_lama_inpainting import SimpleLama  # type: ignore

        # SimpleLaMa selects torch device based on CUDA availability.
        # CUDA visibility is controlled by CUDA_VISIBLE_DEVICES.
        self.pipe = SimpleLama()

    def inpaint(self, image_bgr: np.ndarray, mask_u8: np.ndarray, prompt: str = "natural background, photorealistic") -> np.ndarray:
        if self.pipe is None:
            self.load()
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        m = (mask_u8 * 255).astype(np.uint8) if mask_u8.max() <= 1 else mask_u8.astype(np.uint8)
        image = Image.fromarray(image_rgb)
        mask = Image.fromarray(m)
        out = self.pipe(image, mask)
        out_bgr = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
        return out_bgr

    def get_or_create_plate(self, image_id: str, image_bgr: np.ndarray, mask_u8: np.ndarray, plate_dir: Path) -> tuple[np.ndarray, str, str]:
        plate_dir.mkdir(parents=True, exist_ok=True)
        msha = mask_sha(mask_u8)
        plate_path = plate_dir / f"{image_id}_{msha[:12]}.png"
        if plate_path.exists():
            plate = cv2.imread(str(plate_path), cv2.IMREAD_COLOR)
            if plate is not None:
                return plate, str(plate_path), msha
        plate = self.inpaint(image_bgr=image_bgr, mask_u8=mask_u8)
        cv2.imwrite(str(plate_path), plate)
        return plate, str(plate_path), msha

