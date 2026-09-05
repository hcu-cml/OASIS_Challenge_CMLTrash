"""Zero-shot instance segmentation and mask utilities using SAM3."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

MASK_CACHE = Path(os.environ.get("MASK_CACHE", "fixtures/masks"))
GATE_CONFIDENCE = float(os.environ.get("GATE_CONFIDENCE", "0.60"))


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    """Run-length encode a binary mask."""
    flat = mask.astype(np.uint8).ravel()
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], changes, [flat.size]])
    return {
        "shape": list(mask.shape),
        "start": int(flat[0]),
        "runs": np.diff(bounds).astype(int).tolist(),
    }


def rle_decode(payload: dict[str, Any]) -> np.ndarray:
    """Decode a run-length encoded mask payload."""
    flat = np.zeros(int(np.prod(payload["shape"])), dtype=np.uint8)
    value = int(payload["start"])
    pos = 0
    for run in payload["runs"]:
        if value:
            flat[pos : pos + run] = 1
        pos += run
        value ^= 1
    return flat.reshape(payload["shape"])


@dataclass
class Instance:
    """A detected instance mask with prompt and confidence score."""

    mask: np.ndarray
    prompt: str
    score: float | None


def _cache_key(
    image_id: str,
    prompts: list[str],
    revision: str,
    namespace: str = "default",
    category: str | None = None,
) -> Path:
    key: dict[str, Any] = {"i": image_id, "p": sorted(prompts), "r": revision}
    if namespace != "default" or category is not None:
        key.update({"namespace": namespace, "category": category})
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
    return MASK_CACHE / f"{digest}.json"


class Segmenter:
    """SAM3 text-prompted instance segmentation."""

    def __init__(
        self,
        weights_dir: str,
        revision: str = "unpinned",
        confidence: float = GATE_CONFIDENCE,
        mask_threshold: float = 0.9,
        min_area_pct: float = 0.15,
        max_area_pct: float = 40.0,
    ):
        self.weights_dir = weights_dir
        self.revision = revision
        self.confidence = confidence
        self.mask_threshold = mask_threshold
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _ensure_weights(self) -> None:
        """Ensure SAM3 model weights exist locally, downloading if HF_TOKEN is available."""
        import os
        config_file = os.path.join(self.weights_dir, "config.json")
        if os.path.exists(config_file):
            return

        token = os.environ.get("HF_TOKEN", "").strip() or None
        log.info("SAM3 weights not found at %s. Attempting download from Hugging Face (facebook/sam3)...", self.weights_dir)
        try:
            from huggingface_hub import snapshot_download

            os.makedirs(self.weights_dir, exist_ok=True)
            snapshot_download(
                repo_id="facebook/sam3",
                local_dir=self.weights_dir,
                token=token,
            )
            log.info("Successfully downloaded SAM3 weights to %s", self.weights_dir)
        except Exception as exc:
            if not token:
                raise RuntimeError(
                    f"SAM3 weights not found in '{self.weights_dir}' and HF_TOKEN is not set. "
                    "Please provide your Hugging Face token via HF_TOKEN in .env or through the web UI "
                    "Configure panel to auto-download facebook/sam3 weights."
                ) from exc
            raise RuntimeError(
                f"Failed to auto-download SAM3 weights from Hugging Face: {exc}. "
                "Ensure your HF_TOKEN has approved access to https://huggingface.co/facebook/sam3"
            ) from exc

    def _load(self) -> None:
        if self._model is not None:
            return
        self._ensure_weights()

        import torch
        from transformers import Sam3Model, Sam3Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        log.info("Loading SAM3 from %s on %s", self.weights_dir, self._device)
        self._model = (
            Sam3Model.from_pretrained(self.weights_dir, torch_dtype=dtype, local_files_only=True)
            .to(self._device)
            .eval()
        )
        self._processor = Sam3Processor.from_pretrained(self.weights_dir, local_files_only=True)

    def segment(
        self,
        image_path: str,
        prompts: list[str],
        check_cancel: Callable[[], None] | None = None,
    ) -> list[Instance]:
        """Run SAM3 segmentation for a list of text prompts."""
        import torch
        from PIL import Image

        self._load()
        prompts = [str(p).strip() for p in (prompts or []) if str(p).strip()]
        if not prompts:
            raise ValueError("Segmenter.segment requires a non-empty prompt list.")

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            min_area_px = width * height * (self.min_area_pct / 100.0)
            max_area_px = width * height * (self.max_area_pct / 100.0)

            instances: list[Instance] = []
            for prompt in prompts:
                if check_cancel:
                    check_cancel()
                inputs = self._processor(
                    images=image, text=[prompt], return_tensors="pt"
                ).to(self._device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                if check_cancel:
                    check_cancel()

                masks = outputs.pred_masks.squeeze(0)
                scores = torch.sigmoid(outputs.pred_logits.squeeze(0)).float().cpu().numpy()
                if masks.ndim == 3:
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(1).float(),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

                for mask, score in zip(masks.cpu().numpy(), np.atleast_1d(scores)):
                    if float(score) < self.confidence:
                        continue
                    binary = (mask > self.mask_threshold).astype(np.uint8)
                    area = binary.sum()
                    if area < min_area_px:
                        continue
                    if area > max_area_px:
                        log.debug(
                            "dropped degenerate mask: %.1f%% of frame for %r",
                            100.0 * area / (width * height), prompt,
                        )
                        continue
                    import cv2
                    # ~7px radius, treating near-touching masks as one instance.
                    kernel = np.ones((15, 15), np.uint8)
                    dilated = cv2.dilate(binary, kernel, iterations=1)
                    
                    overlap_indices = []
                    for idx, existing_instance in enumerate(instances):
                        if np.logical_and(dilated, existing_instance.mask).sum() > 0:
                            overlap_indices.append(idx)
                            
                    if overlap_indices:
                        first_idx = overlap_indices[0]
                        target_instance = instances[first_idx]
                        target_instance.mask = np.logical_or(target_instance.mask, binary).astype(np.uint8)
                        target_instance.score = max(target_instance.score, float(score))
                        
                        prompts_set = set(p.strip() for p in target_instance.prompt.split(","))
                        if prompt not in prompts_set:
                            target_instance.prompt += f", {prompt}"
                            prompts_set.add(prompt)
                            
                        # Fold any remaining overlaps into the same instance.
                        for idx in reversed(overlap_indices[1:]):
                            other = instances.pop(idx)
                            target_instance.mask = np.logical_or(target_instance.mask, other.mask).astype(np.uint8)
                            target_instance.score = max(target_instance.score, other.score)
                            for p in other.prompt.split(","):
                                p = p.strip()
                                if p and p not in prompts_set:
                                    target_instance.prompt += f", {p}"
                                    prompts_set.add(p)
                    else:
                        instances.append(
                            Instance(mask=binary, prompt=prompt, score=float(score))
                        )

        if self._device == "cuda":
            import torch as _t

            _t.cuda.empty_cache()
        return instances

    def segment_cached(
        self,
        image_id: str,
        image_path: str,
        prompts: list[str],
        namespace: str = "default",
        category: str | None = None,
        check_cancel: Callable[[], None] | None = None,
    ) -> list[Instance]:
        """Cache-first segmentation lookup, running SAM3 on cache miss."""
        effective_prompts = [str(p).strip() for p in (prompts or []) if str(p).strip()]
        if not effective_prompts:
            raise ValueError(f"segment_cached({image_id!r}) requires a non-empty prompt list.")
        path = _cache_key(
            image_id, effective_prompts, self.revision,
            namespace=namespace, category=category,
        )
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            scores = payload.get("scores")
            mask_prompts = payload.get("mask_prompts")
            out: list[Instance] = []
            for i, mask in enumerate(payload["masks"]):
                if check_cancel:
                    check_cancel()
                out.append(
                    Instance(
                        mask=rle_decode(mask),
                        prompt=(mask_prompts[i] if mask_prompts else effective_prompts[0]),
                        score=(float(scores[i]) if scores and scores[i] is not None else None),
                    )
                )
            return out

        from .sources import replay_only

        if replay_only():
            raise FileNotFoundError(
                f"REPLAY=1 but no cached {namespace}/{category or 'uncategorized'} "
                f"masks for image {image_id}."
            )

        instances = self.segment(
            image_path, effective_prompts, check_cancel=check_cancel
        )
        return instances





def overlay(image_path: str, masks: list[np.ndarray], out_path: str) -> str:
    """Green mask overlay for the review UI."""
    from PIL import Image

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as handle:
        arr = np.array(handle.convert("RGB"))
    for mask in masks:
        if mask.shape != arr.shape[:2]:
            from .geo import resize_mask

            mask = resize_mask(mask, arr.shape[:2])
        sel = mask > 0
        arr[sel] = (arr[sel] * 0.45 + np.array([32, 210, 96]) * 0.55).astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=88)
    return out_path






def redact_masks(
    image_path: str,
    masks: list[np.ndarray],
    out_path: str,
    blur_radius: float = 28.0,
    dilation_px: int = 25,
) -> str:
    """Strongly blur dilated human masks while retaining the rest of the scene."""
    from PIL import Image, ImageFilter

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as handle:
        original = handle.convert("RGB")
    union = np.zeros((original.height, original.width), dtype=np.uint8)
    for raw in masks:
        mask = raw
        if mask.shape != union.shape:
            from .geo import resize_mask

            mask = resize_mask(mask, union.shape)
        union |= (mask > 0).astype(np.uint8)
    selection = Image.fromarray(union * 255, mode="L")
    size = max(3, int(dilation_px) | 1)
    selection = selection.filter(ImageFilter.MaxFilter(size=size))
    blurred = original.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    Image.composite(blurred, original, selection).save(out_path, quality=88)
    return out_path
