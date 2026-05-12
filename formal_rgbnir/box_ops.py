from __future__ import annotations

import torch


AREA_REFERENCE_SIZE = 640.0
SMALL_AREA_SIDE = 32.0
MEDIUM_AREA_SIDE = 96.0


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = boxes.clone()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    return out


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def letterbox_area_scale(image_shape, reference_size: float = AREA_REFERENCE_SIZE) -> float:
    """Return the area scale from native image space to a square letterbox size."""
    if image_shape is None:
        return 1.0
    if isinstance(image_shape, torch.Tensor):
        shape = image_shape.detach().cpu().flatten().tolist()
    else:
        shape = list(image_shape)
    if len(shape) < 2:
        return 1.0
    height = max(float(shape[0]), 1e-6)
    width = max(float(shape[1]), 1e-6)
    gain = min(float(reference_size) / height, float(reference_size) / width)
    return gain * gain


def area_bucket(area: float, image_shape=None, reference_size: float = AREA_REFERENCE_SIZE) -> str:
    scaled_area = float(area) * letterbox_area_scale(image_shape, reference_size)
    if scaled_area < SMALL_AREA_SIDE * SMALL_AREA_SIDE:
        return "small"
    if scaled_area < MEDIUM_AREA_SIDE * MEDIUM_AREA_SIDE:
        return "medium"
    return "large"
