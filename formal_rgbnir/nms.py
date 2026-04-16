from __future__ import annotations

import torch

from .box_ops import box_iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current_index = int(order[0])
        keep.append(current_index)
        if order.numel() == 1:
            break
        current_box = boxes[current_index].unsqueeze(0)
        rest_boxes = boxes[order[1:]]
        ious = box_iou(current_box, rest_boxes).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    keep_all: list[torch.Tensor] = []
    for class_id in labels.unique(sorted=True):
        class_mask = labels == class_id
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
        class_keep = nms(boxes[class_indices], scores[class_indices], iou_threshold)
        keep_all.append(class_indices[class_keep])
    keep = torch.cat(keep_all)
    _, order = scores[keep].sort(descending=True)
    return keep[order]
