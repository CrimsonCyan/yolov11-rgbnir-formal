from __future__ import annotations

from dataclasses import dataclass

import torch

from .box_ops import area_bucket, box_iou


@dataclass
class DetectionRecord:
    image_id: str
    score: float
    label: int
    bbox_xyxy: torch.Tensor


def _compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0
    recall_levels = torch.linspace(0, 1, 101)
    ap = 0.0
    for level in recall_levels:
        valid = precisions[recalls >= level]
        ap += float(valid.max()) if valid.numel() > 0 else 0.0
    return ap / 101.0


def _collect_records(predictions, targets, class_id: int, area_filter: str | None):
    pred_records: list[DetectionRecord] = []
    gt_map: dict[str, list[tuple[torch.Tensor, bool]]] = {}
    for pred, target in zip(predictions, targets):
        image_id = target["sample_id"]
        entries: list[tuple[torch.Tensor, bool]] = []
        for box, label, bucket in zip(target["boxes_xyxy"], target["labels"], target["area_buckets"]):
            if int(label) == class_id and (area_filter is None or bucket == area_filter):
                entries.append((box, False))
        gt_map[image_id] = entries
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if int(label) == class_id:
                pred_records.append(
                    DetectionRecord(image_id=image_id, score=float(score), label=int(label), bbox_xyxy=box)
                )
    pred_records.sort(key=lambda item: item.score, reverse=True)
    total_gts = sum(len(items) for items in gt_map.values())
    return pred_records, gt_map, total_gts


def _ap_for_threshold(predictions, targets, class_id: int, iou_threshold: float, area_filter: str | None) -> float:
    pred_records, gt_map, total_gts = _collect_records(predictions, targets, class_id, area_filter)
    if total_gts == 0:
        return 0.0
    tps = []
    fps = []
    for record in pred_records:
        gt_entries = gt_map.get(record.image_id, [])
        if not gt_entries:
            tps.append(0.0)
            fps.append(1.0)
            continue
        gt_boxes = torch.stack([entry[0] for entry in gt_entries], dim=0)
        ious = box_iou(record.bbox_xyxy.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        matched = gt_entries[int(best_idx)][1]
        if float(best_iou) >= iou_threshold and not matched:
            gt_entries[int(best_idx)] = (gt_entries[int(best_idx)][0], True)
            tps.append(1.0)
            fps.append(0.0)
        else:
            tps.append(0.0)
            fps.append(1.0)
    if not tps:
        return 0.0
    tps_t = torch.tensor(tps).cumsum(0)
    fps_t = torch.tensor(fps).cumsum(0)
    recalls = tps_t / max(total_gts, 1)
    precisions = tps_t / (tps_t + fps_t).clamp(min=1e-6)
    return _compute_ap(recalls, precisions)


def evaluate_predictions(predictions, targets, num_classes: int) -> dict[str, float]:
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    aps_50 = []
    aps_all = []
    aps_small = []
    aps_medium = []
    aps_large = []
    for class_id in range(num_classes):
        aps_50.append(_ap_for_threshold(predictions, targets, class_id, 0.5, None))
        aps_all.append(sum(_ap_for_threshold(predictions, targets, class_id, thr, None) for thr in thresholds) / len(thresholds))
        aps_small.append(_ap_for_threshold(predictions, targets, class_id, 0.5, "small"))
        aps_medium.append(_ap_for_threshold(predictions, targets, class_id, 0.5, "medium"))
        aps_large.append(_ap_for_threshold(predictions, targets, class_id, 0.5, "large"))

    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    return {
        "AP50": _mean(aps_50),
        "mAP@0.5:0.95": _mean(aps_all),
        "AP_S": _mean(aps_small),
        "AP_M": _mean(aps_medium),
        "AP_L": _mean(aps_large),
    }


def build_eval_targets(targets) -> list[dict]:
    built = []
    for target in targets:
        area_buckets = [area_bucket(float((box[2] - box[0]) * (box[3] - box[1]))) for box in target["boxes_xyxy"]]
        built.append(
            {
                "sample_id": target["sample_id"],
                "boxes_xyxy": target["boxes_xyxy"],
                "labels": target["labels"],
                "area_buckets": area_buckets,
            }
        )
    return built
