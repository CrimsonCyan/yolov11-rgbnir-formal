from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.box_ops import AREA_REFERENCE_SIZE, area_bucket, box_iou, xywh_to_xyxy
from formal_rgbnir.iddaw import (
    DEFAULT_PAIRS,
    build_dataset_yaml,
    category_names_for_mode,
    common_val_kwargs,
    mode_specific_kwargs,
    resolve_dataset_root,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class, box_iou as ultralytics_box_iou


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    weights: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OA gate behavior and small/medium/large AP.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Repeated case in the form name|mode|weights. If omitted, --name/--mode/--weights are used.",
    )
    parser.add_argument("--name", default="", help="Single-case display name.")
    parser.add_argument("--mode", default="", help="Single-case mode.")
    parser.add_argument("--weights", default="", help="Single-case checkpoint path.")
    parser.add_argument("--out", default="runs/analysis/oa_small_targets", help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument(
        "--area-imgsz",
        type=float,
        default=AREA_REFERENCE_SIZE,
        help="Reference letterbox size for AP_S/AP_M/AP_L buckets. Defaults to COCO-style 640: <32^2, 32^2-96^2, >=96^2.",
    )
    parser.add_argument("--batch", type=int, default=0, help="Batch size for validator metrics. Defaults to mode val batch.")
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence used for AP collection.")
    parser.add_argument("--pr-conf", type=float, default=0.25, help="Confidence threshold for precision/recall summary.")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0", help="Inference device passed to Ultralytics, e.g. 0, 1, cpu.")
    parser.add_argument("--half", action="store_true", help="Use half precision inference on CUDA devices.")
    parser.add_argument(
        "--metric-backend",
        choices=["validator", "predict"],
        default="validator",
        help="Use an isolated Ultralytics validator subclass for AP by default; predict keeps the legacy diagnostic path.",
    )
    parser.add_argument(
        "--skip-predict-diagnostics",
        action="store_true",
        help="Only run validator metrics. This disables gate stats, match CSV, and saved visualizations.",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Run one image per predict() call. Slower, but avoids long-stream CUDA cache growth.",
    )
    parser.add_argument("--save-images", type=int, default=0, help="Save up to N annotated prediction images per case.")
    parser.add_argument("--save-image-conf", type=float, default=0.25, help="Confidence threshold for saved images.")
    parser.add_argument("--save-match-images", type=int, default=0, help="Save up to N GT/Pred/TP-FP-FN comparison images per case.")
    parser.add_argument("--save-gate-images", type=int, default=0, help="Save up to N OA gate/residual heatmap images per case.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for TP/FP/FN visualization.")
    parser.add_argument(
        "--containment-suppress",
        action="store_true",
        help="Suppress same-class large boxes that mostly contain smaller boxes before metric/match summaries.",
    )
    parser.add_argument(
        "--containment-conf",
        type=float,
        default=None,
        help="Confidence threshold used by containment suppression/statistics. Defaults to --pr-conf.",
    )
    parser.add_argument(
        "--containment-overlap",
        type=float,
        default=0.8,
        help="Minimum intersection/small-box-area ratio for same-class containment.",
    )
    parser.add_argument(
        "--containment-area-ratio",
        type=float,
        default=1.5,
        help="Minimum large-box/small-box area ratio for containment suppression/statistics.",
    )
    parser.add_argument(
        "--containment-conf-margin",
        type=float,
        default=0.05,
        help="Suppress the large box unless its score exceeds the small box by this margin.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap for quick debugging.")
    return parser.parse_args()


def parse_cases(args: argparse.Namespace) -> list[Case]:
    cases: list[Case] = []
    for raw in args.case:
        parts = raw.split("|")
        if len(parts) != 3:
            raise ValueError(f"--case must use name|mode|weights, got: {raw}")
        cases.append(Case(parts[0], parts[1], Path(parts[2])))
    if not cases:
        if not args.mode or not args.weights:
            raise ValueError("Provide repeated --case entries or both --mode and --weights.")
        cases.append(Case(args.name or args.mode, args.mode, Path(args.weights)))
    return cases


def image_paths_for_mode(mode: str) -> list[Path]:
    root = resolve_dataset_root(mode)
    visible_val = root / "visible" / "val"
    return sorted(path for path in visible_val.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def load_target(image_path: Path, orig_shape: tuple[int, int], area_imgsz: float) -> dict[str, object]:
    label_path = image_path.with_suffix(".txt")
    h, w = orig_shape
    boxes_xywh = []
    labels = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels.append(int(float(parts[0])))
            boxes_xywh.append([float(value) for value in parts[1:5]])
    if boxes_xywh:
        xywh = torch.tensor(boxes_xywh, dtype=torch.float32)
        xyxy = xywh_to_xyxy(xywh)
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        boxes = xyxy * scale
        labels_t = torch.tensor(labels, dtype=torch.long)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.long)
    buckets = [
        area_bucket(float((box[2] - box[0]) * (box[3] - box[1])), image_shape=(h, w), reference_size=area_imgsz)
        for box in boxes
    ]
    return {
        "sample_id": image_path.stem,
        "boxes_xyxy": boxes,
        "labels": labels_t,
        "area_buckets": buckets,
        "shape": (h, w),
    }


def collect_prediction(result) -> dict[str, object]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    return {
        "boxes": boxes.xyxy.detach().cpu().float(),
        "scores": boxes.conf.detach().cpu().float(),
        "labels": boxes.cls.detach().cpu().long(),
    }


def bucket_boxes(boxes: torch.Tensor, image_shape, area_imgsz: float) -> list[str]:
    return [
        area_bucket(
            float((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)),
            image_shape=image_shape,
            reference_size=area_imgsz,
        )
        for box in boxes
    ]


def box_area_xyxy(box: torch.Tensor) -> float:
    return float(((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)).item())


def intersection_area_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> float:
    left_top = torch.maximum(box1[:2], box2[:2])
    right_bottom = torch.minimum(box1[2:], box2[2:])
    wh = (right_bottom - left_top).clamp(min=0)
    return float((wh[0] * wh[1]).item())


def find_containment_pairs(
    prediction: dict[str, torch.Tensor],
    conf: float,
    overlap_threshold: float,
    area_ratio_threshold: float,
) -> list[tuple[int, int, int]]:
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    indices = torch.nonzero(scores >= conf, as_tuple=False).flatten().tolist()
    pairs: list[tuple[int, int, int]] = []
    for pos, idx_a in enumerate(indices):
        for idx_b in indices[pos + 1 :]:
            if int(labels[idx_a]) != int(labels[idx_b]):
                continue
            area_a = box_area_xyxy(boxes[idx_a])
            area_b = box_area_xyxy(boxes[idx_b])
            if area_a <= 0.0 or area_b <= 0.0:
                continue
            if area_a >= area_b:
                large_idx, small_idx = idx_a, idx_b
                large_area, small_area = area_a, area_b
            else:
                large_idx, small_idx = idx_b, idx_a
                large_area, small_area = area_b, area_a
            if large_area / max(small_area, 1e-6) < area_ratio_threshold:
                continue
            contained = intersection_area_xyxy(boxes[large_idx], boxes[small_idx]) / max(small_area, 1e-6)
            if contained >= overlap_threshold:
                pairs.append((large_idx, small_idx, int(labels[large_idx])))
    return pairs


def containment_counts_by_class(pairs: list[tuple[int, int, int]], class_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for _, _, cls in pairs:
        if 0 <= cls < len(class_names):
            counts[class_names[cls]] += 1
    return counts


def suppress_contained_predictions(
    prediction: dict[str, torch.Tensor],
    conf: float,
    overlap_threshold: float,
    area_ratio_threshold: float,
    conf_margin: float,
) -> tuple[dict[str, torch.Tensor], list[int], list[tuple[int, int, int]]]:
    pairs = find_containment_pairs(prediction, conf, overlap_threshold, area_ratio_threshold)
    if not pairs:
        return prediction, [], pairs

    boxes = prediction["boxes"]
    scores = prediction["scores"]
    # Resolve stronger containment conflicts first, so chained overlaps stay deterministic.
    pairs.sort(
        key=lambda item: box_area_xyxy(boxes[item[0]]) / max(box_area_xyxy(boxes[item[1]]), 1e-6),
        reverse=True,
    )
    suppressed: set[int] = set()
    for large_idx, small_idx, _ in pairs:
        if large_idx in suppressed or small_idx in suppressed:
            continue
        large_score = float(scores[large_idx])
        small_score = float(scores[small_idx])
        if large_score <= small_score + conf_margin:
            suppressed.add(large_idx)
        else:
            suppressed.add(small_idx)

    if not suppressed:
        return prediction, [], pairs
    keep = torch.ones(len(scores), dtype=torch.bool)
    keep[list(suppressed)] = False
    filtered = {
        "boxes": prediction["boxes"][keep],
        "scores": prediction["scores"][keep],
        "labels": prediction["labels"][keep],
    }
    return filtered, sorted(suppressed), pairs


def prediction_area_mask(boxes: torch.Tensor, area_filter: str, image_shape, area_imgsz: float) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes.device)
    values = [
        area_bucket(
            float((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)),
            image_shape=image_shape,
            reference_size=area_imgsz,
        )
        == area_filter
        for box in boxes
    ]
    return torch.tensor(values, dtype=torch.bool, device=boxes.device)


def same_class_overlap_mask_device(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((len(pred_boxes),), dtype=torch.bool, device=pred_boxes.device)
    ious = ultralytics_box_iou(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels
    return (ious * same_class).max(0).values >= iou_threshold


def filter_predn_for_area(
    predn: torch.Tensor,
    valid_boxes: torch.Tensor,
    valid_labels: torch.Tensor,
    ignored_boxes: torch.Tensor,
    ignored_labels: torch.Tensor,
    area_filter: str,
    image_shape,
    area_imgsz: float,
) -> torch.Tensor:
    if len(predn) == 0:
        return predn
    pred_boxes = predn[:, :4]
    pred_labels = predn[:, 5]
    pred_area = prediction_area_mask(pred_boxes, area_filter, image_shape, area_imgsz)
    valid_overlap = same_class_overlap_mask_device(pred_boxes, pred_labels, valid_boxes, valid_labels)
    ignored_overlap = same_class_overlap_mask_device(pred_boxes, pred_labels, ignored_boxes, ignored_labels)
    keep = (pred_area | valid_overlap) & ~(ignored_overlap & ~valid_overlap)
    return predn[keep]


class AreaDetectionValidator(DetectionValidator):
    """Isolated validator used only by this diagnostic script; it does not patch Ultralytics' default validator."""

    last_instance = None
    area_names = ("small", "medium", "large")
    area_reference_size = AREA_REFERENCE_SIZE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        AreaDetectionValidator.last_instance = self
        self.area_stats: dict[str, dict[str, list[torch.Tensor]]] = {}

    def init_metrics(self, model):
        super().init_metrics(model)
        self.area_stats = {
            area: {
                "tp": [],
                "conf": [],
                "pred_cls": [],
                "target_cls": [],
            }
            for area in self.area_names
        }

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            cls = pbatch["cls"]
            bbox = pbatch["bbox"]
            predn = self._prepare_pred(pred, pbatch) if len(pred) else torch.zeros((0, 6), device=self.device)
            if self.args.single_cls and len(predn):
                predn[:, 5] = 0

            area_imgsz = float(getattr(self, "area_reference_size", AREA_REFERENCE_SIZE))
            buckets = bucket_boxes(bbox.detach().cpu(), pbatch["ori_shape"], area_imgsz) if len(bbox) else []
            for area in self.area_names:
                valid_mask = torch.tensor([bucket == area for bucket in buckets], dtype=torch.bool, device=self.device)
                ignored_mask = ~valid_mask if len(valid_mask) else torch.zeros((len(bbox),), dtype=torch.bool, device=self.device)
                valid_bbox = bbox[valid_mask]
                valid_cls = cls[valid_mask]
                ignored_bbox = bbox[ignored_mask]
                ignored_cls = cls[ignored_mask]
                area_predn = filter_predn_for_area(
                    predn,
                    valid_bbox,
                    valid_cls,
                    ignored_bbox,
                    ignored_cls,
                    area,
                    pbatch["ori_shape"],
                    area_imgsz,
                )
                stat = self.area_stats[area]
                stat["conf"].append(area_predn[:, 4] if len(area_predn) else torch.zeros(0, device=self.device))
                stat["pred_cls"].append(area_predn[:, 5] if len(area_predn) else torch.zeros(0, device=self.device))
                stat["target_cls"].append(valid_cls)
                if len(area_predn) and len(valid_cls):
                    stat["tp"].append(self._process_batch(area_predn, valid_bbox, valid_cls))
                else:
                    stat["tp"].append(torch.zeros((len(area_predn), self.niou), dtype=torch.bool, device=self.device))

    def area_metric_rows(self, class_names: list[str], pr_conf: float) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for area, stats in self.area_stats.items():
            arrays = {
                key: torch.cat(value, 0).cpu().numpy()
                for key, value in stats.items()
            }
            metric_by_cls = metrics_from_stats(arrays, class_names)
            conf_mask = arrays["conf"] >= pr_conf if len(arrays["conf"]) else np.zeros((0,), dtype=bool)
            pr_stats = {
                "tp": arrays["tp"][conf_mask, :1] if len(arrays["tp"]) else np.zeros((0, 1), dtype=bool),
                "conf": arrays["conf"][conf_mask],
                "pred_cls": arrays["pred_cls"][conf_mask],
                "target_cls": arrays["target_cls"],
            }
            pr_by_cls = pr_at_conf_from_stats(pr_stats, class_names)
            for cls, name in enumerate(class_names):
                metric = metric_by_cls[cls]
                p_at_conf, r_at_conf = pr_by_cls[cls]
                rows.append(
                    {
                        "class": name,
                        "area": area,
                        "gt_count": metric["gt_count"],
                        "precision": metric["precision"],
                        "recall": metric["recall"],
                        "AP50": metric["AP50"],
                        "mAP50_95": metric["mAP50_95"],
                        "precision_at_conf": p_at_conf,
                        "recall_at_conf": r_at_conf,
                        "conf": pr_conf,
                    }
                )
        return rows


def draw_labeled_box(image, box, label: str, color: tuple[int, int, int], thickness: int = 2) -> None:
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(y1, th + baseline + 3)
    cv2.rectangle(image, (x1, y_text - th - baseline - 3), (min(x1 + tw + 3, w - 1), y_text + 2), color, -1)
    cv2.putText(image, label, (x1 + 1, y_text - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def add_panel_title(image, title: str, color: tuple[int, int, int] = (32, 32, 32)):
    import cv2
    import numpy as np

    header = np.full((48, image.shape[1], image.shape[2]), 245, dtype=image.dtype)
    cv2.putText(header, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return np.vstack([header, image])


def as_bgr_image(image):
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] > 3:
        return image[:, :, :3].copy()
    return image.copy()


def save_prediction_image(
    result,
    class_names: list[str],
    out_path: Path,
    conf_threshold: float,
    prediction: dict[str, torch.Tensor] | None = None,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Saving annotated images requires opencv-python/cv2.") from exc

    image = as_bgr_image(result.orig_img)
    if prediction is None:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), image)
            return
        xyxy = boxes.xyxy.detach().cpu().float()
        scores = boxes.conf.detach().cpu().float()
        labels = boxes.cls.detach().cpu().long()
    else:
        xyxy = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]
    if len(xyxy) > 0:
        palette = [
            (56, 56, 255),
            (151, 157, 255),
            (31, 112, 255),
            (29, 178, 255),
            (49, 210, 207),
            (10, 249, 72),
        ]
        for box, score, label in zip(xyxy, scores, labels):
            if float(score) < conf_threshold:
                continue
            cls = int(label)
            color = palette[cls % len(palette)]
            name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
            draw_labeled_box(image, box, f"{name} {float(score):.2f}", color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def match_image_predictions(prediction: dict[str, torch.Tensor], target: dict[str, object], conf: float, iou: float):
    pred_boxes = prediction["boxes"]
    pred_scores = prediction["scores"]
    pred_labels = prediction["labels"]
    keep = pred_scores >= conf
    indices = torch.nonzero(keep, as_tuple=False).flatten().tolist()
    indices.sort(key=lambda idx: float(pred_scores[idx]), reverse=True)

    gt_boxes = target["boxes_xyxy"]
    gt_labels = target["labels"]
    matched_gt: set[int] = set()
    tp_pred: list[int] = []
    fp_pred: list[int] = []
    for pred_idx in indices:
        same_class = [
            gt_idx
            for gt_idx, gt_label in enumerate(gt_labels)
            if int(gt_label) == int(pred_labels[pred_idx]) and gt_idx not in matched_gt
        ]
        if same_class:
            ious = box_iou(pred_boxes[pred_idx].unsqueeze(0), gt_boxes[same_class]).squeeze(0)
            best_iou, best_local_idx = torch.max(ious, dim=0)
            if float(best_iou) >= iou:
                matched_gt.add(same_class[int(best_local_idx)])
                tp_pred.append(pred_idx)
                continue
        fp_pred.append(pred_idx)
    fn_gt = [gt_idx for gt_idx in range(len(gt_boxes)) if gt_idx not in matched_gt]
    return tp_pred, fp_pred, fn_gt


def summarize_image_matches(
    image_path: Path,
    prediction: dict[str, torch.Tensor],
    target: dict[str, object],
    class_names: list[str],
    conf: float,
    iou: float,
    containment_pairs: list[tuple[int, int, int]],
    suppressed_indices: list[int],
    raw_prediction: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    tp_pred, fp_pred, fn_gt = match_image_predictions(prediction, target, conf, iou)
    pred_labels = prediction["labels"]
    pred_scores = prediction["scores"]
    target_labels = target["labels"]
    suppressed_labels = raw_prediction["labels"]
    pair_counts = containment_counts_by_class(containment_pairs, class_names)
    suppressed_counts = {name: 0 for name in class_names}
    for idx in suppressed_indices:
        cls = int(suppressed_labels[idx])
        if 0 <= cls < len(class_names):
            suppressed_counts[class_names[cls]] += 1

    rows = []
    classes = ["all", *class_names]
    for cls_name in classes:
        cls_id = None if cls_name == "all" else class_names.index(cls_name)
        gt_count = int(len(target_labels)) if cls_id is None else int((target_labels == cls_id).sum().item())
        if cls_id is None:
            pred_mask = pred_scores >= conf
        else:
            pred_mask = (pred_scores >= conf) & (pred_labels == cls_id)
        pred_count = int(pred_mask.sum().item())
        tp_count = len(tp_pred) if cls_id is None else sum(int(pred_labels[idx]) == cls_id for idx in tp_pred)
        fp_count = len(fp_pred) if cls_id is None else sum(int(pred_labels[idx]) == cls_id for idx in fp_pred)
        fn_count = len(fn_gt) if cls_id is None else sum(int(target_labels[idx]) == cls_id for idx in fn_gt)
        containment_pair_count = len(containment_pairs) if cls_id is None else pair_counts[cls_name]
        suppressed_count = len(suppressed_indices) if cls_id is None else suppressed_counts[cls_name]
        rows.append(
            {
                "image_id": target["sample_id"],
                "image_path": str(image_path),
                "class": cls_name,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "tp": tp_count,
                "fp": fp_count,
                "fn": fn_count,
                "precision": tp_count / max(tp_count + fp_count, 1),
                "recall": tp_count / max(gt_count, 1),
                "conf": conf,
                "match_iou": iou,
                "containment_pairs": containment_pair_count,
                "containment_suppressed": suppressed_count,
            }
        )
    return rows


def save_match_comparison_image(
    result,
    prediction: dict[str, torch.Tensor],
    target: dict[str, object],
    class_names: list[str],
    out_dir: Path,
    conf_threshold: float,
    match_iou: float,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Saving TP/FP/FN images requires opencv-python/cv2.") from exc

    stem = Path(result.path).stem
    base_image = as_bgr_image(result.orig_img)
    gt_panel = base_image.copy()
    pred_panel = base_image.copy()
    match_panel = base_image.copy()
    gt_color = (255, 128, 0)
    tp_color = (0, 180, 0)
    fp_color = (0, 0, 255)
    fn_color = (255, 0, 0)

    for box, label in zip(target["boxes_xyxy"], target["labels"]):
        cls = int(label)
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        draw_labeled_box(gt_panel, box, name, gt_color)

    for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
        if float(score) < conf_threshold:
            continue
        cls = int(label)
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        draw_labeled_box(pred_panel, box, f"{name} {float(score):.2f}", (56, 56, 255))

    tp_pred, fp_pred, fn_gt = match_image_predictions(prediction, target, conf_threshold, match_iou)
    for pred_idx in tp_pred:
        cls = int(prediction["labels"][pred_idx])
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        score = float(prediction["scores"][pred_idx])
        draw_labeled_box(match_panel, prediction["boxes"][pred_idx], f"TP {name} {score:.2f}", tp_color, thickness=3)
    for pred_idx in fp_pred:
        cls = int(prediction["labels"][pred_idx])
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        score = float(prediction["scores"][pred_idx])
        draw_labeled_box(match_panel, prediction["boxes"][pred_idx], f"FP {name} {score:.2f}", fp_color, thickness=2)
    for gt_idx in fn_gt:
        cls = int(target["labels"][gt_idx])
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        draw_labeled_box(match_panel, target["boxes_xyxy"][gt_idx], f"FN {name}", fn_color, thickness=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_titled = add_panel_title(gt_panel, "GT", gt_color)
    pred_titled = add_panel_title(pred_panel, f"Prediction conf>={conf_threshold:.2f}", (56, 56, 255))
    match_titled = add_panel_title(match_panel, f"TP/FP/FN IoU>={match_iou:.2f}", (32, 32, 32))
    compare = cv2.hconcat([gt_titled, pred_titled, match_titled])
    cv2.imwrite(str(out_dir / f"{stem}_gt.jpg"), gt_titled)
    cv2.imwrite(str(out_dir / f"{stem}_pred.jpg"), pred_titled)
    cv2.imwrite(str(out_dir / f"{stem}_match.jpg"), match_titled)
    cv2.imwrite(str(out_dir / f"{stem}_compare.jpg"), compare)


def normalized_heatmap(values: torch.Tensor, value_range: tuple[float, float] | None = None):
    import numpy as np

    array = values.detach().cpu().float().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if value_range is None:
        lo, hi = np.percentile(array, [1, 99])
    else:
        lo, hi = value_range
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def overlay_heatmap(image, values: torch.Tensor, title: str, value_range: tuple[float, float] | None = None):
    import cv2

    h, w = image.shape[:2]
    normalized = normalized_heatmap(values, value_range)
    resized = cv2.resize(normalized, (w, h), interpolation=cv2.INTER_LINEAR)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap((resized * 255).astype("uint8"), colormap)
    overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
    return add_panel_title(overlay, title, (32, 32, 32))


def blank_panel_like(image, title: str, message: str):
    import cv2
    import numpy as np

    panel = np.full_like(image, 245)
    cv2.putText(panel, message, (24, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (96, 96, 96), 2)
    return add_panel_title(panel, title, (96, 96, 96))


def save_gate_residual_comparison_image(
    result,
    modules,
    target: dict[str, object],
    class_names: list[str],
    out_dir: Path,
) -> bool:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Saving OA gate/residual images requires opencv-python/cv2.") from exc

    module = next(
        (
            candidate
            for candidate in modules
            if isinstance(getattr(candidate, "last_object_gate", None), torch.Tensor)
        ),
        None,
    )
    if module is None:
        return False

    stem = Path(result.path).stem
    image = as_bgr_image(result.orig_img)
    gt_panel = image.copy()
    for box, label in zip(target["boxes_xyxy"], target["labels"]):
        cls = int(label)
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        draw_labeled_box(gt_panel, box, name, (255, 128, 0), thickness=2)
    gt_panel = add_panel_title(gt_panel, "GT", (255, 128, 0))

    gate = module.last_object_gate.detach().cpu().float()
    gate_map = gate[0, 0]
    gate_panel = overlay_heatmap(
        image,
        gate_map,
        f"OA gate mean={float(gate_map.mean()):.3f}",
        value_range=(0.0, 1.0),
    )

    delta = getattr(module, "last_residual_delta", None)
    if isinstance(delta, torch.Tensor):
        residual_map = delta.detach().cpu().float()[0].abs().mean(0)
        residual_panel = overlay_heatmap(
            image,
            residual_map,
            f"Residual |delta| mean={float(residual_map.mean()):.4f}",
            value_range=None,
        )
    else:
        residual_panel = blank_panel_like(image, "Residual |delta|", "No residual captured")

    out_dir.mkdir(parents=True, exist_ok=True)
    compare = cv2.hconcat([gt_panel, gate_panel, residual_panel])
    cv2.imwrite(str(out_dir / f"{stem}_gate.jpg"), gate_panel)
    cv2.imwrite(str(out_dir / f"{stem}_residual.jpg"), residual_panel)
    cv2.imwrite(str(out_dir / f"{stem}_gate_residual_compare.jpg"), compare)
    return True


def prediction_area_bucket(box: torch.Tensor, image_shape=None, area_imgsz: float = AREA_REFERENCE_SIZE) -> str:
    width = float((box[2] - box[0]).clamp(min=0).item())
    height = float((box[3] - box[1]).clamp(min=0).item())
    return area_bucket(width * height, image_shape=image_shape, reference_size=area_imgsz)


def collect_gate_modules(model) -> list[torch.nn.Module]:
    modules = []
    for module in model.model.modules():
        has_fg_loss = float(getattr(module, "foreground_loss_weight", 0.0) or 0.0) > 0
        has_gate_capture = hasattr(module, "capture_object_gate") or hasattr(module, "last_object_gate")
        has_residual_capture = hasattr(module, "last_residual_delta")
        if has_fg_loss or has_gate_capture or has_residual_capture:
            module.capture_object_gate = True
            module.last_object_gate = None
            if hasattr(module, "last_residual_delta"):
                module.last_residual_delta = None
            modules.append(module)
    return modules


def mask_for_boxes(boxes: torch.Tensor, labels: torch.Tensor, cls: int | None, gate_shape: tuple[int, int], image_shape):
    gh, gw = gate_shape
    ih, iw = image_shape
    mask = torch.zeros((gh, gw), dtype=torch.bool)
    if boxes.numel() == 0:
        return mask
    for box, label in zip(boxes, labels):
        if cls is not None and int(label) != cls:
            continue
        x1 = int(torch.floor(box[0] / iw * gw).clamp(0, max(gw - 1, 0)).item())
        y1 = int(torch.floor(box[1] / ih * gh).clamp(0, max(gh - 1, 0)).item())
        x2 = int(torch.ceil(box[2] / iw * gw).clamp(1, gw).item())
        y2 = int(torch.ceil(box[3] / ih * gh).clamp(1, gh).item())
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def update_gate_stats(stats: dict[str, list[float]], modules, target, class_names: list[str]) -> None:
    if not modules:
        return
    module = modules[0]
    gate = getattr(module, "last_object_gate", None)
    if gate is None or not isinstance(gate, torch.Tensor):
        return
    gate_map = gate.detach().cpu().float()[0, 0]
    gh, gw = gate_map.shape
    boxes = target["boxes_xyxy"]
    labels = target["labels"]
    image_shape = target["shape"]
    inside = mask_for_boxes(boxes, labels, None, (gh, gw), image_shape)
    outside = ~inside
    stats["gate_all"].append(float(gate_map.mean()))
    if inside.any():
        stats["gate_inside"].append(float(gate_map[inside].mean()))
    if outside.any():
        stats["gate_outside"].append(float(gate_map[outside].mean()))

    for cls_name in ("person", "motorcycle", "traffic light", "traffic sign"):
        if cls_name not in class_names:
            continue
        cls = class_names.index(cls_name)
        cls_mask = mask_for_boxes(boxes, labels, cls, (gh, gw), image_shape)
        if cls_mask.any():
            stats[f"gate_{cls_name.replace(' ', '_')}_inside"].append(float(gate_map[cls_mask].mean()))

    delta = getattr(module, "last_residual_delta", None)
    if delta is not None and isinstance(delta, torch.Tensor):
        delta_map = delta.detach().cpu().float()[0].abs().mean(0)
        stats["residual_all"].append(float(delta_map.mean()))
        stats["residual_max"].append(float(delta_map.max()))
        if inside.any():
            stats["residual_inside"].append(float(delta_map[inside].mean()))
            stats["residual_inside_max"].append(float(delta_map[inside].max()))
        if outside.any():
            stats["residual_outside"].append(float(delta_map[outside].mean()))
            stats["residual_outside_max"].append(float(delta_map[outside].max()))


def iter_predictions(model: YOLO, images: list[Path], args: argparse.Namespace, mode_kwargs: dict[str, object]):
    half = bool(args.half and str(args.device).lower() != "cpu")
    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "batch": 1,
        "stream": True,
        "verbose": False,
        "save": False,
        "device": args.device,
        "half": half,
        **mode_kwargs,
    }
    if args.per_image:
        for path in images:
            for result in model.predict(source=str(path), **predict_kwargs):
                yield result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return
    if args.max_images > 0:
        source_dir = Path(args.out) / "_source_lists"
        source_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=source_dir, delete=False, encoding="utf-8") as handle:
            for path in images:
                handle.write(f"{path}\n")
            source = handle.name
    else:
        source = str(images[0].parent)
    for result in model.predict(source=source, **predict_kwargs):
        yield result


def match_predictions_for_area(
    predictions,
    targets,
    class_id: int,
    iou_threshold: float,
    area_filter: str | None,
    conf: float | None = None,
    area_imgsz: float = AREA_REFERENCE_SIZE,
) -> tuple[list[float], list[float], int]:
    pred_records = []
    gt_map = {}
    for pred, target in zip(predictions, targets):
        image_id = target["sample_id"]
        valid_entries = []
        ignored_boxes = []
        for box, label, bucket in zip(target["boxes_xyxy"], target["labels"], target["area_buckets"]):
            if int(label) != class_id:
                continue
            if area_filter is None or bucket == area_filter:
                valid_entries.append([box, False])
            else:
                ignored_boxes.append(box)
        gt_map[image_id] = {"valid": valid_entries, "ignored": ignored_boxes}
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if int(label) == class_id and (conf is None or float(score) >= conf):
                pred_records.append((image_id, float(score), box, target.get("shape")))
    pred_records.sort(key=lambda item: item[1], reverse=True)
    total_gts = sum(len(items["valid"]) for items in gt_map.values())
    tps, fps = [], []
    for image_id, _, pred_box, image_shape in pred_records:
        entries = gt_map.get(image_id, {"valid": [], "ignored": []})
        gt_entries = entries["valid"]
        unmatched = [(idx, entry[0]) for idx, entry in enumerate(gt_entries) if not entry[1]]
        if unmatched:
            gt_boxes = torch.stack([box for _, box in unmatched], dim=0)
            ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, local_best_idx = torch.max(ious, dim=0)
            if float(best_iou) >= iou_threshold:
                best_idx = unmatched[int(local_best_idx)][0]
                gt_entries[best_idx][1] = True
                tps.append(1.0)
                fps.append(0.0)
                continue

        ignored_boxes = entries["ignored"]
        if ignored_boxes:
            ignored_ious = box_iou(pred_box.unsqueeze(0), torch.stack(ignored_boxes, dim=0)).squeeze(0)
            if float(ignored_ious.max()) >= iou_threshold:
                continue

        if area_filter is not None and prediction_area_bucket(pred_box, image_shape, area_imgsz) != area_filter:
            continue
        tps.append(0.0)
        fps.append(1.0)
    return tps, fps, total_gts


def split_target_by_area(target: dict[str, object], area_filter: str | None):
    boxes = target["boxes_xyxy"]
    labels = target["labels"]
    if area_filter is None:
        return boxes, labels, torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    mask = torch.tensor([bucket == area_filter for bucket in target["area_buckets"]], dtype=torch.bool)
    return boxes[mask], labels[mask], boxes[~mask], labels[~mask]


def same_class_overlap_mask(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros((len(pred_boxes),), dtype=torch.bool)
    ious = ultralytics_box_iou(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels
    return (ious * same_class).max(0).values >= iou_threshold


def filter_prediction_for_area(
    prediction: dict[str, torch.Tensor],
    valid_boxes: torch.Tensor,
    valid_labels: torch.Tensor,
    ignored_boxes: torch.Tensor,
    ignored_labels: torch.Tensor,
    area_filter: str | None,
    image_shape=None,
    area_imgsz: float = AREA_REFERENCE_SIZE,
) -> dict[str, torch.Tensor]:
    if area_filter is None or len(prediction["boxes"]) == 0:
        return prediction

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    pred_area_mask = torch.tensor(
        [prediction_area_bucket(box, image_shape, area_imgsz) == area_filter for box in boxes],
        dtype=torch.bool,
    )
    valid_overlap = same_class_overlap_mask(boxes, labels, valid_boxes, valid_labels)
    ignored_overlap = same_class_overlap_mask(boxes, labels, ignored_boxes, ignored_labels)

    # COCO-style area diagnostics: predictions that can match a valid-area GT are kept even if
    # their own box area falls outside the bucket; unmatched detections outside the bucket or
    # detections that only hit ignored-area GTs are removed from that area-specific AP pool.
    keep = (pred_area_mask | valid_overlap) & ~(ignored_overlap & ~valid_overlap)
    return {
        "boxes": prediction["boxes"][keep],
        "scores": prediction["scores"][keep],
        "labels": prediction["labels"][keep],
    }


def ultralytics_match_predictions(
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return torch.tensor(correct, dtype=torch.bool)
    correct_class = true_classes[:, None] == pred_classes
    iou_np = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool)


def collect_ultralytics_stats(
    predictions,
    targets,
    area_filter: str | None,
    iouv: torch.Tensor,
    conf: float | None = None,
    area_imgsz: float = AREA_REFERENCE_SIZE,
) -> dict[str, np.ndarray]:
    tp_chunks: list[np.ndarray] = []
    conf_chunks: list[np.ndarray] = []
    pred_cls_chunks: list[np.ndarray] = []
    target_cls_chunks: list[np.ndarray] = []

    for prediction, target in zip(predictions, targets):
        valid_boxes, valid_labels, ignored_boxes, ignored_labels = split_target_by_area(target, area_filter)
        filtered = filter_prediction_for_area(
            prediction,
            valid_boxes,
            valid_labels,
            ignored_boxes,
            ignored_labels,
            area_filter,
            target.get("shape"),
            area_imgsz,
        )
        pred_boxes = filtered["boxes"]
        pred_scores = filtered["scores"]
        pred_labels = filtered["labels"]
        if conf is not None and len(pred_scores):
            keep = pred_scores >= conf
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

        npr = len(pred_boxes)
        correct = torch.zeros((npr, len(iouv)), dtype=torch.bool)
        if npr and len(valid_labels):
            iou = ultralytics_box_iou(valid_boxes, pred_boxes)
            correct = ultralytics_match_predictions(pred_labels, valid_labels, iou, iouv)

        tp_chunks.append(correct.numpy())
        conf_chunks.append(pred_scores.numpy())
        pred_cls_chunks.append(pred_labels.numpy())
        target_cls_chunks.append(valid_labels.numpy())

    return {
        "tp": np.concatenate(tp_chunks, axis=0) if tp_chunks else np.zeros((0, len(iouv)), dtype=bool),
        "conf": np.concatenate(conf_chunks, axis=0) if conf_chunks else np.zeros((0,), dtype=float),
        "pred_cls": np.concatenate(pred_cls_chunks, axis=0) if pred_cls_chunks else np.zeros((0,), dtype=float),
        "target_cls": np.concatenate(target_cls_chunks, axis=0) if target_cls_chunks else np.zeros((0,), dtype=float),
    }


def metrics_from_stats(stats: dict[str, np.ndarray], class_names: list[str]) -> dict[int, dict[str, float]]:
    target_cls = stats["target_cls"].astype(int)
    gt_counts = np.bincount(target_cls, minlength=len(class_names)) if len(target_cls) else np.zeros(len(class_names), dtype=int)
    rows = {
        cls: {
            "gt_count": int(gt_counts[cls]),
            "precision": 0.0,
            "recall": 0.0,
            "AP50": 0.0,
            "mAP50_95": 0.0,
        }
        for cls in range(len(class_names))
    }
    if len(target_cls) and len(stats["tp"]):
        _, _, p, r, _, ap, unique_classes, *_ = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            names=dict(enumerate(class_names)),
        )
        for idx, cls in enumerate(unique_classes.astype(int)):
            rows[cls].update(
                {
                    "precision": float(p[idx]),
                    "recall": float(r[idx]),
                    "AP50": float(ap[idx, 0]),
                    "mAP50_95": float(ap[idx].mean()),
                }
            )
    return rows


def pr_at_conf_from_stats(stats: dict[str, np.ndarray], class_names: list[str]) -> dict[int, tuple[float, float]]:
    target_cls = stats["target_cls"].astype(int)
    pred_cls = stats["pred_cls"].astype(int)
    tp = stats["tp"][:, 0].astype(bool) if len(stats["tp"]) else np.zeros((0,), dtype=bool)
    out: dict[int, tuple[float, float]] = {}
    for cls in range(len(class_names)):
        pred_mask = pred_cls == cls
        gt_count = int((target_cls == cls).sum())
        tp_count = int(tp[pred_mask].sum())
        fp_count = int(pred_mask.sum() - tp_count)
        out[cls] = (
            tp_count / max(tp_count + fp_count, 1),
            tp_count / max(gt_count, 1),
        )
    return out


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def append_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    areas = []
    for row in rows:
        area = str(row["area"])
        if area not in areas:
            areas.append(area)
    for area in areas:
        subset = [row for row in rows if row["area"] == area and row["class"] != "mean" and int(row["gt_count"]) > 0]
        rows.append(
            {
                "class": "mean",
                "area": area,
                "gt_count": sum(int(row["gt_count"]) for row in subset),
                "precision": mean([float(row["precision"]) for row in subset]),
                "recall": mean([float(row["recall"]) for row in subset]),
                "AP50": mean([float(row["AP50"]) for row in subset]),
                "mAP50_95": mean([float(row["mAP50_95"]) for row in subset]),
                "precision_at_conf": mean([float(row["precision_at_conf"]) for row in subset]),
                "recall_at_conf": mean([float(row["recall_at_conf"]) for row in subset]),
                "conf": subset[0]["conf"] if subset else 0.0,
            }
        )
    return rows


def validator_all_metric_rows(validator: AreaDetectionValidator, class_names: list[str], pr_conf: float) -> list[dict[str, object]]:
    rows = {
        cls: {
            "class": name,
            "area": "all",
            "gt_count": int(validator.nt_per_class[cls]) if validator.nt_per_class is not None else 0,
            "precision": 0.0,
            "recall": 0.0,
            "AP50": 0.0,
            "mAP50_95": 0.0,
            "precision_at_conf": 0.0,
            "recall_at_conf": 0.0,
            "conf": pr_conf,
        }
        for cls, name in enumerate(class_names)
    }
    for metric_idx, cls in enumerate(validator.metrics.ap_class_index):
        p, r, ap50, ap = validator.metrics.class_result(metric_idx)
        rows[int(cls)].update(
            {
                "precision": float(p),
                "recall": float(r),
                "AP50": float(ap50),
                "mAP50_95": float(ap),
            }
        )

    raw_stats = {
        key: torch.cat(value, 0).cpu().numpy()
        for key, value in validator.stats.items()
        if key in {"tp", "conf", "pred_cls", "target_cls"}
    }
    conf_mask = raw_stats["conf"] >= pr_conf if len(raw_stats["conf"]) else np.zeros((0,), dtype=bool)
    pr_stats = {
        "tp": raw_stats["tp"][conf_mask, :1] if len(raw_stats["tp"]) else np.zeros((0, 1), dtype=bool),
        "conf": raw_stats["conf"][conf_mask],
        "pred_cls": raw_stats["pred_cls"][conf_mask],
        "target_cls": raw_stats["target_cls"],
    }
    pr_by_cls = pr_at_conf_from_stats(pr_stats, class_names)
    for cls, (p_at_conf, r_at_conf) in pr_by_cls.items():
        rows[cls]["precision_at_conf"] = p_at_conf
        rows[cls]["recall_at_conf"] = r_at_conf
    return [rows[cls] for cls in range(len(class_names))]


def run_validator_metrics(case: Case, args: argparse.Namespace, class_names: list[str]) -> list[dict[str, object]]:
    AreaDetectionValidator.last_instance = None
    AreaDetectionValidator.area_reference_size = float(args.area_imgsz)
    model = YOLO(str(case.weights))
    val_kwargs = common_val_kwargs(case.mode, args.imgsz, batch=args.batch or None)
    val_kwargs.update(
        {
            "device": args.device,
            "conf": args.conf,
            "iou": args.iou,
            "plots": False,
            "verbose": False,
            "half": bool(args.half),
            **mode_specific_kwargs(case.mode),
        }
    )
    model.val(data=str(build_dataset_yaml(case.mode)), validator=AreaDetectionValidator, **val_kwargs)
    validator = AreaDetectionValidator.last_instance
    if validator is None:
        raise RuntimeError("AreaDetectionValidator did not run; cannot collect validator metrics.")
    rows = validator_all_metric_rows(validator, class_names, args.pr_conf)
    rows.extend(validator.area_metric_rows(class_names, args.pr_conf))
    return append_mean_rows(rows)


def summarize_metrics(
    predictions,
    targets,
    class_names: list[str],
    pr_conf: float,
    area_imgsz: float = AREA_REFERENCE_SIZE,
) -> list[dict[str, object]]:
    iouv = torch.linspace(0.5, 0.95, 10)
    pr_iouv = torch.tensor([0.5])
    rows = []
    areas = [None, "small", "medium", "large"]
    for area in areas:
        stats = collect_ultralytics_stats(predictions, targets, area, iouv, area_imgsz=area_imgsz)
        metric_by_cls = metrics_from_stats(stats, class_names)
        pr_stats = collect_ultralytics_stats(predictions, targets, area, pr_iouv, conf=pr_conf, area_imgsz=area_imgsz)
        pr_by_cls = pr_at_conf_from_stats(pr_stats, class_names)
        for cls, name in enumerate(class_names):
            metric = metric_by_cls[cls]
            p_at_conf, r_at_conf = pr_by_cls[cls]
            rows.append(
                {
                    "class": name,
                    "area": area or "all",
                    "gt_count": metric["gt_count"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "AP50": metric["AP50"],
                    "mAP50_95": metric["mAP50_95"],
                    "precision_at_conf": p_at_conf,
                    "recall_at_conf": r_at_conf,
                    "conf": pr_conf,
                }
            )
    return append_mean_rows(rows)


def analyze_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
    class_names = category_names_for_mode(case.mode)
    images = image_paths_for_mode(case.mode)
    if args.max_images > 0:
        images = images[: args.max_images]

    metric_rows: list[dict[str, object]] | None = None
    if args.metric_backend == "validator":
        metric_rows = run_validator_metrics(case, args, class_names)

    predictions = []
    targets = []
    match_rows: list[dict[str, object]] = []
    gate_stats: dict[str, list[float]] = {
        "gate_all": [],
        "gate_inside": [],
        "gate_outside": [],
        "gate_person_inside": [],
        "gate_motorcycle_inside": [],
        "gate_traffic_light_inside": [],
        "gate_traffic_sign_inside": [],
        "residual_all": [],
        "residual_max": [],
        "residual_inside": [],
        "residual_inside_max": [],
        "residual_outside": [],
        "residual_outside_max": [],
    }
    saved_images = 0
    saved_match_images = 0
    saved_gate_images = 0
    image_out_dir = Path(args.out) / case.name / "pred_images"
    match_image_out_dir = Path(args.out) / case.name / "match_images"
    gate_image_out_dir = Path(args.out) / case.name / "gate_images"
    containment_conf = args.containment_conf if args.containment_conf is not None else args.pr_conf

    if not args.skip_predict_diagnostics:
        model = YOLO(str(case.weights))
        modules = collect_gate_modules(model)
        mode_kwargs = mode_specific_kwargs(case.mode)
        for result in iter_predictions(model, images, args, mode_kwargs):
            image_path = Path(result.path)
            target = load_target(image_path, result.orig_shape, args.area_imgsz)
            raw_prediction = collect_prediction(result)
            containment_pairs = find_containment_pairs(
                raw_prediction,
                containment_conf,
                args.containment_overlap,
                args.containment_area_ratio,
            )
            suppressed_indices: list[int] = []
            prediction = raw_prediction
            if args.containment_suppress:
                prediction, suppressed_indices, containment_pairs = suppress_contained_predictions(
                    raw_prediction,
                    containment_conf,
                    args.containment_overlap,
                    args.containment_area_ratio,
                    args.containment_conf_margin,
                )
            targets.append(target)
            predictions.append(prediction)
            update_gate_stats(gate_stats, modules, target, class_names)
            match_rows.extend(
                summarize_image_matches(
                    image_path,
                    prediction,
                    target,
                    class_names,
                    args.pr_conf,
                    args.match_iou,
                    containment_pairs,
                    suppressed_indices,
                    raw_prediction,
                )
            )
            if args.save_images > 0 and saved_images < args.save_images:
                save_prediction_image(
                    result,
                    class_names,
                    image_out_dir / f"{image_path.stem}_pred.jpg",
                    args.save_image_conf,
                    prediction,
                )
                saved_images += 1
            if args.save_match_images > 0 and saved_match_images < args.save_match_images:
                save_match_comparison_image(
                    result,
                    prediction,
                    target,
                    class_names,
                    match_image_out_dir,
                    args.save_image_conf,
                    args.match_iou,
                )
                saved_match_images += 1
            if args.save_gate_images > 0 and saved_gate_images < args.save_gate_images:
                if save_gate_residual_comparison_image(result, modules, target, class_names, gate_image_out_dir):
                    saved_gate_images += 1

    if metric_rows is None:
        metric_rows = summarize_metrics(predictions, targets, class_names, args.pr_conf, args.area_imgsz)
    gate_summary = {key: mean(values) for key, values in gate_stats.items()}
    gate_summary["num_gate_samples"] = len(gate_stats["gate_all"])
    gate_summary["num_saved_images"] = saved_images
    gate_summary["num_saved_match_images"] = saved_match_images
    gate_summary["num_saved_gate_images"] = saved_gate_images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": case.name,
        "mode": case.mode,
        "weights": str(case.weights),
        "num_images": len(images),
        "diagnostic_args": {
            "metric_backend": args.metric_backend,
            "validator_backend": "isolated AreaDetectionValidator" if args.metric_backend == "validator" else "",
            "ap_backend": "ultralytics.ap_per_class",
            "match_backend": "ultralytics.DetectionValidator.match_predictions",
            "area_policy": (
                f"COCO-style AP_S/AP_M/AP_L at {args.area_imgsz:g} letterbox scale: "
                "small < 32^2, medium 32^2 <= area < 96^2, large >= 96^2; "
                "native boxes are scaled by letterbox gain before bucketing"
            ),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "pr_conf": args.pr_conf,
            "nms_iou": args.iou,
            "match_iou": args.match_iou,
            "containment_suppress": args.containment_suppress,
            "containment_conf": containment_conf,
            "containment_overlap": args.containment_overlap,
            "containment_area_ratio": args.containment_area_ratio,
            "containment_conf_margin": args.containment_conf_margin,
        },
        "metrics": metric_rows,
        "match_summary": match_rows,
        "gate_summary": gate_summary,
    }


def write_case_outputs(out_dir: Path, payload: dict[str, object]) -> None:
    case_dir = out_dir / str(payload["name"])
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = payload["metrics"]
    with (case_dir / "metrics_by_class_area.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    with (case_dir / "gate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in payload["gate_summary"].items():
            writer.writerow([key, value])
    match_rows = payload.get("match_summary") or []
    if match_rows:
        with (case_dir / "match_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(match_rows[0].keys()))
            writer.writeheader()
            writer.writerows(match_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for case in parse_cases(args):
        payload = analyze_case(case, args)
        payloads.append(payload)
        write_case_outputs(out_dir, payload)
        mean_all = next(row for row in payload["metrics"] if row["class"] == "mean" and row["area"] == "all")
        mean_small = next(row for row in payload["metrics"] if row["class"] == "mean" and row["area"] == "small")
        print(
            f"{payload['name']}: mAP50-95={mean_all['mAP50_95']:.5f}, "
            f"mAP50-95_small={mean_small['mAP50_95']:.5f}, AP50_small={mean_small['AP50']:.5f}, "
            f"gate_samples={payload['gate_summary']['num_gate_samples']}"
        )
    (out_dir / "all_cases_summary.json").write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
