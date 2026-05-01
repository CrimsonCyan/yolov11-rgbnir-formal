from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.box_ops import area_bucket, box_iou, xywh_to_xyxy
from formal_rgbnir.iddaw import DEFAULT_PAIRS, category_names_for_mode, mode_specific_kwargs, resolve_dataset_root
from ultralytics import YOLO


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
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence used for AP collection.")
    parser.add_argument("--pr-conf", type=float, default=0.25, help="Confidence threshold for precision/recall summary.")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0", help="Inference device passed to Ultralytics, e.g. 0, 1, cpu.")
    parser.add_argument("--half", action="store_true", help="Use half precision inference on CUDA devices.")
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Run one image per predict() call. Slower, but avoids long-stream CUDA cache growth.",
    )
    parser.add_argument("--save-images", type=int, default=0, help="Save up to N annotated prediction images per case.")
    parser.add_argument("--save-image-conf", type=float, default=0.25, help="Confidence threshold for saved images.")
    parser.add_argument("--save-match-images", type=int, default=0, help="Save up to N GT/Pred/TP-FP-FN comparison images per case.")
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


def load_target(image_path: Path, orig_shape: tuple[int, int]) -> dict[str, object]:
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
    buckets = [area_bucket(float((box[2] - box[0]) * (box[3] - box[1]))) for box in boxes]
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

    image = result.orig_img.copy()
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
    gt_panel = result.orig_img.copy()
    pred_panel = result.orig_img.copy()
    match_panel = result.orig_img.copy()
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


def prediction_area_bucket(box: torch.Tensor) -> str:
    width = float((box[2] - box[0]).clamp(min=0).item())
    height = float((box[3] - box[1]).clamp(min=0).item())
    return area_bucket(width * height)


def collect_gate_modules(model) -> list[torch.nn.Module]:
    modules = []
    for module in model.model.modules():
        if float(getattr(module, "foreground_loss_weight", 0.0) or 0.0) > 0:
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

    for cls_name in ("person", "motorcycle"):
        if cls_name not in class_names:
            continue
        cls = class_names.index(cls_name)
        cls_mask = mask_for_boxes(boxes, labels, cls, (gh, gw), image_shape)
        if cls_mask.any():
            stats[f"gate_{cls_name}_inside"].append(float(gate_map[cls_mask].mean()))

    delta = getattr(module, "last_residual_delta", None)
    if delta is not None and isinstance(delta, torch.Tensor):
        delta_map = delta.detach().cpu().float()[0].abs().mean(0)
        stats["residual_all"].append(float(delta_map.mean()))
        if inside.any():
            stats["residual_inside"].append(float(delta_map[inside].mean()))
        if outside.any():
            stats["residual_outside"].append(float(delta_map[outside].mean()))


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
                pred_records.append((image_id, float(score), box))
    pred_records.sort(key=lambda item: item[1], reverse=True)
    total_gts = sum(len(items["valid"]) for items in gt_map.values())
    tps, fps = [], []
    for image_id, _, pred_box in pred_records:
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

        if area_filter is not None and prediction_area_bucket(pred_box) != area_filter:
            continue
        tps.append(0.0)
        fps.append(1.0)
    return tps, fps, total_gts


def ap_for_threshold(predictions, targets, class_id: int, iou_threshold: float, area_filter: str | None) -> float:
    tps, fps, total_gts = match_predictions_for_area(predictions, targets, class_id, iou_threshold, area_filter)
    if total_gts == 0:
        return 0.0
    if not tps:
        return 0.0
    tps_t = torch.tensor(tps).cumsum(0)
    fps_t = torch.tensor(fps).cumsum(0)
    recalls = tps_t / max(total_gts, 1)
    precisions = tps_t / (tps_t + fps_t).clamp(min=1e-6)
    ap = 0.0
    for level in torch.linspace(0, 1, 101):
        valid = precisions[recalls >= level]
        ap += float(valid.max()) if valid.numel() > 0 else 0.0
    return ap / 101.0


def pr_at_conf(predictions, targets, class_id: int, area_filter: str | None, conf: float, iou_threshold: float = 0.5):
    tps, fps, total_gts = match_predictions_for_area(
        predictions, targets, class_id, iou_threshold, area_filter, conf=conf
    )
    tp = int(sum(tps))
    fp = int(sum(fps))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gts, 1)
    return precision, recall, total_gts


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_metrics(predictions, targets, class_names: list[str], pr_conf: float) -> list[dict[str, object]]:
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    rows = []
    areas = [None, "small", "medium", "large"]
    for cls, name in enumerate(class_names):
        for area in areas:
            ap50 = ap_for_threshold(predictions, targets, cls, 0.5, area)
            map5095 = mean([ap_for_threshold(predictions, targets, cls, thr, area) for thr in thresholds])
            p, r, n = pr_at_conf(predictions, targets, cls, area, pr_conf)
            rows.append(
                {
                    "class": name,
                    "area": area or "all",
                    "gt_count": n,
                    "AP50": ap50,
                    "mAP50_95": map5095,
                    "precision_at_conf": p,
                    "recall_at_conf": r,
                    "conf": pr_conf,
                }
            )
    for area in areas:
        subset = [row for row in rows if row["area"] == (area or "all")]
        rows.append(
            {
                "class": "mean",
                "area": area or "all",
                "gt_count": sum(int(row["gt_count"]) for row in subset),
                "AP50": mean([float(row["AP50"]) for row in subset]),
                "mAP50_95": mean([float(row["mAP50_95"]) for row in subset]),
                "precision_at_conf": mean([float(row["precision_at_conf"]) for row in subset]),
                "recall_at_conf": mean([float(row["recall_at_conf"]) for row in subset]),
                "conf": pr_conf,
            }
        )
    return rows


def analyze_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
    model = YOLO(str(case.weights))
    modules = collect_gate_modules(model)
    class_names = category_names_for_mode(case.mode)
    mode_kwargs = mode_specific_kwargs(case.mode)
    images = image_paths_for_mode(case.mode)
    if args.max_images > 0:
        images = images[: args.max_images]

    predictions = []
    targets = []
    match_rows: list[dict[str, object]] = []
    gate_stats: dict[str, list[float]] = {
        "gate_all": [],
        "gate_inside": [],
        "gate_outside": [],
        "gate_person_inside": [],
        "gate_motorcycle_inside": [],
        "residual_all": [],
        "residual_inside": [],
        "residual_outside": [],
    }
    saved_images = 0
    saved_match_images = 0
    image_out_dir = Path(args.out) / case.name / "pred_images"
    match_image_out_dir = Path(args.out) / case.name / "match_images"
    containment_conf = args.containment_conf if args.containment_conf is not None else args.pr_conf

    for result in iter_predictions(model, images, args, mode_kwargs):
        image_path = Path(result.path)
        target = load_target(image_path, result.orig_shape)
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

    metric_rows = summarize_metrics(predictions, targets, class_names, args.pr_conf)
    gate_summary = {key: mean(values) for key, values in gate_stats.items()}
    gate_summary["num_gate_samples"] = len(gate_stats["gate_all"])
    gate_summary["num_saved_images"] = saved_images
    gate_summary["num_saved_match_images"] = saved_match_images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": case.name,
        "mode": case.mode,
        "weights": str(case.weights),
        "num_images": len(images),
        "diagnostic_args": {
            "imgsz": args.imgsz,
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
            f"AP_small={mean_small['AP50']:.5f}, gate_samples={payload['gate_summary']['num_gate_samples']}"
        )
    (out_dir / "all_cases_summary.json").write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
