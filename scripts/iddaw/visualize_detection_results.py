from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.iddaw import (  # noqa: E402
    category_names_for_mode,
    latest_weights_for,
    mode_specific_kwargs,
    resolve_dataset_root,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
TP_COLOR = (40, 210, 40)
FP_COLOR = (30, 30, 230)
FN_COLOR = (0, 190, 255)
GT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    mode: str
    backend: str
    weights: Path | None = None
    predictions: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-ready RGB/NIR/GT/model detection comparison panels. "
            "YOLO case: name|mode|weights.pt. External case: name|coco|predictions.json."
        )
    )
    parser.add_argument("--case", action="append", required=True, help="Case spec: name|mode|weights.pt or name|coco|predictions.json")
    parser.add_argument("--dataset-mode", default="", help="Mode used only to resolve the visualization dataset root.")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root override.")
    parser.add_argument("--source", default="", help="Optional visible image file or directory override.")
    parser.add_argument("--image-list", default="", help="Optional text file containing visible image paths or names.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", default="runs/analysis/visualizations/detection_results")
    parser.add_argument("--baseline", default="", help="Case name used as baseline for automatic case categories.")
    parser.add_argument("--proposed", default="", help="Case name used as proposed method for automatic case categories.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for drawing and TP/FP/FN matching.")
    parser.add_argument("--predict-conf", type=float, default=0.001, help="Lower predict threshold before post-filtering.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold used by model.predict().")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for TP/FP/FN visualization.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--save-top-k", type=int, default=6, help="Images saved per category.")
    parser.add_argument("--gt-min", type=int, default=0, help="Only consider images with at least this many GT boxes.")
    parser.add_argument("--gt-max", type=int, default=0, help="Only consider images with at most this many GT boxes. 0 disables.")
    parser.add_argument(
        "--save-best-per-case",
        action="store_true",
        help="Save one comparison image per case, selecting GT-filtered images where that case has the best TP/FN/FP score.",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Save only model prediction panels, without RGB/NIR/GT context panels.",
    )
    parser.add_argument(
        "--save-model-panels",
        action="store_true",
        help="Also save each selected model prediction panel as a separate image.",
    )
    parser.add_argument(
        "--plain-model-panels",
        action="store_true",
        help="For --save-model-panels, save boxes directly on the original image without title padding.",
    )
    parser.add_argument(
        "--keep-model-panel-size",
        action="store_true",
        help="For --save-model-panels, keep original image resolution instead of resizing to --panel-width.",
    )
    parser.add_argument(
        "--select-proposed-better",
        action="store_true",
        help="Save images where --proposed ranks above every other case and non-proposed cases differ.",
    )
    parser.add_argument("--small-area", type=float, default=32 * 32, help="Small target area threshold at visualization scale.")
    parser.add_argument("--panel-width", type=int, default=640, help="Resize each panel to this width before concatenation.")
    parser.add_argument(
        "--coco-prediction-space",
        choices=["letterbox", "original"],
        default="letterbox",
        help="Coordinate space of external COCO prediction bboxes before drawing.",
    )
    parser.add_argument(
        "--coco-class-id-base",
        type=int,
        default=1,
        help="External COCO category id base. Use 1 for standard COCO-style ids, 0 for zero-based ids.",
    )
    parser.add_argument(
        "--coco-gt-json",
        default="",
        help="Optional external GT JSON used to remap external image_id values through file_name/stem.",
    )
    return parser.parse_args()


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("visualize_detection_results.py requires opencv-python/cv2.") from exc
    return cv2


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "visualization"


def parse_case(raw: str) -> CaseSpec:
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) == 3 and parts[1].lower() in {"coco", "coco-json", "json"}:
        name, _backend, pred_raw = parts
        predictions = Path(pred_raw).expanduser().resolve()
        if not predictions.exists():
            raise FileNotFoundError(f"COCO prediction JSON does not exist for case {name}: {predictions}")
        return CaseSpec(name=name, mode="", backend="coco", predictions=predictions)
    if len(parts) == 2:
        name, mode, weight_raw = parts[0], parts[0], parts[1]
    elif len(parts) == 3:
        name, mode, weight_raw = parts
    else:
        raise ValueError(f"Invalid --case value: {raw!r}; expected name|mode|weights.pt or name|coco|predictions.json")
    if weight_raw.lower() in {"latest", "best", "best.pt"}:
        weights = latest_weights_for(mode, "best.pt")
    else:
        weights = Path(weight_raw).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weight file does not exist for case {name}: {weights}")
    return CaseSpec(name=name, mode=mode, backend="yolo", weights=weights)


def first_yolo_mode(cases: list[CaseSpec]) -> str:
    for case in cases:
        if case.mode:
            return case.mode
    raise ValueError("At least one YOLO case or --dataset-mode is required to infer class names and dataset root.")


def source_subdir_for_mode(mode: str) -> str:
    kwargs = mode_specific_kwargs(mode)
    return "nir" if kwargs.get("use_simotm") == "Gray" else "visible"


def list_images(path: Path, max_images: int) -> list[Path]:
    if path.is_file():
        images = [path]
    elif path.is_dir():
        images = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(f"Source does not exist: {path}")
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise RuntimeError(f"No images found in {path}")
    return images


def resolve_images(args: argparse.Namespace, cases: list[CaseSpec]) -> list[Path]:
    if args.source:
        return list_images(Path(args.source).expanduser().resolve(), args.max_images)

    dataset_mode = args.dataset_mode or first_yolo_mode(cases)
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else resolve_dataset_root(dataset_mode)
    visible_dir = dataset_root / "visible" / args.split

    if args.image_list:
        rows = [line.strip() for line in Path(args.image_list).read_text(encoding="utf-8").splitlines() if line.strip()]
        images: list[Path] = []
        for row in rows:
            candidate = Path(row)
            if not candidate.is_absolute():
                candidate = visible_dir / row
            if candidate.exists():
                images.append(candidate.resolve())
        if args.max_images > 0:
            images = images[: args.max_images]
        if not images:
            raise RuntimeError(f"No valid images found from --image-list {args.image_list}")
        return images

    return list_images(visible_dir, args.max_images)


def paired_nir_path(visible_path: Path) -> Path | None:
    parts = list(visible_path.parts)
    if "visible" not in parts:
        return None
    idx = parts.index("visible")
    parts[idx] = "nir"
    nir_path = Path(*parts)
    return nir_path if nir_path.exists() else None


def image_path_for_case(case: CaseSpec, visible_path: Path) -> Path:
    if case.backend != "yolo":
        return visible_path
    if source_subdir_for_mode(case.mode) != "nir":
        return visible_path
    return paired_nir_path(visible_path) or visible_path


def read_image_bgr(path: Path) -> np.ndarray:
    cv2 = import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def add_title(image: np.ndarray, title: str, color: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    cv2 = import_cv2()
    pad = 36
    canvas = np.full((image.shape[0] + pad, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[pad:] = image
    cv2.putText(canvas, title[:76], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return canvas


def resize_panel(panel: np.ndarray, width: int) -> np.ndarray:
    cv2 = import_cv2()
    if panel.shape[1] == width:
        return panel
    height = int(round(panel.shape[0] * (width / panel.shape[1])))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def save_grid(path: Path, panels: list[np.ndarray], panel_width: int) -> None:
    cv2 = import_cv2()
    resized = [resize_panel(panel, panel_width) for panel in panels]
    target_h = min(panel.shape[0] for panel in resized)
    aligned = [panel[:target_h] if panel.shape[0] != target_h else panel for panel in resized]
    cv2.imwrite(str(path), cv2.hconcat(aligned))


def load_gt_boxes(visible_path: Path, image_shape: tuple[int, int]) -> dict[str, torch.Tensor]:
    label_path = visible_path.with_suffix(".txt")
    h, w = image_shape
    boxes: list[list[float]] = []
    labels: list[int] = []
    if label_path.exists():
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if len(parts) == 5:
                cx, cy, bw, bh = [float(value) for value in parts[1:5]]
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
            else:
                coords = [float(value) for value in parts[1:]]
                xs = coords[0::2]
                ys = coords[1::2]
                if not xs or not ys:
                    continue
                x1, x2 = min(xs) * w, max(xs) * w
                y1, y2 = min(ys) * h, max(ys) * h
            x1 = float(np.clip(x1, 0, w - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            y2 = float(np.clip(y2, 0, h - 1))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def prediction_from_result(result: Any, conf: float) -> dict[str, torch.Tensor]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    scores = boxes.conf.detach().cpu().float()
    keep = scores >= conf
    return {
        "boxes": boxes.xyxy.detach().cpu().float()[keep],
        "scores": scores[keep],
        "labels": boxes.cls.detach().cpu().long()[keep],
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def external_id_to_stem_map(path: Path) -> dict[Any, str]:
    data = load_json(path)
    mapping: dict[Any, str] = {}
    for image in data.get("images", []):
        stem = Path(str(image["file_name"])).stem
        mapping[int(image["id"])] = stem
        mapping[str(image["id"])] = stem
    return mapping


def load_coco_records_by_stem(
    path: Path,
    images: list[Path],
    class_id_base: int,
    num_classes: int,
    external_gt_json: str = "",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    raw = load_json(path)
    if isinstance(raw, dict) and "annotations" in raw:
        raw = raw["annotations"]
    if not isinstance(raw, list):
        raise ValueError(f"COCO prediction JSON must be a list or contain annotations: {path}")

    stem_by_current_id = {idx: image_path.stem for idx, image_path in enumerate(images, start=1)}
    remap = external_id_to_stem_map(Path(external_gt_json).expanduser().resolve()) if external_gt_json else {}
    out: dict[str, list[dict[str, Any]]] = {}
    skipped = {"image_id_missing": 0, "class_out_of_range": 0, "invalid_bbox": 0}

    for item in raw:
        raw_image_id = item.get("image_id")
        stem = remap.get(raw_image_id, remap.get(str(raw_image_id)))
        if stem is None:
            try:
                stem = stem_by_current_id[int(raw_image_id)]
            except (KeyError, TypeError, ValueError):
                skipped["image_id_missing"] += 1
                continue
        cls = int(item["category_id"]) - class_id_base
        if not 0 <= cls < num_classes:
            skipped["class_out_of_range"] += 1
            continue
        bbox = [float(value) for value in item["bbox"]]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            skipped["invalid_bbox"] += 1
            continue
        out.setdefault(stem, []).append({"bbox": bbox, "score": float(item.get("score", 1.0)), "label": cls})
    return out, skipped


def inverse_letterbox_xyxy(box: list[float], width: int, height: int, imgsz: int) -> list[float]:
    gain = min(float(imgsz) / float(width), float(imgsz) / float(height))
    new_width = round(width * gain)
    new_height = round(height * gain)
    pad_x = (imgsz - new_width) / 2.0
    pad_y = (imgsz - new_height) / 2.0
    x1, y1, x2, y2 = box
    out = [(x1 - pad_x) / gain, (y1 - pad_y) / gain, (x2 - pad_x) / gain, (y2 - pad_y) / gain]
    return [
        min(max(float(out[0]), 0.0), float(width)),
        min(max(float(out[1]), 0.0), float(height)),
        min(max(float(out[2]), 0.0), float(width)),
        min(max(float(out[3]), 0.0), float(height)),
    ]


def prediction_from_coco_records(
    records: list[dict[str, Any]],
    image_shape: tuple[int, int],
    conf: float,
    imgsz: int,
    prediction_space: str,
) -> dict[str, torch.Tensor]:
    h, w = image_shape
    boxes: list[list[float]] = []
    scores: list[float] = []
    labels: list[int] = []
    for record in records:
        score = float(record["score"])
        if score < conf:
            continue
        x, y, bw, bh = [float(value) for value in record["bbox"]]
        xyxy = [x, y, x + bw, y + bh]
        if prediction_space == "letterbox":
            xyxy = inverse_letterbox_xyxy(xyxy, w, h, imgsz)
        else:
            xyxy = [
                min(max(float(xyxy[0]), 0.0), float(w)),
                min(max(float(xyxy[1]), 0.0), float(h)),
                min(max(float(xyxy[2]), 0.0), float(w)),
                min(max(float(xyxy[3]), 0.0), float(h)),
            ]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        boxes.append(xyxy)
        scores.append(score)
        labels.append(int(record["label"]))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-12)


def match_predictions(
    pred: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    iou_threshold: float,
) -> dict[str, Any]:
    pred_boxes = pred["boxes"]
    pred_labels = pred["labels"]
    pred_scores = pred["scores"]
    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]
    order = torch.argsort(pred_scores, descending=True).tolist()
    matched_gt: set[int] = set()
    tp_pred: list[int] = []
    fp_pred: list[int] = []
    pred_ious: dict[int, float] = {}

    for pred_idx in order:
        same_class = [idx for idx, label in enumerate(gt_labels.tolist()) if idx not in matched_gt and int(label) == int(pred_labels[pred_idx])]
        if not same_class:
            fp_pred.append(pred_idx)
            continue
        ious = box_iou(pred_boxes[pred_idx].unsqueeze(0), gt_boxes[same_class]).squeeze(0)
        best_iou, best_local_idx = torch.max(ious, dim=0)
        if float(best_iou) >= iou_threshold:
            gt_idx = same_class[int(best_local_idx)]
            matched_gt.add(gt_idx)
            tp_pred.append(pred_idx)
            pred_ious[pred_idx] = float(best_iou)
        else:
            fp_pred.append(pred_idx)

    fn_gt = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
    return {
        "tp_pred": tp_pred,
        "fp_pred": fp_pred,
        "fn_gt": fn_gt,
        "matched_gt": matched_gt,
        "tp_iou_mean": float(np.mean(list(pred_ious.values()))) if pred_ious else 0.0,
    }


def draw_box(
    image: np.ndarray,
    box: torch.Tensor | np.ndarray,
    text: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2 = import_cv2()
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, text, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def draw_gt_panel(base: np.ndarray, gt: dict[str, torch.Tensor], class_names: list[str]) -> np.ndarray:
    panel = base.copy()
    for box, label in zip(gt["boxes"], gt["labels"]):
        cls = int(label)
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        draw_box(panel, box, f"GT {name}", GT_COLOR, 2)
    return add_title(panel, "GT", GT_COLOR)


def draw_prediction_panel(
    base: np.ndarray,
    case_name: str,
    pred: dict[str, torch.Tensor],
    gt: dict[str, torch.Tensor],
    match: dict[str, Any],
    class_names: list[str],
    add_panel_title: bool = True,
) -> np.ndarray:
    panel = base.copy()
    for pred_idx in match["tp_pred"]:
        cls = int(pred["labels"][pred_idx])
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        score = float(pred["scores"][pred_idx])
        draw_box(panel, pred["boxes"][pred_idx], f"TP {name} {score:.2f}", TP_COLOR, 3)
    for pred_idx in match["fp_pred"]:
        cls = int(pred["labels"][pred_idx])
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        score = float(pred["scores"][pred_idx])
        draw_box(panel, pred["boxes"][pred_idx], f"FP {name} {score:.2f}", FP_COLOR, 2)
    for gt_idx in match["fn_gt"]:
        cls = int(gt["labels"][gt_idx])
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        draw_box(panel, gt["boxes"][gt_idx], f"FN {name}", FN_COLOR, 3)
    return add_title(panel, case_name) if add_panel_title else panel


def has_all_missed_small_target(
    gt: dict[str, torch.Tensor],
    matches: dict[str, dict[str, Any]],
    small_area: float,
) -> bool:
    if len(gt["boxes"]) == 0:
        return False
    widths = (gt["boxes"][:, 2] - gt["boxes"][:, 0]).clamp(min=0)
    heights = (gt["boxes"][:, 3] - gt["boxes"][:, 1]).clamp(min=0)
    small_indices = [idx for idx, area in enumerate((widths * heights).tolist()) if area < small_area]
    if not small_indices:
        return False
    for gt_idx in small_indices:
        if all(gt_idx not in match["matched_gt"] for match in matches.values()):
            return True
    return False


def choose_category(
    baseline_match: dict[str, Any],
    proposed_match: dict[str, Any],
    gt: dict[str, torch.Tensor],
    matches: dict[str, dict[str, Any]],
    small_area: float,
) -> str | None:
    if len(proposed_match["tp_pred"]) > len(baseline_match["tp_pred"]) or len(proposed_match["fn_gt"]) < len(baseline_match["fn_gt"]):
        return "proposed_detects_baseline_misses"
    if proposed_match["tp_iou_mean"] > baseline_match["tp_iou_mean"] + 0.05 and proposed_match["tp_iou_mean"] > 0:
        return "better_localization"
    if len(proposed_match["fp_pred"]) > 0:
        return "proposed_false_positive"
    if has_all_missed_small_target(gt, matches, small_area):
        return "all_miss_small_target"
    return None


def image_gt_allowed(gt_count: int, gt_min: int, gt_max: int) -> bool:
    if gt_min > 0 and gt_count < gt_min:
        return False
    if gt_max > 0 and gt_count > gt_max:
        return False
    return True


def case_selection_score(match: dict[str, Any]) -> tuple[int, int, int, float]:
    return (
        int(len(match["tp_pred"])),
        -int(len(match["fn_gt"])),
        -int(len(match["fp_pred"])),
        float(match["tp_iou_mean"]),
    )


def non_proposed_cases_differ(case_names: list[str], proposed_name: str, matches: dict[str, dict[str, Any]]) -> bool:
    metrics = {
        (
            len(matches[name]["tp_pred"]),
            len(matches[name]["fp_pred"]),
            len(matches[name]["fn_gt"]),
            round(float(matches[name]["tp_iou_mean"]), 4),
        )
        for name in case_names
        if name != proposed_name
    }
    return len(metrics) > 1


def proposed_better_than_all(
    case_names: list[str],
    proposed_name: str,
    matches: dict[str, dict[str, Any]],
) -> bool:
    proposed_score = case_selection_score(matches[proposed_name])
    return all(proposed_score > case_selection_score(matches[name]) for name in case_names if name != proposed_name)


def build_comparison_panels(
    base: np.ndarray,
    nir_image: np.ndarray,
    gt: dict[str, torch.Tensor],
    predictions: dict[str, dict[str, torch.Tensor]],
    matches: dict[str, dict[str, Any]],
    cases: list[CaseSpec],
    class_names: list[str],
    model_only: bool,
) -> list[np.ndarray]:
    panels: list[np.ndarray] = []
    if not model_only:
        panels.extend(
            [
                add_title(base.copy(), "RGB"),
                add_title(nir_image.copy(), "NIR"),
                draw_gt_panel(base, gt, class_names),
            ]
        )
    for case in cases:
        panels.append(draw_prediction_panel(base, case.name, predictions[case.name], gt, matches[case.name], class_names))
    return panels


def save_model_panels(
    out_dir: Path,
    stem: str,
    base: np.ndarray,
    gt: dict[str, torch.Tensor],
    predictions: dict[str, dict[str, torch.Tensor]],
    matches: dict[str, dict[str, Any]],
    cases: list[CaseSpec],
    class_names: list[str],
    panel_width: int,
    plain: bool = False,
    keep_size: bool = False,
) -> list[str]:
    cv2 = import_cv2()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for case in cases:
        panel = draw_prediction_panel(
            base,
            case.name,
            predictions[case.name],
            gt,
            matches[case.name],
            class_names,
            add_panel_title=not plain,
        )
        if not keep_size:
            panel = resize_panel(panel, panel_width)
        out_path = out_dir / f"{stem}_{sanitize(case.name)}.jpg"
        cv2.imwrite(str(out_path), panel)
        saved.append(str(out_path))
    return saved


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()
    cases = [parse_case(raw) for raw in args.case]
    if len(cases) < 1:
        raise RuntimeError("At least one --case is required.")

    baseline_name = args.baseline or (cases[-2].name if len(cases) >= 2 else cases[0].name)
    proposed_name = args.proposed or cases[-1].name
    if baseline_name not in {case.name for case in cases}:
        raise ValueError(f"--baseline must match a case name: {baseline_name}")
    if proposed_name not in {case.name for case in cases}:
        raise ValueError(f"--proposed must match a case name: {proposed_name}")

    images = resolve_images(args, cases)
    class_mode = args.dataset_mode or first_yolo_mode(cases)
    class_names = category_names_for_mode(class_mode)
    from ultralytics import YOLO

    models = {case.name: YOLO(str(case.weights)) for case in cases if case.backend == "yolo" and case.weights is not None}
    coco_records_by_case: dict[str, dict[str, list[dict[str, Any]]]] = {}
    coco_skipped_by_case: dict[str, dict[str, int]] = {}
    for case in cases:
        if case.backend != "coco" or case.predictions is None:
            continue
        records, skipped = load_coco_records_by_stem(
            case.predictions,
            images,
            args.coco_class_id_base,
            len(class_names),
            args.coco_gt_json,
        )
        coco_records_by_case[case.name] = records
        coco_skipped_by_case[case.name] = skipped
    out_dir = Path(args.out).expanduser().resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    saved_counts: dict[str, int] = {}
    best_per_case: dict[str, dict[str, Any]] = {}

    for visible_path in images:
        base = read_image_bgr(visible_path)
        nir_path = paired_nir_path(visible_path)
        nir_image = read_image_bgr(nir_path) if nir_path else np.zeros_like(base)
        gt = load_gt_boxes(visible_path, base.shape[:2])

        predictions: dict[str, dict[str, torch.Tensor]] = {}
        matches: dict[str, dict[str, Any]] = {}
        for case in cases:
            if case.backend == "yolo":
                source_path = image_path_for_case(case, visible_path)
                kwargs = mode_specific_kwargs(case.mode)
                result = models[case.name].predict(
                    source=str(source_path),
                    imgsz=args.imgsz,
                    device=args.device,
                    conf=min(args.predict_conf, args.conf),
                    iou=args.iou,
                    max_det=args.max_det,
                    half=args.half,
                    save=False,
                    verbose=False,
                    stream=False,
                    **kwargs,
                )[0]
                prediction = prediction_from_result(result, args.conf)
            else:
                prediction = prediction_from_coco_records(
                    coco_records_by_case.get(case.name, {}).get(visible_path.stem, []),
                    base.shape[:2],
                    args.conf,
                    args.imgsz,
                    args.coco_prediction_space,
                )
            match = match_predictions(prediction, gt, args.match_iou)
            predictions[case.name] = prediction
            matches[case.name] = match
            rows.append(
                {
                    "image": visible_path.name,
                    "case": case.name,
                    "mode": case.mode or case.backend,
                    "backend": case.backend,
                    "gt": int(len(gt["boxes"])),
                    "pred": int(len(prediction["boxes"])),
                    "tp": int(len(match["tp_pred"])),
                    "fp": int(len(match["fp_pred"])),
                    "fn": int(len(match["fn_gt"])),
                    "tp_iou_mean": float(match["tp_iou_mean"]),
                    "conf": args.conf,
                    "match_iou": args.match_iou,
                }
            )

        gt_allowed = image_gt_allowed(int(len(gt["boxes"])), args.gt_min, args.gt_max)
        if args.save_best_per_case and gt_allowed:
            for case in cases:
                score = case_selection_score(matches[case.name])
                prev = best_per_case.get(case.name)
                if prev is None or score > prev["score"]:
                    best_per_case[case.name] = {
                        "score": score,
                        "visible_path": visible_path,
                        "base": base.copy(),
                        "nir_image": nir_image.copy(),
                        "gt": gt,
                        "predictions": predictions,
                        "matches": matches,
                    }

        if not gt_allowed:
            continue

        if args.select_proposed_better and saved_counts.get("proposed_better_than_others", 0) < args.save_top_k:
            case_names = [case.name for case in cases]
            if proposed_better_than_all(case_names, proposed_name, matches) and non_proposed_cases_differ(case_names, proposed_name, matches):
                category_dir = images_dir / "proposed_better_than_others"
                category_dir.mkdir(parents=True, exist_ok=True)
                panels = build_comparison_panels(
                    base,
                    nir_image,
                    gt,
                    predictions,
                    matches,
                    cases,
                    class_names,
                    args.model_only,
                )
                out_path = category_dir / f"{visible_path.stem}_compare.jpg"
                save_grid(out_path, panels, args.panel_width)
                if args.save_model_panels:
                    save_model_panels(
                        category_dir / f"{visible_path.stem}_model_panels",
                        visible_path.stem,
                        base,
                        gt,
                        predictions,
                        matches,
                        cases,
                        class_names,
                        args.panel_width,
                        plain=args.plain_model_panels,
                        keep_size=args.keep_model_panel_size,
                    )
                saved_counts["proposed_better_than_others"] = saved_counts.get("proposed_better_than_others", 0) + 1
                selected_rows.append(
                    {
                        "image": visible_path.name,
                        "category": "proposed_better_than_others",
                        "path": str(out_path),
                        "baseline": baseline_name,
                        "proposed": proposed_name,
                        "baseline_tp": len(matches[baseline_name]["tp_pred"]),
                        "baseline_fp": len(matches[baseline_name]["fp_pred"]),
                        "baseline_fn": len(matches[baseline_name]["fn_gt"]),
                        "proposed_tp": len(matches[proposed_name]["tp_pred"]),
                        "proposed_fp": len(matches[proposed_name]["fp_pred"]),
                        "proposed_fn": len(matches[proposed_name]["fn_gt"]),
                        "baseline_iou": matches[baseline_name]["tp_iou_mean"],
                        "proposed_iou": matches[proposed_name]["tp_iou_mean"],
                    }
                )
                print(f"saved={out_path}")

        category = choose_category(
            matches[baseline_name],
            matches[proposed_name],
            gt,
            matches,
            args.small_area,
        )
        if category and saved_counts.get(category, 0) < args.save_top_k:
            category_dir = images_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            panels = build_comparison_panels(
                base,
                nir_image,
                gt,
                predictions,
                matches,
                cases,
                class_names,
                args.model_only,
            )
            out_path = category_dir / f"{visible_path.stem}_compare.jpg"
            save_grid(out_path, panels, args.panel_width)
            if args.save_model_panels:
                save_model_panels(
                    category_dir / f"{visible_path.stem}_model_panels",
                    visible_path.stem,
                    base,
                    gt,
                    predictions,
                    matches,
                    cases,
                    class_names,
                    args.panel_width,
                    plain=args.plain_model_panels,
                    keep_size=args.keep_model_panel_size,
                )
            saved_counts[category] = saved_counts.get(category, 0) + 1
            selected_rows.append(
                {
                    "image": visible_path.name,
                    "category": category,
                    "path": str(out_path),
                    "baseline": baseline_name,
                    "proposed": proposed_name,
                    "baseline_tp": len(matches[baseline_name]["tp_pred"]),
                    "baseline_fp": len(matches[baseline_name]["fp_pred"]),
                    "baseline_fn": len(matches[baseline_name]["fn_gt"]),
                    "proposed_tp": len(matches[proposed_name]["tp_pred"]),
                    "proposed_fp": len(matches[proposed_name]["fp_pred"]),
                    "proposed_fn": len(matches[proposed_name]["fn_gt"]),
                    "baseline_iou": matches[baseline_name]["tp_iou_mean"],
                    "proposed_iou": matches[proposed_name]["tp_iou_mean"],
                }
            )
            print(f"saved={out_path}")

    if args.save_best_per_case:
        best_dir = images_dir / "best_per_case"
        best_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            item = best_per_case.get(case.name)
            if item is None:
                continue
            visible_path = item["visible_path"]
            panels = build_comparison_panels(
                item["base"],
                item["nir_image"],
                item["gt"],
                item["predictions"],
                item["matches"],
                cases,
                class_names,
                args.model_only,
            )
            out_path = best_dir / f"{sanitize(case.name)}_{visible_path.stem}_compare.jpg"
            save_grid(out_path, panels, args.panel_width)
            if args.save_model_panels:
                save_model_panels(
                    best_dir / f"{sanitize(case.name)}_{visible_path.stem}_model_panels",
                    visible_path.stem,
                    item["base"],
                    item["gt"],
                    item["predictions"],
                    item["matches"],
                    cases,
                    class_names,
                    args.panel_width,
                    plain=args.plain_model_panels,
                    keep_size=args.keep_model_panel_size,
                )
            selected_rows.append(
                {
                    "image": visible_path.name,
                    "category": f"best_for_{case.name}",
                    "path": str(out_path),
                    "baseline": baseline_name,
                    "proposed": proposed_name,
                    "baseline_tp": len(item["matches"][baseline_name]["tp_pred"]),
                    "baseline_fp": len(item["matches"][baseline_name]["fp_pred"]),
                    "baseline_fn": len(item["matches"][baseline_name]["fn_gt"]),
                    "proposed_tp": len(item["matches"][proposed_name]["tp_pred"]),
                    "proposed_fp": len(item["matches"][proposed_name]["fp_pred"]),
                    "proposed_fn": len(item["matches"][proposed_name]["fn_gt"]),
                    "baseline_iou": item["matches"][baseline_name]["tp_iou_mean"],
                    "proposed_iou": item["matches"][proposed_name]["tp_iou_mean"],
                }
            )
            saved_counts[f"best_for_{case.name}"] = 1
            print(f"saved={out_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "case_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["image", "case", "mode", "backend", "gt", "pred", "tp", "fp", "fn", "tp_iou_mean", "conf", "match_iou"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "selected_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "image",
            "category",
            "path",
            "baseline",
            "proposed",
            "baseline_tp",
            "baseline_fp",
            "baseline_fn",
            "proposed_tp",
            "proposed_fp",
            "proposed_fn",
            "baseline_iou",
            "proposed_iou",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "cases": [
            {
                "name": case.name,
                "mode": case.mode,
                "backend": case.backend,
                "weights": str(case.weights) if case.weights else "",
                "predictions": str(case.predictions) if case.predictions else "",
            }
            for case in cases
        ],
        "baseline": baseline_name,
        "proposed": proposed_name,
        "num_images": len(images),
        "saved_counts": saved_counts,
        "gt_min": args.gt_min,
        "gt_max": args.gt_max,
        "coco_skipped_by_case": coco_skipped_by_case,
        "coco_prediction_space": args.coco_prediction_space,
        "coco_class_id_base": args.coco_class_id_base,
        "coco_gt_json": args.coco_gt_json,
        "conf": args.conf,
        "predict_conf": args.predict_conf,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "out": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"case_summary={out_dir / 'case_summary.csv'}")
    print(f"selected_cases={out_dir / 'selected_cases.csv'}")


if __name__ == "__main__":
    main()
