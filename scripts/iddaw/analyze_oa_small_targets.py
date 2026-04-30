from __future__ import annotations

import argparse
import csv
import json
import sys
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


def ap_for_threshold(predictions, targets, class_id: int, iou_threshold: float, area_filter: str | None) -> float:
    pred_records = []
    gt_map = {}
    for pred, target in zip(predictions, targets):
        image_id = target["sample_id"]
        gt_entries = []
        for box, label, bucket in zip(target["boxes_xyxy"], target["labels"], target["area_buckets"]):
            if int(label) == class_id and (area_filter is None or bucket == area_filter):
                gt_entries.append([box, False])
        gt_map[image_id] = gt_entries
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if int(label) == class_id:
                pred_records.append((image_id, float(score), box))
    pred_records.sort(key=lambda item: item[1], reverse=True)
    total_gts = sum(len(items) for items in gt_map.values())
    if total_gts == 0:
        return 0.0
    tps, fps = [], []
    for image_id, _, pred_box in pred_records:
        gt_entries = gt_map.get(image_id, [])
        if not gt_entries:
            tps.append(0.0)
            fps.append(1.0)
            continue
        gt_boxes = torch.stack([entry[0] for entry in gt_entries], dim=0)
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        matched = gt_entries[int(best_idx)][1]
        if float(best_iou) >= iou_threshold and not matched:
            gt_entries[int(best_idx)][1] = True
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
    ap = 0.0
    for level in torch.linspace(0, 1, 101):
        valid = precisions[recalls >= level]
        ap += float(valid.max()) if valid.numel() > 0 else 0.0
    return ap / 101.0


def pr_at_conf(predictions, targets, class_id: int, area_filter: str | None, conf: float, iou_threshold: float = 0.5):
    filtered = []
    for pred in predictions:
        keep = pred["scores"] >= conf
        filtered.append({"boxes": pred["boxes"][keep], "scores": pred["scores"][keep], "labels": pred["labels"][keep]})
    pred_records = []
    gt_map = {}
    for pred, target in zip(filtered, targets):
        image_id = target["sample_id"]
        gt_entries = []
        for box, label, bucket in zip(target["boxes_xyxy"], target["labels"], target["area_buckets"]):
            if int(label) == class_id and (area_filter is None or bucket == area_filter):
                gt_entries.append([box, False])
        gt_map[image_id] = gt_entries
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            if int(label) == class_id:
                pred_records.append((image_id, float(score), box))
    pred_records.sort(key=lambda item: item[1], reverse=True)
    tp = fp = 0
    for image_id, _, pred_box in pred_records:
        gt_entries = gt_map.get(image_id, [])
        if not gt_entries:
            fp += 1
            continue
        gt_boxes = torch.stack([entry[0] for entry in gt_entries], dim=0)
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        matched = gt_entries[int(best_idx)][1]
        if float(best_iou) >= iou_threshold and not matched:
            gt_entries[int(best_idx)][1] = True
            tp += 1
        else:
            fp += 1
    total_gts = sum(len(items) for items in gt_map.values())
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

    for result in model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=1,
        stream=True,
        verbose=False,
        save=False,
        **mode_kwargs,
    ):
        image_path = Path(result.path)
        target = load_target(image_path, result.orig_shape)
        targets.append(target)
        predictions.append(collect_prediction(result))
        update_gate_stats(gate_stats, modules, target, class_names)

    metric_rows = summarize_metrics(predictions, targets, class_names, args.pr_conf)
    gate_summary = {key: mean(values) for key, values in gate_stats.items()}
    gate_summary["num_gate_samples"] = len(gate_stats["gate_all"])
    return {
        "name": case.name,
        "mode": case.mode,
        "weights": str(case.weights),
        "num_images": len(images),
        "metrics": metric_rows,
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
