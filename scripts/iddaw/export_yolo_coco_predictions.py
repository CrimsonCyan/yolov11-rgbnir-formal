from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.iddaw import (  # noqa: E402
    TRAFFIC_DETECTABLE640_DATASET_NAME,
    category_names_for_mode,
    mode_specific_kwargs,
    resolve_dataset_root,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export YOLO detections to COCO prediction JSON and evaluate with pycocotools COCOeval. "
            "The default 8-class dataset is the filtered detectable640 bbox dataset."
        )
    )
    parser.add_argument("--mode", required=True, help="IDD-AW mode name, e.g. rgb_p2p5_yolo11s_8cls_personmerge_traffic.")
    parser.add_argument("--weights", required=True, help="YOLO checkpoint path, usually weights/best.pt.")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root override. 8cls bbox runs must be detectable640.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to export/evaluate.")
    parser.add_argument("--out", default="", help="Output directory. Defaults to runs/analysis/coco_eval/<name>.")
    parser.add_argument("--name", default="", help="Output run name. Defaults to <mode>_<split>_<weight-stem>.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--eval-space",
        choices=["letterbox", "original"],
        default="letterbox",
        help=(
            "Coordinate space for COCO JSON. Ultralytics prediction boxes are result.orig_shape native coordinates. "
            "'letterbox' maps both GT and predictions once into imgsz x imgsz, preserving the project's 640-scale AP_S/M/L buckets; "
            "'original' keeps native image coordinates."
        ),
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="Low export confidence for AP calculation.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU used during YOLO inference.")
    parser.add_argument("--max-det", type=int, default=300, help="YOLO inference max_det.")
    parser.add_argument(
        "--coco-max-det",
        type=int,
        default=100,
        help="Largest COCOeval maxDets entry. Standard COCO uses 100; set 300 to evaluate with maxDets=[1,10,300].",
    )
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported.")
    parser.add_argument("--max-images", type=int, default=0, help="Debug cap; 0 means full split.")
    parser.add_argument("--no-eval", action="store_true", help="Only export JSON files, do not run COCOeval.")
    parser.add_argument("--save-txt", action="store_true", help="Also let Ultralytics save YOLO-format predictions.")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def source_subdir_for_mode(mode: str) -> str:
    kwargs = mode_specific_kwargs(mode)
    return "nir" if kwargs.get("use_simotm") == "Gray" else "visible"


def default_bbox_dataset_root(mode: str, class_names: list[str]) -> Path:
    if len(class_names) != 8:
        return resolve_dataset_root(mode)
    # Keep this exporter on bbox labels even when the training mode itself used
    # the segment dataset for auxiliary supervision.
    env_value = os.getenv("IDDAW_YOLO_ROOT_8CLS_PERSONMERGE_TRAFFIC")
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            ROOT.parent / "datasets" / TRAFFIC_DETECTABLE640_DATASET_NAME,
            ROOT / "datasets" / TRAFFIC_DETECTABLE640_DATASET_NAME,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Could not find the 8cls detectable640 bbox dataset. Set IDDAW_YOLO_ROOT_8CLS_PERSONMERGE_TRAFFIC "
        f"or pass --dataset-root. Checked: {', '.join(str(path) for path in candidates)}"
    )


def list_images(source_dir: Path, max_images: int = 0) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {source_dir}")
    images = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise RuntimeError(f"No images found in {source_dir}")
    return images


def letterbox_params(width: int, height: int, imgsz: int) -> tuple[float, float, float]:
    if width <= 0 or height <= 0 or imgsz <= 0:
        raise ValueError(f"Invalid image or target size for letterbox conversion: width={width}, height={height}, imgsz={imgsz}")
    gain = min(float(imgsz) / float(width), float(imgsz) / float(height))
    new_width = round(width * gain)
    new_height = round(height * gain)
    pad_x = (imgsz - new_width) / 2.0
    pad_y = (imgsz - new_height) / 2.0
    return gain, pad_x, pad_y


def transform_xyxy_to_letterbox(box: list[float], width: int, height: int, imgsz: int) -> list[float]:
    gain, pad_x, pad_y = letterbox_params(width, height, imgsz)
    x1, y1, x2, y2 = box
    out = [x1 * gain + pad_x, y1 * gain + pad_y, x2 * gain + pad_x, y2 * gain + pad_y]
    return [
        min(max(float(out[0]), 0.0), float(imgsz)),
        min(max(float(out[1]), 0.0), float(imgsz)),
        min(max(float(out[2]), 0.0), float(imgsz)),
        min(max(float(out[3]), 0.0), float(imgsz)),
    ]


def xyxy_to_xywh(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(float(x1), 0.0), float(width))
    y1 = min(max(float(y1), 0.0), float(height))
    x2 = min(max(float(x2), 0.0), float(width))
    y2 = min(max(float(y2), 0.0), float(height))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def yolo_xywhn_to_xyxy(label_values: list[float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = label_values
    x = max(0.0, (cx - bw / 2.0) * width)
    y = max(0.0, (cy - bh / 2.0) * height)
    box_w = max(0.0, bw * width)
    box_h = max(0.0, bh * height)
    if x + box_w > width:
        box_w = max(0.0, width - x)
    if y + box_h > height:
        box_h = max(0.0, height - y)
    return [x, y, x + box_w, y + box_h]


def build_coco_gt(
    images: list[Path],
    class_names: list[str],
    imgsz: int,
    eval_space: str,
) -> tuple[dict[str, object], dict[str, int]]:
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    image_id_by_stem: dict[str, int] = {}
    ann_id = 1
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        coco_width, coco_height = (imgsz, imgsz) if eval_space == "letterbox" else (width, height)
        image_id_by_stem[image_path.stem] = image_id
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(coco_width),
                "height": int(coco_height),
            }
        )
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"Expected bbox YOLO label with 5 columns, got {len(parts)} at {label_path}:{line_no}. "
                    "Use the detectable640 bbox dataset, not the segment dataset."
                )
            cls = int(float(parts[0]))
            if not 0 <= cls < len(class_names):
                raise ValueError(f"Class id {cls} out of range 0..{len(class_names) - 1} at {label_path}:{line_no}")
            xyxy = yolo_xywhn_to_xyxy([float(value) for value in parts[1:5]], width, height)
            if eval_space == "letterbox":
                xyxy = transform_xyxy_to_letterbox(xyxy, width, height, imgsz)
            bbox = xyxy_to_xywh(xyxy, coco_width, coco_height)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(class_names)]
    return {
        "info": {"description": "IDD-AW YOLO-format bbox annotations converted to COCO"},
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": categories,
    }, image_id_by_stem


def write_source_list(images: list[Path], out_dir: Path) -> Path:
    """Write an image list so Ultralytics uses its file loader instead of in-memory list batching."""
    source_file = out_dir / "source_images.txt"
    source_file.write_text("\n".join(str(path) for path in images), encoding="utf-8")
    return source_file


def export_predictions(
    args: argparse.Namespace,
    images: list[Path],
    image_id_by_stem: dict[str, int],
    class_names: list[str],
    out_dir: Path,
) -> tuple[list[dict[str, object]], float, dict[str, int]]:
    from ultralytics import YOLO

    model = YOLO(str(Path(args.weights).expanduser().resolve()))
    mode_kwargs = mode_specific_kwargs(args.mode)
    source_file = write_source_list(images, out_dir)
    predictions: list[dict[str, object]] = []
    skipped = {
        "empty_results": 0,
        "class_out_of_range": 0,
        "invalid_bbox": 0,
    }
    start = time.perf_counter()
    results = model.predict(
        source=str(source_file),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        half=args.half,
        verbose=False,
        save=False,
        save_txt=args.save_txt,
        stream=True,
        **mode_kwargs,
    )
    for result in results:
        image_path = Path(result.path)
        image_id = image_id_by_stem.get(image_path.stem)
        if image_id is None:
            raise KeyError(f"Prediction image not found in GT image map: {image_path}")
        height, width = result.orig_shape
        coco_width, coco_height = (args.imgsz, args.imgsz) if args.eval_space == "letterbox" else (width, height)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            skipped["empty_results"] += 1
            continue
        xyxy = boxes.xyxy.detach().cpu().tolist()
        scores = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        for box, score, cls_value in zip(xyxy, scores, classes):
            cls = int(cls_value)
            if not 0 <= cls < len(class_names):
                skipped["class_out_of_range"] += 1
                continue
            if args.eval_space == "letterbox":
                box = transform_xyxy_to_letterbox(box, width, height, args.imgsz)
            bbox = xyxy_to_xywh(box, coco_width, coco_height)
            if bbox[2] <= 0 or bbox[3] <= 0:
                skipped["invalid_bbox"] += 1
                continue
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cls + 1),
                    "bbox": [float(value) for value in bbox],
                    "score": float(score),
                }
            )
    elapsed = time.perf_counter() - start
    if not predictions:
        raise RuntimeError(
            "No detections were exported. "
            f"Skipped counts: {skipped}. Check weights, modality, dataset path, confidence threshold, and class mapping."
        )
    return predictions, elapsed, skipped


def run_coco_eval(coco_gt_data: dict[str, object], detections: list[dict[str, object]], coco_max_det: int):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("COCOeval requires pycocotools. Install it in the active environment.") from exc
    stdout_buffer = StringIO()
    with redirect_stdout(stdout_buffer):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_data
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [int(image["id"]) for image in coco_gt_data["images"]]
        evaluator.params.catIds = [int(category["id"]) for category in coco_gt_data["categories"]]
        evaluator.params.maxDets = [1, 10, int(coco_max_det)]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator, stdout_buffer.getvalue()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def class_area_rows(evaluator, class_names: list[str], coco_gt_data: dict[str, object]) -> list[dict[str, object]]:
    precision = evaluator.eval["precision"]  # [TxRxKxAxM]
    area_labels = list(evaluator.params.areaRngLbl)
    gt_counts = gt_counts_by_class_area(coco_gt_data, class_names, area_labels)
    rows: list[dict[str, object]] = []
    for area_index, area_name in enumerate(area_labels):
        for cls, class_name in enumerate(class_names):
            values = precision[:, :, cls, area_index, -1]
            valid = values[values > -1]
            ap = float(valid.mean()) if valid.size else 0.0
            ap50_values = precision[0, :, cls, area_index, -1]
            ap50_valid = ap50_values[ap50_values > -1]
            ap50 = float(ap50_valid.mean()) if ap50_valid.size else 0.0
            rows.append(
                {
                    "class": class_name,
                    "area": area_name,
                    "gt_count": gt_counts[area_name][cls],
                    "AP50": ap50,
                    "mAP50_95": ap,
                }
            )
    for area_name in area_labels:
        subset = [row for row in rows if row["area"] == area_name and row["class"] != "mean" and int(row["gt_count"]) > 0]
        rows.append(
            {
                "class": "mean",
                "area": area_name,
                "gt_count": sum(int(row["gt_count"]) for row in subset),
                "AP50": mean([float(row["AP50"]) for row in subset]),
                "mAP50_95": mean([float(row["mAP50_95"]) for row in subset]),
            }
        )
    return rows


def gt_counts_by_class_area(coco_gt_data: dict[str, object], class_names: list[str], area_labels: list[str]) -> dict[str, dict[int, int]]:
    # COCO default area buckets: all, small < 32^2, medium 32^2..96^2, large >= 96^2.
    counts = {area: {cls: 0 for cls in range(len(class_names))} for area in area_labels}
    for annotation in coco_gt_data["annotations"]:
        cls = int(annotation["category_id"]) - 1
        area = float(annotation["area"])
        counts["all"][cls] += 1
        if area < 32.0 * 32.0:
            counts["small"][cls] += 1
        elif area < 96.0 * 96.0:
            counts["medium"][cls] += 1
        else:
            counts["large"][cls] += 1
    return counts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    class_names = category_names_for_mode(args.mode)
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else default_bbox_dataset_root(args.mode, class_names)
    if len(class_names) == 8 and dataset_root.name != TRAFFIC_DETECTABLE640_DATASET_NAME:
        raise ValueError(
            "8cls traffic COCO export must use the filtered detectable640 bbox dataset.\n"
            f"Expected: {TRAFFIC_DETECTABLE640_DATASET_NAME}\nGot: {dataset_root}"
        )
    source_dir = dataset_root / source_subdir_for_mode(args.mode) / args.split
    images = list_images(source_dir, args.max_images)
    coco_gt, image_id_by_stem = build_coco_gt(images, class_names, args.imgsz, args.eval_space)
    run_name = args.name or f"{args.mode}_{args.split}_{Path(args.weights).stem}"
    out_dir = Path(args.out).expanduser().resolve() if args.out else ROOT / "runs" / "analysis" / "coco_eval" / sanitize_name(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions, elapsed, skipped_predictions = export_predictions(args, images, image_id_by_stem, class_names, out_dir)
    gt_json = out_dir / "instances_gt.json"
    pred_json = out_dir / "predictions.json"
    gt_json.write_text(json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8")
    pred_json.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    summary: dict[str, object] = {
        "mode": args.mode,
        "weights": str(Path(args.weights).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "source_dir": str(source_dir),
        "split": args.split,
        "num_images": len(images),
        "num_gt": len(coco_gt["annotations"]),
        "num_predictions": len(predictions),
        "skipped_predictions": skipped_predictions,
        "imgsz": args.imgsz,
        "eval_space": args.eval_space,
        "prediction_box_space": "result.orig_shape/native before optional eval-space transform",
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "nms_iou": args.iou,
        "max_det": args.max_det,
        "coco_max_dets": [1, 10, int(args.coco_max_det)],
        "infer_wall_sec": elapsed,
        "infer_ms_per_img_wall": elapsed * 1000.0 / max(len(images), 1),
        "fps_wall": len(images) / max(elapsed, 1e-9),
        "gt_json": str(gt_json),
        "pred_json": str(pred_json),
    }

    if not args.no_eval:
        evaluator, eval_text = run_coco_eval(coco_gt, predictions, args.coco_max_det)
        (out_dir / "coco_eval.txt").write_text(eval_text, encoding="utf-8")
        rows = class_area_rows(evaluator, class_names, coco_gt)
        write_csv(out_dir / "metrics_by_class_area.csv", rows)
        summary.update(
            {
                "mAP50_95": float(evaluator.stats[0]),
                "mAP50": float(evaluator.stats[1]),
                "mAP75": float(evaluator.stats[2]),
                "AP_S": float(evaluator.stats[3]),
                "AP_M": float(evaluator.stats[4]),
                "AP_L": float(evaluator.stats[5]),
                "AR_1": float(evaluator.stats[6]),
                "AR_10": float(evaluator.stats[7]),
                f"AR_{int(args.coco_max_det)}": float(evaluator.stats[8]),
            }
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
