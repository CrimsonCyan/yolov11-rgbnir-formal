from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.iddaw import category_names_for_mode  # noqa: E402
from scripts.iddaw.export_yolo_coco_predictions import (  # noqa: E402
    build_coco_gt,
    class_area_rows,
    default_bbox_dataset_root,
    list_images,
    run_coco_eval,
    sanitize_name,
    source_subdir_for_mode,
    transform_xyxy_to_letterbox,
    xyxy_to_xywh,
    yolo_xywhn_to_xyxy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external detector predictions with the same IDD-AW COCOeval protocol. "
            "This script does not run inference and therefore does not mix COCO conversion time into FPS."
        )
    )
    parser.add_argument("--mode", required=True, help="Mode name only used to infer class names and default modality.")
    parser.add_argument("--dataset-root", default="", help="Dataset root. 8cls should point to detectable640 bbox data.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--image-subdir",
        default="",
        choices=["", "visible", "nir"],
        help="Override image subdir. Default follows mode, but external RGB-NIR methods should usually use visible.",
    )
    parser.add_argument("--predictions", required=True, help="Prediction file or per-image txt directory.")
    parser.add_argument(
        "--pred-format",
        choices=["coco", "yolo-txt"],
        default="coco",
        help="coco: COCO detection JSON. yolo-txt: one txt per image with configurable row layout.",
    )
    parser.add_argument(
        "--txt-layout",
        choices=["cls_conf_xyxy", "xyxy_conf_cls", "cls_xywhn_conf", "cls_xywh_conf", "imageid_xywh_conf"],
        default="cls_conf_xyxy",
        help="Layout for --pred-format yolo-txt. Comma and whitespace delimiters are both accepted.",
    )
    parser.add_argument("--default-class", type=int, default=0, help="Class id used by layouts without class column.")
    parser.add_argument(
        "--class-id-base",
        type=int,
        default=-1,
        help="0 if class ids are 0-based, 1 if already COCO-style. Default -1 means: coco=>1, yolo-txt=>0.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--eval-space", choices=["letterbox", "original"], default="letterbox")
    parser.add_argument(
        "--prediction-space",
        choices=["letterbox", "original"],
        default="letterbox",
        help="Coordinate space of prediction boxes before optional conversion into --eval-space.",
    )
    parser.add_argument(
        "--prediction-gt-json",
        default="",
        help=(
            "Optional COCO GT JSON used by the external method. With --remap-image-ids, prediction image_id values "
            "are mapped through file_name/stem into this script's GT ids."
        ),
    )
    parser.add_argument("--remap-image-ids", action="store_true")
    parser.add_argument("--coco-max-det", type=int, default=100)
    parser.add_argument("--timing-json", default="", help="Optional sidecar timing JSON emitted by native inference.")
    parser.add_argument("--out", default="", help="Output directory. Defaults to runs/analysis/coco_eval/<name>.")
    parser.add_argument("--name", default="", help="Output run name.")
    return parser.parse_args()


def split_row(raw: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", raw.strip()) if part]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def image_maps(coco_gt: dict[str, object]) -> tuple[dict[str, int], dict[int, dict[str, object]]]:
    by_stem = {Path(str(image["file_name"])).stem: int(image["id"]) for image in coco_gt["images"]}
    by_id = {int(image["id"]): image for image in coco_gt["images"]}
    return by_stem, by_id


def runtime_image_info(images: list[Path], coco_gt: dict[str, object]) -> dict[int, dict[str, object]]:
    stem_to_id, coco_by_id = image_maps(coco_gt)
    out: dict[int, dict[str, object]] = {}
    for image_path in images:
        image_id = stem_to_id[image_path.stem]
        with Image.open(image_path) as image:
            width, height = image.size
        info = dict(coco_by_id[image_id])
        info["original_width"] = int(width)
        info["original_height"] = int(height)
        out[image_id] = info
    return out


def external_id_to_stem_map(path: Path) -> dict[Any, str]:
    data = load_json(path)
    mapping: dict[Any, str] = {}
    for image in data.get("images", []):
        stem = Path(str(image["file_name"])).stem
        mapping[int(image["id"])] = stem
        mapping[str(image["id"])] = stem
    return mapping


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


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def convert_box_space(
    bbox_xywh: list[float],
    image_info: dict[str, object],
    imgsz: int,
    prediction_space: str,
    eval_space: str,
) -> list[float]:
    width = int(image_info.get("original_width", image_info["width"]))
    height = int(image_info.get("original_height", image_info["height"]))
    xyxy = xywh_to_xyxy(bbox_xywh)
    if prediction_space == eval_space:
        target_w, target_h = (imgsz, imgsz) if eval_space == "letterbox" else (width, height)
        return xyxy_to_xywh(xyxy, target_w, target_h)
    if prediction_space == "original" and eval_space == "letterbox":
        xyxy = transform_xyxy_to_letterbox(xyxy, width, height, imgsz)
        return xyxy_to_xywh(xyxy, imgsz, imgsz)
    if prediction_space == "letterbox" and eval_space == "original":
        xyxy = inverse_letterbox_xyxy(xyxy, width, height, imgsz)
        return xyxy_to_xywh(xyxy, width, height)
    raise ValueError(f"Unsupported box conversion: {prediction_space} -> {eval_space}")


def category_id_from_external(cls: int, class_id_base: int, num_classes: int) -> int | None:
    zero_based = cls - class_id_base
    if not 0 <= zero_based < num_classes:
        return None
    return zero_based + 1


def load_coco_predictions(
    path: Path,
    coco_gt: dict[str, object],
    image_info_by_id: dict[int, dict[str, object]],
    args: argparse.Namespace,
    class_names: list[str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    raw_predictions = load_json(path)
    if isinstance(raw_predictions, dict) and "annotations" in raw_predictions:
        raw_predictions = raw_predictions["annotations"]
    if not isinstance(raw_predictions, list):
        raise ValueError(f"COCO prediction JSON must be a list or contain annotations list: {path}")
    current_stem_to_id, current_id_to_image = image_maps(coco_gt)
    remap: dict[Any, int] = {}
    if args.remap_image_ids:
        if not args.prediction_gt_json:
            raise ValueError("--remap-image-ids requires --prediction-gt-json")
        for external_id, stem in external_id_to_stem_map(Path(args.prediction_gt_json).expanduser().resolve()).items():
            if stem in current_stem_to_id:
                remap[external_id] = current_stem_to_id[stem]
    out: list[dict[str, object]] = []
    skipped = {"image_id_missing": 0, "class_out_of_range": 0, "invalid_bbox": 0}
    for item in raw_predictions:
        raw_image_id = item.get("image_id")
        image_id = remap.get(raw_image_id, remap.get(str(raw_image_id), raw_image_id))
        try:
            image_id = int(image_id)
        except (TypeError, ValueError):
            skipped["image_id_missing"] += 1
            continue
        if image_id not in current_id_to_image:
            skipped["image_id_missing"] += 1
            continue
        image_info = image_info_by_id[image_id]
        category_id = category_id_from_external(int(item["category_id"]), args.class_id_base, len(class_names))
        if category_id is None:
            skipped["class_out_of_range"] += 1
            continue
        bbox = convert_box_space(
            [float(value) for value in item["bbox"]],
            image_info,
            args.imgsz,
            args.prediction_space,
            args.eval_space,
        )
        if bbox[2] <= 0 or bbox[3] <= 0:
            skipped["invalid_bbox"] += 1
            continue
        out.append({"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": float(item.get("score", 1.0))})
    return out, skipped


def txt_row_to_prediction(
    parts: list[str],
    layout: str,
    default_class: int,
    class_id_base: int,
    class_names: list[str],
    image_id: int,
    image_path: Path,
    image_info: dict[str, object],
    imgsz: int,
    prediction_space: str,
    eval_space: str,
) -> dict[str, object] | None:
    with Image.open(image_path) as image:
        width, height = image.size
    if layout == "cls_conf_xyxy":
        cls = int(float(parts[0]))
        score = float(parts[1])
        xyxy = [float(value) for value in parts[2:6]]
        bbox = xyxy_to_xywh(xyxy, width if prediction_space == "original" else imgsz, height if prediction_space == "original" else imgsz)
    elif layout == "xyxy_conf_cls":
        xyxy = [float(value) for value in parts[0:4]]
        score = float(parts[4])
        cls = int(float(parts[5]))
        bbox = xyxy_to_xywh(xyxy, width if prediction_space == "original" else imgsz, height if prediction_space == "original" else imgsz)
    elif layout == "cls_xywhn_conf":
        cls = int(float(parts[0]))
        score = float(parts[5])
        xyxy = yolo_xywhn_to_xyxy([float(value) for value in parts[1:5]], width, height)
        bbox = xyxy_to_xywh(xyxy, width, height)
        prediction_space = "original"
    elif layout == "cls_xywh_conf":
        cls = int(float(parts[0]))
        bbox = [float(value) for value in parts[1:5]]
        score = float(parts[5])
    elif layout == "imageid_xywh_conf":
        cls = default_class
        bbox = [float(value) for value in parts[1:5]]
        score = float(parts[5])
    else:
        raise ValueError(layout)
    category_id = category_id_from_external(cls, class_id_base, len(class_names))
    if category_id is None:
        return None
    bbox = convert_box_space(bbox, image_info, imgsz, prediction_space, eval_space)
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}


def load_txt_predictions(
    pred_dir: Path,
    images: list[Path],
    coco_gt: dict[str, object],
    image_info_by_id: dict[int, dict[str, object]],
    args: argparse.Namespace,
    class_names: list[str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    stem_to_id, _ = image_maps(coco_gt)
    predictions: list[dict[str, object]] = []
    skipped = {"missing_txt": 0, "bad_row": 0, "class_or_bbox_invalid": 0}
    for image_path in images:
        txt_path = pred_dir / f"{image_path.stem}.txt"
        if not txt_path.exists():
            skipped["missing_txt"] += 1
            continue
        image_id = stem_to_id[image_path.stem]
        image_info = image_info_by_id[image_id]
        for raw in txt_path.read_text(encoding="utf-8").splitlines():
            parts = split_row(raw)
            min_cols = 6
            if len(parts) < min_cols:
                skipped["bad_row"] += 1
                continue
            pred = txt_row_to_prediction(
                parts,
                args.txt_layout,
                args.default_class,
                args.class_id_base,
                class_names,
                image_id,
                image_path,
                image_info,
                args.imgsz,
                args.prediction_space,
                args.eval_space,
            )
            if pred is None:
                skipped["class_or_bbox_invalid"] += 1
                continue
            predictions.append(pred)
    return predictions, skipped


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_timing(path: str) -> dict[str, object]:
    if not path:
        return {}
    timing = load_json(Path(path).expanduser().resolve())
    if not isinstance(timing, dict):
        raise ValueError("--timing-json must contain a JSON object")
    return timing


def main() -> None:
    args = parse_args()
    class_names = category_names_for_mode(args.mode)
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else default_bbox_dataset_root(args.mode, class_names)
    image_subdir = args.image_subdir or source_subdir_for_mode(args.mode)
    images = list_images(dataset_root / image_subdir / args.split)
    coco_gt, _ = build_coco_gt(images, class_names, args.imgsz, args.eval_space)
    image_info_by_id = runtime_image_info(images, coco_gt)
    if args.class_id_base < 0:
        args.class_id_base = 1 if args.pred_format == "coco" else 0
    predictions_path = Path(args.predictions).expanduser().resolve()
    convert_start = time.perf_counter()
    if args.pred_format == "coco":
        detections, skipped = load_coco_predictions(predictions_path, coco_gt, image_info_by_id, args, class_names)
    else:
        detections, skipped = load_txt_predictions(predictions_path, images, coco_gt, image_info_by_id, args, class_names)
    convert_elapsed = time.perf_counter() - convert_start
    if not detections:
        raise RuntimeError(f"No valid predictions loaded from {predictions_path}. skipped={skipped}")

    run_name = args.name or f"{args.mode}_{args.split}_{predictions_path.stem}"
    out_dir = Path(args.out).expanduser().resolve() if args.out else ROOT / "runs" / "analysis" / "coco_eval" / sanitize_name(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_json = out_dir / "instances_gt.json"
    pred_json = out_dir / "predictions.normalized.json"
    write_start = time.perf_counter()
    gt_json.write_text(json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8")
    pred_json.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
    write_elapsed = time.perf_counter() - write_start

    eval_start = time.perf_counter()
    evaluator, eval_text = run_coco_eval(coco_gt, detections, args.coco_max_det)
    eval_elapsed = time.perf_counter() - eval_start
    (out_dir / "coco_eval.txt").write_text(eval_text, encoding="utf-8")
    rows = class_area_rows(evaluator, class_names, coco_gt)
    write_csv(out_dir / "metrics_by_class_area.csv", rows)
    timing = load_timing(args.timing_json)
    summary: dict[str, object] = {
        "mode": args.mode,
        "dataset_root": str(dataset_root),
        "image_subdir": image_subdir,
        "split": args.split,
        "num_images": len(images),
        "num_gt": len(coco_gt["annotations"]),
        "num_predictions": len(detections),
        "predictions_input": str(predictions_path),
        "pred_format": args.pred_format,
        "prediction_space": args.prediction_space,
        "eval_space": args.eval_space,
        "coco_max_dets": [1, 10, int(args.coco_max_det)],
        "skipped_predictions": skipped,
        "coco_convert_wall_sec": convert_elapsed,
        "coco_convert_ms_per_img": convert_elapsed * 1000.0 / max(len(images), 1),
        "json_write_wall_sec": write_elapsed,
        "json_write_ms_per_img": write_elapsed * 1000.0 / max(len(images), 1),
        "coco_eval_wall_sec": eval_elapsed,
        "coco_eval_ms_per_img": eval_elapsed * 1000.0 / max(len(images), 1),
        "native_timing": timing,
        "mAP50_95": float(evaluator.stats[0]),
        "mAP50": float(evaluator.stats[1]),
        "mAP75": float(evaluator.stats[2]),
        "AP_S": float(evaluator.stats[3]),
        "AP_M": float(evaluator.stats[4]),
        "AP_L": float(evaluator.stats[5]),
        "AR_1": float(evaluator.stats[6]),
        "AR_10": float(evaluator.stats[7]),
        f"AR_{int(args.coco_max_det)}": float(evaluator.stats[8]),
        "gt_json": str(gt_json),
        "pred_json": str(pred_json),
    }
    if "infer_wall_sec" in timing:
        infer_sec = float(timing["infer_wall_sec"])
        summary["fps_infer_wall"] = len(images) / max(infer_sec, 1e-9)
        summary["infer_ms_per_img_wall"] = infer_sec * 1000.0 / max(len(images), 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
