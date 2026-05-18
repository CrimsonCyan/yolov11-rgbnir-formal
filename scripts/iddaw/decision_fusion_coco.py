from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.iddaw import (  # noqa: E402
    TRAFFIC_DETECTABLE640_DATASET_NAME,
    category_names_for_mode,
)
from scripts.iddaw.export_yolo_coco_predictions import (  # noqa: E402
    build_coco_gt,
    class_area_rows,
    default_bbox_dataset_root,
    export_predictions,
    list_images,
    run_coco_eval,
    sanitize_name,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse RGB and NIR COCO prediction JSONs with class-wise NMS and evaluate using the same COCOeval "
            "path as export_yolo_coco_predictions.py. If prediction JSONs are not provided, the script exports "
            "them from the supplied YOLO checkpoints first."
        )
    )
    parser.add_argument("--rgb-mode", default="rgb_p2p5_yolo11s_8cls_personmerge_traffic")
    parser.add_argument("--nir-mode", default="nir_p2p5_yolo11s_8cls_personmerge_traffic")
    parser.add_argument("--rgb-weights", default="", help="RGB checkpoint. Required when --rgb-pred-json is empty.")
    parser.add_argument("--nir-weights", default="", help="NIR checkpoint. Required when --nir-pred-json is empty.")
    parser.add_argument("--rgb-pred-json", default="", help="Existing RGB COCO prediction JSON.")
    parser.add_argument("--nir-pred-json", default="", help="Existing NIR COCO prediction JSON.")
    parser.add_argument("--dataset-root", default="", help="8cls detectable640 dataset root.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", default="", help="Output directory.")
    parser.add_argument("--name", default="decision_fusion_rgb_nir_p2p5_8cls")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7, help="Branch YOLO NMS IoU if exporting predictions.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--coco-max-det", type=int, default=100)
    parser.add_argument("--fusion-iou", type=float, default=0.7, help="Class-wise NMS IoU after merging RGB/NIR predictions.")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--half", action="store_true", help="Use FP16 for YOLO branch inference.")
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def load_predictions(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Prediction JSON must be a list: {path}")
    required = {"image_id", "category_id", "bbox", "score"}
    for index, item in enumerate(data[:10]):
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Missing keys {sorted(missing)} in {path} at prediction index {index}")
    return data


def xywh_to_xyxy_array(boxes: np.ndarray) -> np.ndarray:
    out = boxes.astype(np.float32, copy=True)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes_xyxy.size == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-12)
        order = rest[iou <= iou_threshold]
    return keep


def fuse_predictions(
    rgb_predictions: list[dict[str, object]],
    nir_predictions: list[dict[str, object]],
    fusion_iou: float,
    max_det: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for source, prediction in [("rgb", item) for item in rgb_predictions] + [("nir", item) for item in nir_predictions]:
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        item = dict(prediction)
        item["source"] = source
        grouped.setdefault((image_id, category_id), []).append(item)

    fused_by_image: dict[int, list[dict[str, object]]] = {}
    for (image_id, _category_id), items in grouped.items():
        boxes = np.asarray([item["bbox"] for item in items], dtype=np.float32)
        scores = np.asarray([float(item["score"]) for item in items], dtype=np.float32)
        keep = nms_numpy(xywh_to_xyxy_array(boxes), scores, fusion_iou)
        fused_by_image.setdefault(image_id, []).extend(items[index] for index in keep)

    fused: list[dict[str, object]] = []
    for image_id, items in fused_by_image.items():
        items = sorted(items, key=lambda item: float(item["score"]), reverse=True)[:max_det]
        for item in items:
            fused.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(item["category_id"]),
                    "bbox": [float(value) for value in item["bbox"]],
                    "score": float(item["score"]),
                }
            )
    return sorted(fused, key=lambda item: (int(item["image_id"]), int(item["category_id"]), -float(item["score"])))


def export_branch_predictions(
    mode: str,
    weights: str,
    images: list[Path],
    image_id_by_stem: dict[str, int],
    class_names: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, int]]:
    if not weights:
        raise ValueError(f"--{mode}-weights is required when no existing prediction JSON is provided")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_args = argparse.Namespace(
        mode=mode,
        weights=weights,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        half=args.half,
        save_txt=False,
        eval_space="letterbox",
        stream=False,
    )
    predictions, timing, skipped = export_predictions(export_args, images, image_id_by_stem, class_names, out_dir)
    return predictions, timing, skipped


def main() -> None:
    args = parse_args()
    class_names = category_names_for_mode(args.rgb_mode)
    nir_class_names = category_names_for_mode(args.nir_mode)
    if class_names != nir_class_names:
        raise ValueError(f"RGB/NIR class names differ: {class_names} vs {nir_class_names}")
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else default_bbox_dataset_root(args.rgb_mode, class_names)
    if len(class_names) == 8 and dataset_root.name != TRAFFIC_DETECTABLE640_DATASET_NAME:
        raise ValueError(
            "Decision Fusion for 8cls must use detectable640 bbox dataset.\n"
            f"Expected: {TRAFFIC_DETECTABLE640_DATASET_NAME}\nGot: {dataset_root}"
        )
    visible_dir = dataset_root / "visible" / args.split
    nir_dir = dataset_root / "nir" / args.split
    visible_images = list_images(visible_dir, args.max_images)
    nir_images = [nir_dir / image.name for image in visible_images]
    missing_nir = [path for path in nir_images if not path.exists()]
    if missing_nir:
        raise FileNotFoundError(f"Missing paired NIR image, first missing path: {missing_nir[0]}")

    coco_gt, image_id_by_stem = build_coco_gt(visible_images, class_names, args.imgsz, "letterbox")
    out_dir = Path(args.out).expanduser().resolve() if args.out else ROOT / "runs" / "analysis" / "coco_eval" / sanitize_name(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rgb_pred_json:
        rgb_predictions = load_predictions(Path(args.rgb_pred_json).expanduser().resolve())
        rgb_timing = {}
        rgb_skipped = {}
    else:
        rgb_predictions, rgb_timing, rgb_skipped = export_branch_predictions(
            args.rgb_mode, args.rgb_weights, visible_images, image_id_by_stem, class_names, out_dir / "rgb_export", args
        )
    if args.nir_pred_json:
        nir_predictions = load_predictions(Path(args.nir_pred_json).expanduser().resolve())
        nir_timing = {}
        nir_skipped = {}
    else:
        nir_predictions, nir_timing, nir_skipped = export_branch_predictions(
            args.nir_mode, args.nir_weights, nir_images, image_id_by_stem, class_names, out_dir / "nir_export", args
        )

    start = time.perf_counter()
    fused_predictions = fuse_predictions(rgb_predictions, nir_predictions, args.fusion_iou, args.max_det)
    fusion_elapsed = time.perf_counter() - start
    if not fused_predictions:
        raise RuntimeError("Decision Fusion produced no predictions.")

    gt_json = out_dir / "instances_gt.json"
    rgb_json = out_dir / "rgb_predictions.json"
    nir_json = out_dir / "nir_predictions.json"
    pred_json = out_dir / "predictions.json"
    gt_json.write_text(json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8")
    rgb_json.write_text(json.dumps(rgb_predictions, ensure_ascii=False), encoding="utf-8")
    nir_json.write_text(json.dumps(nir_predictions, ensure_ascii=False), encoding="utf-8")
    pred_json.write_text(json.dumps(fused_predictions, ensure_ascii=False), encoding="utf-8")
    rgb_wall_sec = float(rgb_timing.get("infer_wall_sec", 0.0))
    nir_wall_sec = float(nir_timing.get("infer_wall_sec", 0.0))
    rgb_native_ms = float(rgb_timing.get("ultralytics_total_ms_img", 0.0))
    nir_native_ms = float(nir_timing.get("ultralytics_total_ms_img", 0.0))
    native_ms = rgb_native_ms + nir_native_ms
    native_fps = 1000.0 / native_ms if native_ms > 0 else 0.0

    summary: dict[str, object] = {
        "method": "decision_fusion_classwise_nms",
        "rgb_mode": args.rgb_mode,
        "nir_mode": args.nir_mode,
        "rgb_weights": str(Path(args.rgb_weights).expanduser().resolve()) if args.rgb_weights else "",
        "nir_weights": str(Path(args.nir_weights).expanduser().resolve()) if args.nir_weights else "",
        "dataset_root": str(dataset_root),
        "split": args.split,
        "num_images": len(visible_images),
        "num_gt": len(coco_gt["annotations"]),
        "num_rgb_predictions": len(rgb_predictions),
        "num_nir_predictions": len(nir_predictions),
        "num_predictions": len(fused_predictions),
        "rgb_skipped_predictions": rgb_skipped,
        "nir_skipped_predictions": nir_skipped,
        "imgsz": args.imgsz,
        "half": bool(args.half),
        "eval_space": "letterbox",
        "branch_conf": args.conf,
        "branch_nms_iou": args.iou,
        "fusion_iou": args.fusion_iou,
        "max_det": args.max_det,
        "coco_max_dets": [1, 10, int(args.coco_max_det)],
        "speed_note": (
            "native_ms_per_img sums RGB and NIR Ultralytics preprocess+inference+postprocess speeds. "
            "Fusion NMS wall time is reported separately and is not included in Native FPS."
        ),
        "rgb_ultralytics_total_ms_img": rgb_native_ms,
        "nir_ultralytics_total_ms_img": nir_native_ms,
        "native_ms_per_img": native_ms,
        "native_fps": native_fps,
        "rgb_infer_wall_sec": rgb_wall_sec,
        "nir_infer_wall_sec": nir_wall_sec,
        "fusion_wall_sec": fusion_elapsed,
        "fusion_ms_per_img_wall": fusion_elapsed * 1000.0 / max(len(visible_images), 1),
        "infer_wall_sec_total": rgb_wall_sec + nir_wall_sec + fusion_elapsed,
        "infer_ms_per_img_wall": (rgb_wall_sec + nir_wall_sec + fusion_elapsed) * 1000.0 / max(len(visible_images), 1),
        "fps_wall": len(visible_images) / max(rgb_wall_sec + nir_wall_sec + fusion_elapsed, 1e-9),
        "gt_json": str(gt_json),
        "pred_json": str(pred_json),
        "rgb_pred_json": str(rgb_json),
        "nir_pred_json": str(nir_json),
    }

    if not args.no_eval:
        evaluator, eval_text = run_coco_eval(coco_gt, fused_predictions, args.coco_max_det)
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
