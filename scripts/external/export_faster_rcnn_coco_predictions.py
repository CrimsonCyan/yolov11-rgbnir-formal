from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.iddaw.export_yolo_coco_predictions import (  # noqa: E402
    build_coco_gt,
    class_area_rows,
    run_coco_eval,
    sanitize_name,
    xyxy_to_xywh,
)

CLASS_NAMES = [
    "person",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "autorickshaw",
    "traffic light",
    "traffic sign",
]
DETECTABLE640_DATASET_NAME = "iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Faster R-CNN predictions to COCO JSON and report FPS with conversion/write/eval time separated."
        )
    )
    parser.add_argument("--weights", required=True, help="Checkpoint produced by scripts/external/train_faster_rcnn.py.")
    parser.add_argument("--modality", choices=["rgb", "nir"], required=True)
    parser.add_argument(
        "--dataset-root",
        default="/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--coco-max-det", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--half", action="store_true", help="Use CUDA AMP during inference timing.")
    parser.add_argument("--out", default="")
    parser.add_argument("--name", default="")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if raw.isdigit():
        return torch.device(f"cuda:{raw}")
    return torch.device(raw)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {}


@torch.inference_mode()
def warmup(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_batches: int, half: bool) -> None:
    if num_batches <= 0:
        return
    model.eval()
    for index, (images, _, _) in enumerate(loader):
        if index >= num_batches:
            break
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.cuda.amp.autocast(enabled=half and device.type == "cuda"):
            _ = model(images)
    cuda_sync(device)


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, half: bool
) -> tuple[list[dict], dict[str, float]]:
    model.eval()
    records: list[dict] = []
    model_forward_sec = 0.0
    seen = 0
    cuda_sync(device)
    loop_start = time.perf_counter()
    for images, _, metas in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        cuda_sync(device)
        forward_start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=half and device.type == "cuda"):
            outputs = model(images)
        cuda_sync(device)
        model_forward_sec += time.perf_counter() - forward_start
        seen += len(images)
        for output, meta in zip(outputs, metas):
            records.append(
                {
                    "sample_id": meta["sample_id"],
                    "boxes": output["boxes"].detach().cpu().float(),
                    "scores": output["scores"].detach().cpu().float(),
                    "labels": output["labels"].detach().cpu().long(),
                }
            )
    cuda_sync(device)
    loop_sec = time.perf_counter() - loop_start
    return records, {"infer_wall_sec": loop_sec, "model_forward_wall_sec": model_forward_sec, "num_images": float(seen)}


def convert_to_coco(records: list[dict], image_id_by_stem: dict[str, int], imgsz: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    detections: list[dict[str, object]] = []
    skipped = {"image_id_missing": 0, "class_out_of_range": 0, "invalid_bbox": 0}
    for record in records:
        image_id = image_id_by_stem.get(str(record["sample_id"]))
        if image_id is None:
            skipped["image_id_missing"] += 1
            continue
        boxes = record["boxes"]
        scores = record["scores"]
        labels = record["labels"]
        for box, score, label in zip(boxes, scores, labels):
            category_id = int(label)
            if not 1 <= category_id <= len(CLASS_NAMES):
                skipped["class_out_of_range"] += 1
                continue
            bbox = xyxy_to_xywh([float(value) for value in box.tolist()], imgsz, imgsz)
            if bbox[2] <= 0 or bbox[3] <= 0:
                skipped["invalid_bbox"] += 1
                continue
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": category_id,
                    "bbox": [float(value) for value in bbox],
                    "score": float(score),
                }
            )
    return detections, skipped


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    from scripts.external.train_faster_rcnn import IDDAWYoloDetectionDataset, build_model, collate_fn

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if dataset_root.name != DETECTABLE640_DATASET_NAME:
        raise ValueError(
            "Faster R-CNN export must use the filtered detectable640 bbox dataset.\n"
            f"Expected: {DETECTABLE640_DATASET_NAME}\nGot: {dataset_root}"
        )
    device = resolve_device(args.device)
    dataset = IDDAWYoloDetectionDataset(dataset_root, args.split, args.modality, args.imgsz, args.max_images)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=False,
    )
    model = build_model(len(CLASS_NAMES), pretrained=False, imgsz=args.imgsz, conf=args.conf, nms_iou=args.iou).to(device)
    checkpoint = load_checkpoint(Path(args.weights).expanduser().resolve(), model, device)
    warmup(model, loader, device, args.warmup_batches, args.half)

    records, timing = collect_predictions(model, loader, device, args.half)
    coco_gt, image_id_by_stem = build_coco_gt(dataset.image_paths, CLASS_NAMES, args.imgsz, "letterbox")
    convert_start = time.perf_counter()
    detections, skipped = convert_to_coco(records, image_id_by_stem, args.imgsz)
    convert_elapsed = time.perf_counter() - convert_start
    if not detections:
        raise RuntimeError(f"No valid Faster R-CNN detections exported. skipped={skipped}")

    run_name = args.name or f"faster_rcnn_{args.modality}_{args.split}_{Path(args.weights).stem}"
    out_dir = Path(args.out).expanduser().resolve() if args.out else ROOT / "runs" / "analysis" / "coco_eval" / sanitize_name(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_json = out_dir / "instances_gt.json"
    pred_json = out_dir / "predictions.json"
    write_start = time.perf_counter()
    gt_json.write_text(json.dumps(coco_gt, ensure_ascii=False), encoding="utf-8")
    pred_json.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
    write_elapsed = time.perf_counter() - write_start

    eval_start = time.perf_counter()
    evaluator, eval_text = run_coco_eval(coco_gt, detections, args.coco_max_det)
    eval_elapsed = time.perf_counter() - eval_start
    (out_dir / "coco_eval.txt").write_text(eval_text, encoding="utf-8")
    rows = class_area_rows(evaluator, CLASS_NAMES, coco_gt)
    write_csv(out_dir / "metrics_by_class_area.csv", rows)
    num_images = max(int(timing["num_images"]), 1)
    summary: dict[str, object] = {
        "backend": "torchvision_faster_rcnn_resnet50_fpn",
        "weights": str(Path(args.weights).expanduser().resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "modality": args.modality,
        "dataset_root": str(dataset_root),
        "split": args.split,
        "num_images": num_images,
        "num_gt": len(coco_gt["annotations"]),
        "num_predictions": len(detections),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(device),
        "half": bool(args.half),
        "conf": args.conf,
        "nms_iou": args.iou,
        "coco_max_dets": [1, 10, int(args.coco_max_det)],
        "skipped_predictions": skipped,
        "timing_note": "infer_wall_sec includes dataloader/preprocess/model/postprocess loop; COCO conversion, JSON writing and COCOeval are separate.",
        "infer_wall_sec": timing["infer_wall_sec"],
        "infer_ms_per_img_wall": timing["infer_wall_sec"] * 1000.0 / num_images,
        "fps_infer_wall": num_images / max(float(timing["infer_wall_sec"]), 1e-9),
        "model_forward_wall_sec": timing["model_forward_wall_sec"],
        "model_forward_ms_per_img": timing["model_forward_wall_sec"] * 1000.0 / num_images,
        "fps_model_forward": num_images / max(float(timing["model_forward_wall_sec"]), 1e-9),
        "coco_convert_wall_sec": convert_elapsed,
        "coco_convert_ms_per_img": convert_elapsed * 1000.0 / num_images,
        "json_write_wall_sec": write_elapsed,
        "json_write_ms_per_img": write_elapsed * 1000.0 / num_images,
        "coco_eval_wall_sec": eval_elapsed,
        "coco_eval_ms_per_img": eval_elapsed * 1000.0 / num_images,
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
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
