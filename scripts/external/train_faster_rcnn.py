from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.box_ops import AREA_REFERENCE_SIZE, area_bucket, box_iou
from ultralytics.utils.metrics import ap_per_class


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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class TrainConfig:
    modality: str
    dataset_root: str
    epochs: int
    imgsz: int
    batch_per_gpu: int
    lr: float
    momentum: float
    weight_decay: float
    lr_milestones: tuple[int, ...]
    lr_gamma: float
    warmup_iters: int
    warmup_factor: float
    pretrained: bool
    amp: bool
    conf: float
    pr_conf: float
    iou: float
    area_imgsz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on IDD-AW YOLO-format RGB/NIR data.")
    parser.add_argument("--modality", choices=["rgb", "nir"], required=True)
    parser.add_argument(
        "--dataset-root",
        default="/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-per-gpu", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project", default="runs/external/faster_rcnn")
    parser.add_argument("--name", default="")
    parser.add_argument("--pretrained", default="true", choices=["true", "false", "1", "0"])
    parser.add_argument("--amp", default="true", choices=["true", "false", "1", "0"])
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--lr-milestones", default="60,80")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--warmup-factor", type=float, default=0.001)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold used for AP collection.")
    parser.add_argument("--pr-conf", type=float, default=0.25, help="Confidence threshold used for P/R summaries.")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--area-imgsz", type=float, default=AREA_REFERENCE_SIZE)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-gflops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-images", type=int, default=0, help="Debug cap; 0 means full train split.")
    parser.add_argument("--max-val-images", type=int, default=0, help="Debug cap; 0 means full val split.")
    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return distributed, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def split_milestones(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def modality_dir_name(modality: str) -> str:
    return "visible" if modality == "rgb" else "nir"


def letterbox_image(image: Image.Image, imgsz: int) -> tuple[Image.Image, float, float, float]:
    width, height = image.size
    gain = min(float(imgsz) / max(float(width), 1.0), float(imgsz) / max(float(height), 1.0))
    new_width = max(int(round(width * gain)), 1)
    new_height = max(int(round(height * gain)), 1)
    resized = image.resize((new_width, new_height), Image.BILINEAR)
    pad_x = (imgsz - new_width) / 2.0
    pad_y = (imgsz - new_height) / 2.0
    canvas = Image.new("RGB", (imgsz, imgsz), color=(114, 114, 114))
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    return canvas, gain, float(int(round(pad_x))), float(int(round(pad_y)))


def read_yolo_labels(label_path: Path, width: int, height: int, gain: float, pad_x: float, pad_y: float) -> tuple[torch.Tensor, torch.Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    if label_path.exists():
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = [float(item) for item in parts[1:]]
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            x1 = x1 * gain + pad_x
            y1 = y1 * gain + pad_y
            x2 = x2 * gain + pad_x
            y2 = y2 * gain + pad_y
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cls + 1)  # torchvision reserves 0 for background.
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


class IDDAWYoloDetectionDataset(Dataset):
    def __init__(self, root: Path, split: str, modality: str, imgsz: int, max_images: int = 0):
        self.root = root
        self.split = split
        self.modality = modality
        self.imgsz = imgsz
        image_dir = root / modality_dir_name(modality) / split
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)
        self.image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if max_images > 0:
            self.image_paths = self.image_paths[:max_images]
        if not self.image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        if self.modality == "rgb":
            image = Image.open(image_path).convert("RGB")
        else:
            gray = Image.open(image_path).convert("L")
            image = Image.merge("RGB", (gray, gray, gray))
        width, height = image.size
        image, gain, pad_x, pad_y = letterbox_image(image, self.imgsz)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        boxes, labels = read_yolo_labels(image_path.with_suffix(".txt"), width, height, gain, pad_x, pad_y)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.imgsz)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.imgsz)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) if len(boxes) else torch.zeros((0,), dtype=torch.bool)
        boxes = boxes[keep]
        labels = labels[keep]
        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        meta = {
            "sample_id": image_path.stem,
            "shape": (self.imgsz, self.imgsz),
            "path": str(image_path),
        }
        return image_tensor, target, meta


def collate_fn(batch):
    images, targets, metas = zip(*batch)
    return list(images), list(targets), list(metas)


def move_targets(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def build_model(num_classes: int, pretrained: bool, imgsz: int, conf: float, nms_iou: float) -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=imgsz, max_size=imgsz)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.transform.min_size = (imgsz,)
    model.transform.max_size = imgsz
    model.roi_heads.score_thresh = conf
    model.roi_heads.nms_thresh = nms_iou
    return model


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def lr_for_step(config: TrainConfig, epoch: int, global_step: int) -> float:
    decay_power = sum(1 for milestone in config.lr_milestones if epoch >= milestone)
    lr = config.lr * (config.lr_gamma ** decay_power)
    if global_step < config.warmup_iters:
        alpha = float(global_step) / max(float(config.warmup_iters), 1.0)
        warmup = config.warmup_factor * (1.0 - alpha) + alpha
        lr *= warmup
    return lr


def boxes_to_area_buckets(boxes: torch.Tensor, image_shape: tuple[int, int], area_imgsz: float) -> list[str]:
    return [
        area_bucket(
            float((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)),
            image_shape=image_shape,
            reference_size=area_imgsz,
        )
        for box in boxes
    ]


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
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.zeros((len(pred_boxes),), dtype=torch.bool)
    ious = box_iou(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels
    return (ious * same_class).max(0).values >= iou_threshold


def prediction_area_bucket(box: torch.Tensor, image_shape: tuple[int, int], area_imgsz: float) -> str:
    area = float((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0))
    return area_bucket(area, image_shape=image_shape, reference_size=area_imgsz)


def filter_prediction_for_area(
    prediction: dict[str, torch.Tensor],
    valid_boxes: torch.Tensor,
    valid_labels: torch.Tensor,
    ignored_boxes: torch.Tensor,
    ignored_labels: torch.Tensor,
    area_filter: str | None,
    image_shape: tuple[int, int],
    area_imgsz: float,
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
    for index, threshold in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), index] = True
    return torch.tensor(correct, dtype=torch.bool)


def collect_stats(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, object]],
    area_filter: str | None,
    iouv: torch.Tensor,
    conf: float | None,
    area_imgsz: float,
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
            target["shape"],
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
        correct = torch.zeros((len(pred_boxes), len(iouv)), dtype=torch.bool)
        if len(pred_boxes) and len(valid_boxes):
            correct = ultralytics_match_predictions(pred_labels, valid_labels, box_iou(valid_boxes, pred_boxes), iouv)
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


def metrics_from_stats(stats: dict[str, np.ndarray]) -> dict[int, dict[str, float]]:
    target_cls = stats["target_cls"].astype(int)
    gt_counts = np.bincount(target_cls, minlength=len(CLASS_NAMES)) if len(target_cls) else np.zeros(len(CLASS_NAMES), dtype=int)
    rows = {
        cls: {"gt_count": int(gt_counts[cls]), "precision": 0.0, "recall": 0.0, "AP50": 0.0, "mAP50_95": 0.0}
        for cls in range(len(CLASS_NAMES))
    }
    if len(target_cls) and len(stats["tp"]):
        _, _, precision, recall, _, ap, unique_classes, *_ = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            names=dict(enumerate(CLASS_NAMES)),
        )
        for index, cls in enumerate(unique_classes.astype(int)):
            rows[cls].update(
                {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "AP50": float(ap[index, 0]),
                    "mAP50_95": float(ap[index].mean()),
                }
            )
    return rows


def pr_at_conf_from_stats(stats: dict[str, np.ndarray]) -> dict[int, tuple[float, float]]:
    target_cls = stats["target_cls"].astype(int)
    pred_cls = stats["pred_cls"].astype(int)
    tp = stats["tp"][:, 0].astype(bool) if len(stats["tp"]) else np.zeros((0,), dtype=bool)
    out: dict[int, tuple[float, float]] = {}
    for cls in range(len(CLASS_NAMES)):
        pred_mask = pred_cls == cls
        gt_count = int((target_cls == cls).sum())
        tp_count = int(tp[pred_mask].sum())
        fp_count = int(pred_mask.sum() - tp_count)
        out[cls] = (tp_count / max(tp_count + fp_count, 1), tp_count / max(gt_count, 1))
    return out


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def append_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    areas = []
    for row in rows:
        if row["area"] not in areas:
            areas.append(row["area"])
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


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: TrainConfig) -> tuple[dict[str, float], list[dict[str, object]], float]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_for_metrics: list[dict[str, object]] = []
    infer_ms = 0.0
    seen = 0
    for images, targets, metas in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_ms += (time.perf_counter() - start) * 1000.0
        seen += len(images)
        for output, target, meta in zip(outputs, targets, metas):
            pred_labels = output["labels"].detach().cpu().long() - 1
            keep = (pred_labels >= 0) & (pred_labels < len(CLASS_NAMES))
            pred = {
                "boxes": output["boxes"].detach().cpu().float()[keep],
                "scores": output["scores"].detach().cpu().float()[keep],
                "labels": pred_labels[keep],
            }
            gt_labels = target["labels"].detach().cpu().long() - 1
            metric_target = {
                "sample_id": meta["sample_id"],
                "boxes_xyxy": target["boxes"].detach().cpu().float(),
                "labels": gt_labels,
                "area_buckets": boxes_to_area_buckets(target["boxes"].detach().cpu().float(), meta["shape"], config.area_imgsz),
                "shape": meta["shape"],
            }
            predictions.append(pred)
            targets_for_metrics.append(metric_target)
    iouv = torch.linspace(0.5, 0.95, 10)
    rows: list[dict[str, object]] = []
    summary: dict[str, float] = {}
    for area in [None, "small", "medium", "large"]:
        stats = collect_stats(predictions, targets_for_metrics, area, iouv, config.conf, config.area_imgsz)
        pr_stats = collect_stats(predictions, targets_for_metrics, area, torch.tensor([0.5]), config.pr_conf, config.area_imgsz)
        metrics = metrics_from_stats(stats)
        pr_metrics = pr_at_conf_from_stats(pr_stats)
        area_name = area or "all"
        for cls, class_name in enumerate(CLASS_NAMES):
            precision_at_conf, recall_at_conf = pr_metrics[cls]
            row = {
                "class": class_name,
                "area": area_name,
                "gt_count": metrics[cls]["gt_count"],
                "precision": metrics[cls]["precision"],
                "recall": metrics[cls]["recall"],
                "AP50": metrics[cls]["AP50"],
                "mAP50_95": metrics[cls]["mAP50_95"],
                "precision_at_conf": precision_at_conf,
                "recall_at_conf": recall_at_conf,
                "conf": config.pr_conf,
            }
            rows.append(row)
    rows = append_mean_rows(rows)
    mean_rows = {row["area"]: row for row in rows if row["class"] == "mean"}
    all_mean = mean_rows["all"]
    summary.update(
        {
            "precision": float(all_mean["precision"]),
            "recall": float(all_mean["recall"]),
            "mAP50": float(all_mean["AP50"]),
            "mAP50_95": float(all_mean["mAP50_95"]),
            "AP_S": float(mean_rows["small"]["mAP50_95"]),
            "AP_M": float(mean_rows["medium"]["mAP50_95"]),
            "AP_L": float(mean_rows["large"]["mAP50_95"]),
            "infer_ms_per_img": infer_ms / max(seen, 1),
            "fps": 1000.0 / max(infer_ms / max(seen, 1), 1e-9),
            "val_images": float(seen),
            "val_gt": float(all_mean["gt_count"]),
        }
    )
    return summary, rows, summary["infer_ms_per_img"]


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_results_csv(path: Path, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float], lr: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"epoch": epoch, "lr": lr, **{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float, config: TrainConfig, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": asdict(config),
            "metrics": metrics,
            "class_names": CLASS_NAMES,
        },
        path,
    )


def maybe_load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, float]:
    if not path:
        return 0, -1.0
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("epoch", -1)) + 1, float(payload.get("best_metric", -1.0))


class DetectionProfileWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model([x[0]])


@torch.inference_mode()
def profile_gflops(model: nn.Module, imgsz: int, device: torch.device) -> float | None:
    try:
        from thop import profile

        was_training = model.training
        model.eval()
        dummy = torch.rand(1, 3, imgsz, imgsz, device=device)
        macs, _ = profile(DetectionProfileWrapper(model), inputs=(dummy,), verbose=False)
        if was_training:
            model.train()
        return float(macs * 2.0 / 1e9)
    except Exception as exc:
        print(f"WARNING: GFLOPs profiling failed: {exc}", flush=True)
        return None


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def configure_wandb(config: TrainConfig, run_dir: Path):
    if os.getenv("WANDB_ENABLED", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WANDB_ENABLED=1 but wandb is not installed.") from exc
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "iddaw-rgbnir-formal"),
        group=os.getenv("WANDB_GROUP", "external_faster_rcnn"),
        name=run_dir.name,
        tags=[tag for tag in os.getenv("WANDB_TAGS", f"faster-rcnn,{config.modality},8cls").split(",") if tag],
        config=asdict(config),
        dir=str(run_dir),
    )


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed()
    set_seed(args.seed, rank)
    cuda_available = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device(f"cuda:{local_rank}" if cuda_available else "cpu")
    project = Path(args.project)
    default_name = f"faster_rcnn_{args.modality}_8cls_detectable640"
    run_dir = project / (args.name or default_name)
    if is_rank0(rank):
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        modality=args.modality,
        dataset_root=str(Path(args.dataset_root).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_per_gpu=args.batch_per_gpu,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_milestones=split_milestones(args.lr_milestones),
        lr_gamma=args.lr_gamma,
        warmup_iters=args.warmup_iters,
        warmup_factor=args.warmup_factor,
        pretrained=str_to_bool(args.pretrained),
        amp=str_to_bool(args.amp),
        conf=args.conf,
        pr_conf=args.pr_conf,
        iou=args.iou,
        area_imgsz=args.area_imgsz,
    )

    dataset_root = Path(config.dataset_root)
    train_dataset = IDDAWYoloDetectionDataset(dataset_root, "train", config.modality, config.imgsz, args.max_train_images)
    val_dataset = IDDAWYoloDetectionDataset(dataset_root, "val", config.modality, config.imgsz, args.max_val_images)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_per_gpu,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=cuda_available,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_per_gpu,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=cuda_available,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if distributed and config.pretrained:
        if is_rank0(rank):
            FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True)
        dist.barrier()

    model = build_model(len(CLASS_NAMES), config.pretrained, config.imgsz, config.conf, config.iou).to(device)
    model_without_ddp = model
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    start_epoch, best_metric = maybe_load_checkpoint(args.resume, model_without_ddp, optimizer, device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model_without_ddp = model.module
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and cuda_available)

    params = count_params(model_without_ddp)
    gflops = profile_gflops(model_without_ddp, config.imgsz, device) if is_rank0(rank) and args.profile_gflops else None
    if is_rank0(rank):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "model_summary.json").write_text(
            json.dumps({"params": params, "params_m": params / 1e6, "gflops": gflops}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"run_dir={run_dir}", flush=True)
        print(f"params={params / 1e6:.2f}M gflops={gflops if gflops is not None else 'NA'}", flush=True)
    wandb_run = configure_wandb(config, run_dir) if is_rank0(rank) else None
    if distributed:
        dist.barrier()

    global_step = start_epoch * max(len(train_loader), 1)
    for epoch in range(start_epoch, config.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sums: dict[str, float] = {}
        sample_count = 0
        epoch_start = time.perf_counter()
        for images, targets, _ in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = move_targets(targets, device)
            lr = lr_for_step(config, epoch, global_step)
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.amp and cuda_available):
                loss_dict = model(images, targets)
                loss = sum(value for value in loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Faster R-CNN loss at epoch={epoch + 1}, step={global_step}: {loss}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(images)
            sample_count += batch_size
            loss_sums["loss"] = loss_sums.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
            for key, value in loss_dict.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            global_step += 1
        train_metrics = {key: value / max(sample_count, 1) for key, value in loss_sums.items()}
        train_metrics["epoch_time_sec"] = time.perf_counter() - epoch_start
        lr = lr_for_step(config, epoch, global_step)

        if distributed:
            dist.barrier()
        if is_rank0(rank):
            val_metrics, metric_rows, _ = evaluate(model_without_ddp, val_loader, device, config)
            val_metrics["params_m"] = params / 1e6
            val_metrics["gflops"] = float(gflops) if gflops is not None else math.nan
            append_results_csv(run_dir / "results.csv", epoch + 1, train_metrics, val_metrics, lr)
            write_metrics_csv(run_dir / "metrics_by_class_area.csv", metric_rows)
            summary = {
                "epoch": epoch + 1,
                "best_metric": max(best_metric, val_metrics["mAP50_95"]),
                "train": train_metrics,
                "val": val_metrics,
                "params": params,
                "params_m": params / 1e6,
                "gflops": gflops,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            save_checkpoint(run_dir / "weights" / "last.pt", model_without_ddp, optimizer, epoch, best_metric, config, val_metrics)
            if val_metrics["mAP50_95"] >= best_metric:
                best_metric = val_metrics["mAP50_95"]
                save_checkpoint(run_dir / "weights" / "best.pt", model_without_ddp, optimizer, epoch, best_metric, config, val_metrics)
            print(
                f"epoch={epoch + 1}/{config.epochs} loss={train_metrics['loss']:.4f} "
                f"mAP50={val_metrics['mAP50']:.4f} mAP50-95={val_metrics['mAP50_95']:.4f} "
                f"AP_S/M/L={val_metrics['AP_S']:.4f}/{val_metrics['AP_M']:.4f}/{val_metrics['AP_L']:.4f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{key}": value for key, value in train_metrics.items()},
                        **{f"metrics/{key}": value for key, value in val_metrics.items()},
                        "epoch": epoch + 1,
                        "lr": lr,
                    },
                    step=epoch + 1,
                )
        if distributed:
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
