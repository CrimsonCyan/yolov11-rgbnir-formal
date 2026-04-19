from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO

from .box_ops import xywh_to_xyxy
from .iddaw_fog import CATEGORY_NAMES, DEFAULT_PAIRS, latest_weights_for, mode_specific_kwargs, resolve_dataset_root
from .metrics import build_eval_targets, evaluate_predictions
from .nms import batched_nms


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_files(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _collect_split_entries(split: str):
    dataset_root = resolve_dataset_root()
    visible_dir = dataset_root / "visible" / split
    nir_dir = dataset_root / "nir" / split
    entries = []
    for visible_path in _image_files(visible_dir):
        nir_path = nir_dir / visible_path.name
        label_path = visible_path.with_suffix(".txt")
        if not nir_path.exists():
            raise FileNotFoundError(f"Missing paired NIR image for {visible_path.name}: {nir_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {visible_path.name}: {label_path}")
        entries.append(
            {
                "sample_id": visible_path.stem,
                "visible_path": visible_path,
                "nir_path": nir_path,
                "label_path": label_path,
            }
        )
    return entries


def _load_targets(split: str):
    targets = []
    for entry in _collect_split_entries(split):
        image = cv2.imread(str(entry["visible_path"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {entry['visible_path']}")
        height, width = image.shape[:2]
        rows = []
        text = entry["label_path"].read_text(encoding="utf-8").strip()
        if text:
            rows = [line.split() for line in text.splitlines() if line.strip()]
        if rows:
            labels = torch.tensor([int(row[0]) for row in rows], dtype=torch.long)
            boxes_xywh = torch.tensor([[float(value) for value in row[1:5]] for row in rows], dtype=torch.float32)
            scale = torch.tensor([width, height, width, height], dtype=torch.float32)
            boxes_xywh = boxes_xywh * scale
            boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        else:
            labels = torch.zeros((0,), dtype=torch.long)
            boxes_xyxy = torch.zeros((0, 4), dtype=torch.float32)
        targets.append({"sample_id": entry["sample_id"], "boxes_xyxy": boxes_xyxy, "labels": labels})
    return targets


def _result_to_prediction(result) -> dict[str, torch.Tensor]:
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    return {
        "boxes": result.boxes.xyxy.detach().cpu().to(dtype=torch.float32),
        "scores": result.boxes.conf.detach().cpu().to(dtype=torch.float32),
        "labels": result.boxes.cls.detach().cpu().to(dtype=torch.long),
    }


def _predict_branch(
    weights: Path,
    sources: list[Path],
    use_simotm: str,
    channels: int,
    device: str,
    imgsz: int = 640,
    batch: int = 16,
):
    model = YOLO(str(weights))
    channels = max(int(channels), 1)
    model.overrides["channels"] = channels
    model.overrides["use_simotm"] = use_simotm
    model.overrides["pairs_rgb_ir"] = DEFAULT_PAIRS
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(str(path) for path in sources))
        manifest_path = Path(handle.name)
    try:
        predict_args = {
            **model.overrides,
            "source": str(manifest_path),
            "imgsz": imgsz,
            "conf": 0.001,
            "iou": 0.7,
            "batch": batch,
            "device": device,
            "save": False,
            "verbose": False,
            "mode": "predict",
            "use_simotm": use_simotm,
            "channels": channels,
            "pairs_rgb_ir": DEFAULT_PAIRS,
        }
        predictor = model._smart_load("predictor")(overrides=predict_args, _callbacks=model.callbacks)
        predictor.setup_model(model=model.model, verbose=False)
        # Decision-Fusion is an offline branch predictor. Skip warmup here to avoid the broken
        # single-channel dummy input path and run real-image inference directly.
        predictor.done_warmup = True
        results = predictor(source=str(manifest_path), stream=False)
    finally:
        manifest_path.unlink(missing_ok=True)
    return [_result_to_prediction(result) for result in results]


def _fuse_predictions(rgb_predictions, nir_predictions, iou_threshold: float):
    fused = []
    for rgb_pred, nir_pred in zip(rgb_predictions, nir_predictions):
        boxes = torch.cat([rgb_pred["boxes"], nir_pred["boxes"]], dim=0)
        scores = torch.cat([rgb_pred["scores"], nir_pred["scores"]], dim=0)
        labels = torch.cat([rgb_pred["labels"], nir_pred["labels"]], dim=0)
        if boxes.numel() == 0:
            fused.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }
            )
            continue
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        fused.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
    return fused


def _serialize_predictions(predictions):
    return [
        {
            "boxes": prediction["boxes"].tolist(),
            "scores": prediction["scores"].tolist(),
            "labels": prediction["labels"].tolist(),
        }
        for prediction in predictions
    ]


def run_decision_fusion(
    split: str = "val",
    device: str = "0",
    rgb_weights: str | Path | None = None,
    nir_weights: str | Path | None = None,
    iou_threshold: float = 0.7,
):
    entries = _collect_split_entries(split)
    rgb_weight_path = Path(rgb_weights) if rgb_weights else latest_weights_for("rgb")
    nir_weight_path = Path(nir_weights) if nir_weights else latest_weights_for("nir")
    rgb_mode_kwargs = mode_specific_kwargs("rgb")
    nir_mode_kwargs = mode_specific_kwargs("nir")
    rgb_predictions = _predict_branch(
        rgb_weight_path,
        [entry["visible_path"] for entry in entries],
        use_simotm=str(rgb_mode_kwargs["use_simotm"]),
        channels=int(rgb_mode_kwargs["channels"]),
        device=device,
    )
    nir_predictions = _predict_branch(
        nir_weight_path,
        [entry["nir_path"] for entry in entries],
        use_simotm=str(nir_mode_kwargs["use_simotm"]),
        channels=int(nir_mode_kwargs["channels"]),
        device=device,
    )
    fused_predictions = _fuse_predictions(rgb_predictions, nir_predictions, iou_threshold)
    targets = build_eval_targets(_load_targets(split))
    metrics = evaluate_predictions(fused_predictions, targets, num_classes=len(CATEGORY_NAMES))
    return {
        "predictions": _serialize_predictions(fused_predictions),
        "metrics": metrics,
        "metadata": {
            "split": split,
            "rgb_weights": str(rgb_weight_path),
            "nir_weights": str(nir_weight_path),
            "iou_threshold": iou_threshold,
            "classes": CATEGORY_NAMES,
        },
    }


def save_decision_fusion_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split = payload["metadata"]["split"]
    (output_dir / f"predictions_{split}.json").write_text(
        json.dumps(payload["predictions"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(payload["metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "fusion_metadata.json").write_text(
        json.dumps(payload["metadata"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
