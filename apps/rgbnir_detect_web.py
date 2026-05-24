from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GRADIO_TEMP_DIR = Path(
    os.getenv("GRADIO_TEMP_DIR", str(ROOT / "runs/web_detect/gradio_tmp"))
)
DEFAULT_GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_TEMP_DIR", str(DEFAULT_GRADIO_TEMP_DIR))
os.environ.setdefault("TMPDIR", str(DEFAULT_GRADIO_TEMP_DIR))
os.environ.setdefault("TEMP", str(DEFAULT_GRADIO_TEMP_DIR))
os.environ.setdefault("TMP", str(DEFAULT_GRADIO_TEMP_DIR))

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from ultralytics import YOLO  # noqa: E402


DEFAULT_WEIGHTS = (
    ROOT
    / "runs/IDD_AW/MGFDet/weights/best.pt"
)
DEFAULT_OUT_ROOT = ROOT / "runs/web_detect"
DEFAULT_RGB_FOLDER = os.getenv(
    "RGB_FOLDER",
    "/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640/visible/val",
)
DEFAULT_NIR_FOLDER = os.getenv(
    "NIR_FOLDER",
    "/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640/nir/val",
)
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
IMAGE_PREVIEW_HEIGHT = 420
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TABLE_HEADERS = ["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "width", "height"]
FIXED_IMAGE_CSS = """
.fixed-image-preview {
    height: 460px;
}
.fixed-image-preview img,
.fixed-image-preview canvas {
    object-fit: contain !important;
    max-height: 420px !important;
}
"""
CLASS_COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (69, 123, 157),
    (131, 56, 236),
    (255, 127, 80),
    (30, 144, 255),
    (50, 205, 50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-NIR paired image detection web demo.")
    parser.add_argument("--weights", default=os.getenv("WEIGHTS", str(DEFAULT_WEIGHTS)))
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--device", default=os.getenv("DEVICE", "0"))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("IMGSZ", "640")))
    parser.add_argument("--conf", type=float, default=float(os.getenv("CONF", "0.25")))
    parser.add_argument("--iou", type=float, default=float(os.getenv("IOU", "0.7")))
    parser.add_argument("--max-det", type=int, default=int(os.getenv("MAX_DET", "300")))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def ensure_weights(weights: str | Path) -> Path:
    path = Path(weights).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Weight file does not exist: {path}. Set WEIGHTS=/path/to/best.pt or pass --weights."
        )
    return path


@lru_cache(maxsize=2)
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def model_names(model: YOLO) -> dict[int, str]:
    names = getattr(model, "names", None) or getattr(model.model, "names", None) or {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def effective_batch_size(device: str, batch_size: int) -> int:
    batch_size = max(1, int(batch_size))
    device_ids = [item.strip() for item in str(device).split(",") if item.strip()]
    if len(device_ids) <= 1:
        return batch_size
    remainder = batch_size % len(device_ids)
    if remainder == 0:
        return batch_size
    return batch_size + len(device_ids) - remainder


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(np.percentile(values[finite], 1))
    hi = float(np.percentile(values[finite], 99))
    if hi <= lo:
        hi = float(values[finite].max())
        lo = float(values[finite].min())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.clip((values - lo) * 255.0 / (hi - lo), 0, 255)
    return values.astype(np.uint8)


def read_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read RGB image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def read_nir_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read NIR image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize_to_uint8(image)


def make_rgbnir_input(rgb_path: str | Path, nir_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    bgr, rgb = read_rgb(rgb_path)
    nir = read_nir_gray(nir_path)
    warnings: list[str] = []
    if nir.shape[:2] != bgr.shape[:2]:
        warnings.append(
            f"NIR image resized from {nir.shape[1]}x{nir.shape[0]} to RGB size {bgr.shape[1]}x{bgr.shape[0]}."
        )
        nir = cv2.resize(nir, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    rgbnir = cv2.merge((bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], nir))
    return rgbnir, rgb, warnings


def clip_box(box: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return [x1, y1, x2, y2]


def rows_from_result(
    result: Any,
    image_name: str,
    width: int,
    height: int,
    names: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if result.boxes is None or not len(result.boxes):
        return rows
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    for box, score, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = clip_box(box, width, height)
        rows.append(
            {
                "image": image_name,
                "class_id": int(cls),
                "class_name": names.get(int(cls), str(cls)),
                "confidence": round(float(score), 6),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(max(0.0, x2 - x1), 2),
                "height": round(max(0.0, y2 - y1), 2),
            }
        )
    return rows


def predict_pair(
    model: YOLO,
    rgb_path: str | Path,
    nir_path: str | Path,
    *,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    rgbnir, rgb, warnings = make_rgbnir_input(rgb_path, nir_path)
    results = model.predict(
        source=[rgbnir],
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        channels=4,
        batch=effective_batch_size(device, DEFAULT_BATCH_SIZE),
        save=False,
        verbose=False,
    )
    result = results[0]
    names = model_names(model)
    height, width = rgb.shape[:2]
    rows = rows_from_result(result, Path(rgb_path).name, width, height, names)
    return rgb, rows, warnings


def draw_detections(rgb: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    image = rgb.copy()
    for row in rows:
        cls = int(row["class_id"])
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        x1, y1, x2, y2 = (int(round(float(row[key]))) for key in ("x1", "y1", "x2", "y2"))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f'{row["class_name"]} {float(row["confidence"]):.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(y1, th + baseline + 3)
        cv2.rectangle(image, (x1, y_text - th - baseline - 3), (x1 + tw + 4, y_text + 3), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 2, y_text - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def table_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    return [[row[key] for key in TABLE_HEADERS] for row in rows]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def single_detect(
    rgb_file: str | None,
    nir_file: str | None,
    conf: float,
    iou: float,
    device: str,
    weights: str,
    imgsz: int,
    max_det: int,
) -> tuple[str, list[list[Any]], dict[str, Any], str]:
    if not rgb_file or not nir_file:
        raise gr.Error("Please upload both RGB and NIR images.")
    imgsz = int(imgsz)
    max_det = int(max_det)
    conf = float(conf)
    iou = float(iou)
    device = str(device).strip() or "0"
    weight_path = ensure_weights(weights)
    model = load_model(str(weight_path))
    rgb, rows, warnings = predict_pair(
        model,
        rgb_file,
        nir_file,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
    )
    annotated = draw_detections(rgb, rows)
    out_dir = DEFAULT_OUT_ROOT / "single" / timestamp()
    out_path = out_dir / f"{Path(rgb_file).stem}_annotated.png"
    save_rgb(out_path, annotated)
    payload = {
        "rgb": str(rgb_file),
        "nir": str(nir_file),
        "weights": str(weight_path),
        "imgsz": imgsz,
        "device": device,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "warnings": warnings,
        "detections": rows,
    }
    return str(out_path), table_rows(rows), payload, str(out_path)


def list_images(folder: str | Path) -> tuple[dict[str, Path], list[str]]:
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise gr.Error(f"Folder does not exist: {root}")
    images: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images:
                duplicates.append(path.stem)
                continue
            images[path.stem] = path
    if not images:
        raise gr.Error(f"No supported images found in {root}")
    return images, duplicates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in TABLE_HEADERS})


def make_zip(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(source_dir))


def batch_detect(
    rgb_folder: str,
    nir_folder: str,
    conf: float,
    iou: float,
    device: str,
    weights: str,
    imgsz: int,
    max_det: int,
    batch_size: int,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str, dict[str, Any]]:
    imgsz = int(imgsz)
    max_det = int(max_det)
    batch_size = effective_batch_size(device, batch_size)
    conf = float(conf)
    iou = float(iou)
    device = str(device).strip() or "0"
    weight_path = ensure_weights(weights)
    model = load_model(str(weight_path))
    rgb_files, dup_rgb = list_images(rgb_folder)
    nir_files, dup_nir = list_images(nir_folder)
    common = sorted(set(rgb_files) & set(nir_files))
    if not common:
        raise gr.Error("No RGB/NIR image pairs matched by identical filename stem.")

    out_dir = DEFAULT_OUT_ROOT / timestamp()
    ann_dir = out_dir / "annotated"
    all_rows: list[dict[str, Any]] = []
    warnings: dict[str, list[str]] = {}
    failed: dict[str, str] = {}
    total = len(common)
    names = model_names(model)
    progress((0, total), desc=f"Preparing {total} RGB/NIR pairs; batch={batch_size}")
    for start in range(0, total, batch_size):
        chunk = common[start : start + batch_size]
        end = min(start + len(chunk), total)
        progress((start, total), desc=f"Loading {start + 1}-{end}/{total}")
        batch_inputs: list[np.ndarray] = []
        batch_rgbs: list[np.ndarray] = []
        batch_stems: list[str] = []
        batch_rgb_paths: list[Path] = []
        for stem in chunk:
            rgb_path = rgb_files[stem]
            nir_path = nir_files[stem]
            try:
                rgbnir, rgb, pair_warnings = make_rgbnir_input(rgb_path, nir_path)
                batch_inputs.append(rgbnir)
                batch_rgbs.append(rgb)
                batch_stems.append(stem)
                batch_rgb_paths.append(rgb_path)
                if pair_warnings:
                    warnings[stem] = pair_warnings
            except Exception as exc:
                failed[stem] = str(exc)
        if not batch_inputs:
            progress((end, total), desc=f"Skipped {start + 1}-{end}/{total}")
            continue

        progress((start, total), desc=f"Detecting {start + 1}-{end}/{total}")
        try:
            results = model.predict(
                source=batch_inputs,
                imgsz=imgsz,
                device=device,
                conf=conf,
                iou=iou,
                max_det=max_det,
                channels=4,
                batch=batch_size,
                save=False,
                verbose=False,
            )
            for stem, rgb_path, rgb, result in zip(batch_stems, batch_rgb_paths, batch_rgbs, results):
                height, width = rgb.shape[:2]
                rows = rows_from_result(result, rgb_path.name, width, height, names)
                annotated = draw_detections(rgb, rows)
                out_image = ann_dir / f"{stem}_annotated.png"
                save_rgb(out_image, annotated)
                all_rows.extend(rows)
        except Exception as exc:
            message = f"batch inference failed for {start + 1}-{end}/{total}: {exc}"
            for stem in batch_stems:
                failed[stem] = message
        progress((end, total), desc=f"Detected {end}/{total}")

    csv_path = out_dir / "detections.csv"
    summary_path = out_dir / "summary.json"
    zip_path = out_dir / "rgbnir_detections.zip"
    write_csv(csv_path, all_rows)
    summary = {
        "weights": str(weight_path),
        "rgb_folder": str(Path(rgb_folder).expanduser()),
        "nir_folder": str(Path(nir_folder).expanduser()),
        "imgsz": imgsz,
        "device": device,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "batch_size": batch_size,
        "matched_pairs": len(common),
        "processed_pairs": len(common) - len(failed),
        "detections": len(all_rows),
        "unmatched_rgb": sorted(set(rgb_files) - set(nir_files)),
        "unmatched_nir": sorted(set(nir_files) - set(rgb_files)),
        "duplicate_rgb_stems": sorted(set(dup_rgb)),
        "duplicate_nir_stems": sorted(set(dup_nir)),
        "resize_warnings": warnings,
        "failed": failed,
        "output_dir": str(out_dir),
        "annotated_dir": str(ann_dir),
        "detections_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_zip(out_dir, zip_path)
    status = (
        f"Output: {out_dir}\n"
        f"Matched pairs: {len(common)}; processed: {summary['processed_pairs']}; "
        f"detections: {len(all_rows)}; failed: {len(failed)}; batch_size: {batch_size}"
    )
    return status, str(zip_path), summary


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    default_weights = str(ensure_weights(args.weights))
    with gr.Blocks(title="RGB-NIR Detection Demo") as demo:
        gr.Markdown(
            "# RGB-NIR 配对图像检测\n"
            "默认模型：RGB + 轻量 NIR 分支 + P2-P5 BiFPN + MGF。"
            "输出仅在 RGB 图像上绘制检测框，NIR 只作为模型输入。"
        )
        with gr.Row():
            weights_box = gr.Textbox(label="Weights", value=default_weights, lines=1)
            device_box = gr.Textbox(label="Device", value=args.device, lines=1)
            imgsz_box = gr.Number(label="Image size", value=args.imgsz, precision=0)
            max_det_box = gr.Number(label="Max detections", value=args.max_det, precision=0)
            batch_size_box = gr.Number(label="Batch size", value=args.batch_size, precision=0)
        with gr.Row():
            conf_slider = gr.Slider(0.01, 0.95, value=args.conf, step=0.01, label="Confidence")
            iou_slider = gr.Slider(0.1, 0.95, value=args.iou, step=0.01, label="NMS IoU")

        with gr.Tab("单对 RGB/NIR 图像"):
            with gr.Row():
                rgb_input = gr.Image(
                    label="RGB image",
                    type="filepath",
                    height=IMAGE_PREVIEW_HEIGHT,
                    elem_classes=["fixed-image-preview"],
                )
                nir_input = gr.Image(
                    label="NIR image",
                    type="filepath",
                    height=IMAGE_PREVIEW_HEIGHT,
                    elem_classes=["fixed-image-preview"],
                )
            run_single = gr.Button("Run detection", variant="primary")
            with gr.Row():
                annotated_output = gr.Image(
                    label="Annotated RGB",
                    type="filepath",
                    height=IMAGE_PREVIEW_HEIGHT,
                    elem_classes=["fixed-image-preview"],
                )
                download_output = gr.File(label="Download annotated image")
            table_output = gr.Dataframe(headers=TABLE_HEADERS, label="Detections")
            json_output = gr.JSON(label="JSON result")
            run_single.click(
                single_detect,
                inputs=[rgb_input, nir_input, conf_slider, iou_slider, device_box, weights_box, imgsz_box, max_det_box],
                outputs=[annotated_output, table_output, json_output, download_output],
            )

        with gr.Tab("批量 RGB/NIR 文件夹"):
            rgb_folder = gr.Textbox(label="RGB folder path", value=DEFAULT_RGB_FOLDER)
            nir_folder = gr.Textbox(label="NIR folder path", value=DEFAULT_NIR_FOLDER)
            run_batch = gr.Button("Run batch detection", variant="primary")
            batch_status = gr.Textbox(label="Status", lines=4)
            batch_zip = gr.File(label="Result zip")
            batch_json = gr.JSON(label="Summary")
            run_batch.click(
                batch_detect,
                inputs=[
                    rgb_folder,
                    nir_folder,
                    conf_slider,
                    iou_slider,
                    device_box,
                    weights_box,
                    imgsz_box,
                    max_det_box,
                    batch_size_box,
                ],
                outputs=[batch_status, batch_zip, batch_json],
            )
    return demo


def main() -> None:
    args = parse_args()
    ensure_weights(args.weights)
    DEFAULT_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo = build_demo(args)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=False, show_error=True, css=FIXED_IMAGE_CSS)


if __name__ == "__main__":
    main()
