from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.iddaw import mode_specific_kwargs, resolve_dataset_root  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO FPS/latency on IDD-AW visible/NIR/RGB-NIR modes.")
    parser.add_argument("--mode", required=True, help="Mode name used by formal_rgbnir.iddaw.")
    parser.add_argument("--weights", required=True, help="Checkpoint path, usually weights/best.pt.")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root override.")
    parser.add_argument("--source", default="", help="Optional image file or directory override.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", default="runs/analysis/fps", help="Output directory.")
    parser.add_argument("--name", default="", help="Output name. Defaults to mode + checkpoint run name.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup-images", type=int, default=32, help="Images used for warmup before timing.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the timed source list this many times.")
    parser.add_argument("--max-images", type=int, default=200, help="0 means use the full split.")
    parser.add_argument("--workers", type=int, default=0, help="Predict workers override if supported.")
    return parser.parse_args()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "fps"


def source_subdir_for_mode(mode: str) -> str:
    kwargs = mode_specific_kwargs(mode)
    return "nir" if kwargs.get("use_simotm") == "Gray" else "visible"


def list_images(path: Path, max_images: int = 0) -> list[Path]:
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


def resolve_source(args: argparse.Namespace) -> list[Path]:
    if args.source:
        return list_images(Path(args.source).expanduser().resolve(), args.max_images)
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else resolve_dataset_root(args.mode)
    source_dir = dataset_root / source_subdir_for_mode(args.mode) / args.split
    return list_images(source_dir, args.max_images)


def cuda_sync(device: str) -> None:
    if device.lower() != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def write_source_list(sources: list[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(source) for source in sources) + "\n", encoding="utf-8")
    return path


def predict_once(model: Any, source_file: Path, args: argparse.Namespace):
    kwargs = mode_specific_kwargs(args.mode)
    predict_kwargs = {
        "source": str(source_file),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "half": args.half,
        "save": False,
        "verbose": False,
        "stream": False,
    }
    if args.workers > 0:
        predict_kwargs["workers"] = args.workers
    return model.predict(**predict_kwargs, **kwargs)


def speed_summary(results) -> dict[str, float]:
    speed_rows = [getattr(result, "speed", {}) or {} for result in results]
    keys = ("preprocess", "inference", "postprocess")
    summary = {f"{key}_ms_img": mean(float(row.get(key, 0.0)) for row in speed_rows) for key in keys}
    summary["ultralytics_total_ms_img"] = sum(summary[f"{key}_ms_img"] for key in keys)
    summary["ultralytics_fps"] = 1000.0 / summary["ultralytics_total_ms_img"] if summary["ultralytics_total_ms_img"] > 0 else 0.0
    return summary


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    sources = resolve_source(args)
    warmup_sources = sources[: max(1, min(len(sources), args.warmup_images))]
    from ultralytics import YOLO

    model = YOLO(args.weights)
    run_name = args.name or f"{sanitize(args.mode)}_{sanitize(Path(args.weights).parent.parent.name)}"
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    source_file = write_source_list(sources, out_dir / "source_images.txt")
    warmup_source_file = write_source_list(warmup_sources, out_dir / "warmup_images.txt")

    if args.warmup_images > 0:
        _ = predict_once(model, warmup_source_file, args)

    timed_results = []
    cuda_sync(args.device)
    start = time.perf_counter()
    for _ in range(args.repeat):
        timed_results.extend(predict_once(model, source_file, args))
    cuda_sync(args.device)
    elapsed = time.perf_counter() - start

    total_images = len(sources) * args.repeat
    summary = speed_summary(timed_results)
    summary.update(
        {
            "mode": args.mode,
            "weights": str(Path(args.weights).expanduser().resolve()),
            "num_unique_images": len(sources),
            "repeat": args.repeat,
            "total_images": total_images,
            "wall_time_sec": elapsed,
            "wall_ms_img": elapsed * 1000.0 / total_images,
            "wall_fps": total_images / elapsed if elapsed > 0 else 0.0,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "half": bool(args.half),
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        }
    )

    (out_dir / "fps_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "fps_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_json={out_dir / 'fps_summary.json'}")
    print(f"summary_csv={out_dir / 'fps_summary.csv'}")


if __name__ == "__main__":
    main()
