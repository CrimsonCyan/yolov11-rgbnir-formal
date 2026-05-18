from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.iddaw import category_names_for_mode, latest_weights_for, mode_specific_kwargs, resolve_dataset_root  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
AUTO_LAYER_KEYWORDS = ("BiFPN", "ObjectAware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO intermediate feature maps with forward hooks.")
    parser.add_argument("--mode", required=True, help="Mode name used by formal_rgbnir.iddaw.")
    parser.add_argument("--weights", required=True, help="Checkpoint path, usually weights/best.pt. Use 'latest' for local latest best.pt.")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root override.")
    parser.add_argument("--source", default="", help="Optional image file or directory override.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", default="runs/analysis/feature_maps", help="Output directory.")
    parser.add_argument("--name", default="", help="Output name. Defaults to mode + checkpoint run name.")
    parser.add_argument(
        "--layers",
        default="auto",
        help="Comma-separated top-level model layer indices, e.g. 4,10,28. Use 'auto' for BiFPN/OA layers.",
    )
    parser.add_argument(
        "--module-regex",
        default="",
        help="Optional regex over named_modules(); useful for internal modules such as blocks.0.p2_out_fuse.",
    )
    parser.add_argument("--list-layers", action="store_true", help="Print top-level layers and exit.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-feature-panels", type=int, default=16)
    parser.add_argument("--draw-gt", action="store_true", help="Draw YOLO GT boxes on RGB/NIR input panels.")
    parser.add_argument("--draw-pred", action="store_true", help="Draw prediction boxes on the RGB input panel.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha.")
    parser.add_argument(
        "--paper-layout",
        action="store_true",
        help="Also save a fixed paper grid: RGB | NIR | RGB P2 | NIR P2 | OA/BiFPN P2 | Detect input.",
    )
    parser.add_argument(
        "--feature-titles",
        default="RGB P2 feature,NIR P2 feature,OA/BiFPN P2 feature,Detect input feature",
        help="Comma-separated titles for the first four captured feature maps used by --paper-layout.",
    )
    return parser.parse_args()


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("visualize_feature_maps.py requires opencv-python/cv2.") from exc
    return cv2


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "feature-maps"


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


def resolve_images(args: argparse.Namespace) -> list[Path]:
    if args.source:
        return list_images(Path(args.source).expanduser().resolve(), args.max_images)
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else resolve_dataset_root(args.mode)
    source_dir = dataset_root / source_subdir_for_mode(args.mode) / args.split
    return list_images(source_dir, args.max_images)


def paired_nir_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "visible" not in parts:
        return None
    idx = parts.index("visible")
    parts[idx] = "nir"
    nir_path = Path(*parts)
    return nir_path if nir_path.exists() else None


def read_image_bgr(path: Path):
    cv2 = import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    cv2 = import_cv2()
    pad = 34
    canvas = np.full((image.shape[0] + pad, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[pad:] = image
    cv2.putText(canvas, title[:70], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def normalize_map(feature_map: torch.Tensor) -> np.ndarray:
    data = feature_map.detach().float().cpu().numpy()
    data = data - float(np.nanmin(data))
    denom = float(np.nanmax(data))
    if denom > 1e-12:
        data = data / denom
    return np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)


def overlay_heatmap(base_bgr: np.ndarray, feature_map: torch.Tensor, title: str, alpha: float) -> np.ndarray:
    cv2 = import_cv2()
    heat = normalize_map(feature_map)
    heat = cv2.resize(heat, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0)
    return add_title(overlay, title)


def draw_yolo_gt(image: np.ndarray, image_path: Path, class_names: list[str]) -> np.ndarray:
    cv2 = import_cv2()
    out = image.copy()
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return out
    h, w = out.shape[:2]
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if len(parts) == 5:
            cx, cy, bw, bh = [float(value) for value in parts[1:5]]
        else:
            coords = [float(value) for value in parts[1:]]
            xs = coords[0::2]
            ys = coords[1::2]
            if not xs or not ys:
                continue
            x1n, x2n = min(xs), max(xs)
            y1n, y2n = min(ys), max(ys)
            cx, cy, bw, bh = (x1n + x2n) / 2.0, (y1n + y2n) / 2.0, x2n - x1n, y2n - y1n
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(out, f"GT {name}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def load_gt_boxes_xyxy(image_path: Path, image_shape: tuple[int, int]) -> list[tuple[int, float, float, float, float]]:
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        return []
    h, w = image_shape
    boxes: list[tuple[int, float, float, float, float]] = []
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
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def activation_gt_stats(
    activation: torch.Tensor,
    gt_boxes: list[tuple[int, float, float, float, float]],
    image_shape: tuple[int, int],
) -> dict[str, float | int | None]:
    if not gt_boxes:
        return {"gt_count": 0, "gt_inside_mean": None, "gt_outside_mean": None, "gt_inside_outside_ratio": None}
    act = activation.detach().float().cpu()
    h_feat, w_feat = int(act.shape[0]), int(act.shape[1])
    h_img, w_img = image_shape
    mask = torch.zeros((h_feat, w_feat), dtype=torch.bool)
    for _cls, x1, y1, x2, y2 in gt_boxes:
        fx1 = int(np.floor(x1 / max(w_img, 1) * w_feat))
        fy1 = int(np.floor(y1 / max(h_img, 1) * h_feat))
        fx2 = int(np.ceil(x2 / max(w_img, 1) * w_feat))
        fy2 = int(np.ceil(y2 / max(h_img, 1) * h_feat))
        fx1 = max(0, min(w_feat - 1, fx1))
        fy1 = max(0, min(h_feat - 1, fy1))
        fx2 = max(fx1 + 1, min(w_feat, fx2))
        fy2 = max(fy1 + 1, min(h_feat, fy2))
        mask[fy1:fy2, fx1:fx2] = True
    inside = act[mask]
    outside = act[~mask]
    inside_mean = float(inside.mean().item()) if inside.numel() else None
    outside_mean = float(outside.mean().item()) if outside.numel() else None
    ratio = None
    if inside_mean is not None and outside_mean is not None:
        ratio = inside_mean / max(outside_mean, 1e-12)
    return {
        "gt_count": len(gt_boxes),
        "gt_inside_mean": inside_mean,
        "gt_outside_mean": outside_mean,
        "gt_inside_outside_ratio": ratio,
    }


def draw_predictions(image: np.ndarray, result, class_names: list[str], conf: float) -> np.ndarray:
    cv2 = import_cv2()
    out = image.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy()
    labels = boxes.cls.detach().cpu().numpy().astype(int)
    for box, score, cls in zip(xyxy, scores, labels):
        if float(score) < conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"{name} {score:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
    return out


def print_layers(model: Any) -> None:
    top = model.model.model
    for idx, module in enumerate(top):
        params = sum(param.numel() for param in module.parameters())
        print(f"{idx:>3}  {module.__class__.__name__:<48} params={params}")


def auto_layer_indices(model: Any) -> list[int]:
    top = model.model.model
    indices = [
        idx
        for idx, module in enumerate(top)
        if any(keyword in module.__class__.__name__ for keyword in AUTO_LAYER_KEYWORDS)
    ]
    if indices:
        return indices
    # Fallback for plain YOLO/PAN models: visualize the last four feature-producing layers before Detect.
    detect_idx = next((idx for idx, module in enumerate(top) if module.__class__.__name__ == "Detect"), len(top))
    start = max(0, detect_idx - 4)
    return list(range(start, detect_idx))


def parse_layer_indices(value: str, model: Any) -> list[int]:
    if value.strip().lower() == "auto":
        return auto_layer_indices(model)
    indices = []
    top_len = len(model.model.model)
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0:
            idx += top_len
        if not 0 <= idx < top_len:
            raise ValueError(f"Layer index out of range: {raw}; valid range is 0..{top_len - 1}")
        indices.append(idx)
    if not indices:
        raise ValueError("No valid --layers were provided.")
    return indices


def tensor_maps(value: Any) -> list[tuple[str, torch.Tensor]]:
    maps: list[tuple[str, torch.Tensor]] = []

    def visit(item: Any, suffix: str) -> None:
        if isinstance(item, torch.Tensor) and item.ndim == 4 and item.shape[0] > 0:
            maps.append((suffix, item.detach().float().cpu()))
        elif isinstance(item, (list, tuple)):
            for idx, child in enumerate(item):
                visit(child, f"{suffix}.{idx}" if suffix else str(idx))

    visit(value, "")
    return maps


def feature_activation(feature: torch.Tensor) -> torch.Tensor:
    # Shape: BCHW. Use channel-wise L1 activation for stable visual comparisons.
    return feature[0].abs().mean(0)


def iter_captured_features(
    captures: dict[str, list[tuple[str, torch.Tensor]]],
) -> list[tuple[str, str, torch.Tensor]]:
    features: list[tuple[str, str, torch.Tensor]] = []
    for module_name, maps in captures.items():
        for map_suffix, feature in maps:
            features.append((module_name, map_suffix or "out", feature))
    return features


def install_hooks(model: Any, layer_indices: list[int], module_regex: str, captures: dict[str, list[tuple[str, torch.Tensor]]]):
    handles = []
    top = model.model.model
    for idx in layer_indices:
        module = top[idx]
        name = f"layer{idx}_{module.__class__.__name__}"

        def hook(_module, _inp, out, capture_name=name):
            captures[capture_name] = tensor_maps(out)

        handles.append(module.register_forward_hook(hook))

    if module_regex:
        pattern = re.compile(module_regex)
        for name, module in model.model.named_modules():
            if pattern.search(name):
                capture_name = sanitize(name)

                def hook(_module, _inp, out, capture_name=capture_name):
                    captures[capture_name] = tensor_maps(out)

                handles.append(module.register_forward_hook(hook))
    return handles


def resize_panel(panel: np.ndarray, height: int) -> np.ndarray:
    cv2 = import_cv2()
    if panel.shape[0] == height:
        return panel
    width = int(round(panel.shape[1] * (height / panel.shape[0])))
    return cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)


def save_grid(path: Path, panels: list[np.ndarray]) -> None:
    cv2 = import_cv2()
    if not panels:
        return
    target_h = min(panel.shape[0] for panel in panels)
    resized = [resize_panel(panel, target_h) for panel in panels]
    grid = cv2.hconcat(resized)
    cv2.imwrite(str(path), grid)


def main() -> None:
    args = parse_args()
    cv2 = import_cv2()
    images = resolve_images(args)
    from ultralytics import YOLO

    weights_path = latest_weights_for(args.mode, "best.pt") if args.weights.lower() in {"latest", "best", "best.pt"} else Path(args.weights).expanduser().resolve()
    model = YOLO(str(weights_path))

    if args.list_layers:
        print_layers(model)
        return

    layer_indices = parse_layer_indices(args.layers, model)
    class_names = category_names_for_mode(args.mode)
    kwargs = mode_specific_kwargs(args.mode)
    captures: dict[str, list[tuple[str, torch.Tensor]]] = {}
    handles = install_hooks(model, layer_indices, args.module_regex, captures)

    run_name = args.name or f"{sanitize(args.mode)}_{sanitize(weights_path.parent.parent.name)}"
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, object]] = []

    try:
        for image_path in images:
            captures.clear()
            base = read_image_bgr(image_path)
            rgb_panel = base.copy()
            if args.draw_gt:
                rgb_panel = draw_yolo_gt(rgb_panel, image_path, class_names)

            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                conf=min(args.conf, 0.001),
                iou=args.iou,
                max_det=args.max_det,
                half=args.half,
                save=False,
                verbose=False,
                stream=False,
                **kwargs,
            )
            result = results[0]
            if args.draw_pred:
                rgb_panel = draw_predictions(rgb_panel, result, class_names, args.conf)

            image_dir = out_dir / image_path.stem
            image_dir.mkdir(parents=True, exist_ok=True)
            panels = [add_title(rgb_panel, "RGB input" + (" + GT/Pred" if args.draw_gt or args.draw_pred else ""))]
            paper_panels = [add_title(rgb_panel, "RGB input")]
            gt_boxes = load_gt_boxes_xyxy(image_path, base.shape[:2])

            nir_path = paired_nir_path(image_path)
            if nir_path is not None:
                nir_image = read_image_bgr(nir_path)
                if args.draw_gt:
                    nir_image = draw_yolo_gt(nir_image, image_path, class_names)
                panels.append(add_title(nir_image, "NIR input"))
                paper_panels.append(add_title(nir_image, "NIR input"))

            saved_feature_panels = 0
            captured_features = iter_captured_features(captures)
            for module_name, map_suffix, feature in captured_features:
                if saved_feature_panels >= args.max_feature_panels:
                    break
                activation = feature_activation(feature)
                title = f"{module_name}:{map_suffix} {tuple(feature.shape[1:])}"
                heat_panel = overlay_heatmap(base, activation, title, args.alpha)
                file_stem = sanitize(f"{image_path.stem}_{module_name}_{map_suffix}")
                cv2.imwrite(str(image_dir / f"{file_stem}.jpg"), heat_panel)
                panels.append(heat_panel)
                stats_rows.append(
                    {
                        "image": image_path.name,
                        "module": module_name,
                        "output": map_suffix,
                        "channels": int(feature.shape[1]),
                        "height": int(feature.shape[2]),
                        "width": int(feature.shape[3]),
                        "activation_mean": float(activation.mean().item()),
                        "activation_max": float(activation.max().item()),
                        **activation_gt_stats(activation, gt_boxes, base.shape[:2]),
                    }
                )
                saved_feature_panels += 1

            save_grid(image_dir / f"{image_path.stem}_feature_grid.jpg", panels)
            if args.paper_layout:
                feature_titles = [title.strip() for title in args.feature_titles.split(",") if title.strip()]
                while len(feature_titles) < 4:
                    feature_titles.append(f"Feature {len(feature_titles) + 1}")
                for idx, (_module_name, _map_suffix, feature) in enumerate(captured_features[:4]):
                    activation = feature_activation(feature)
                    paper_panels.append(overlay_heatmap(base, activation, feature_titles[idx], args.alpha))
                save_grid(image_dir / f"{image_path.stem}_paper_grid.jpg", paper_panels)
            print(f"saved={image_dir / f'{image_path.stem}_feature_grid.jpg'}")
    finally:
        for handle in handles:
            handle.remove()

    with (out_dir / "feature_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "image",
            "module",
            "output",
            "channels",
            "height",
            "width",
            "activation_mean",
            "activation_max",
            "gt_count",
            "gt_inside_mean",
            "gt_outside_mean",
            "gt_inside_outside_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    summary = {
        "mode": args.mode,
        "weights": str(weights_path),
        "images": [str(path) for path in images],
        "layers": layer_indices,
        "module_regex": args.module_regex,
        "num_feature_maps": len(stats_rows),
        "out": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"feature_stats={out_dir / 'feature_stats.csv'}")


if __name__ == "__main__":
    main()
