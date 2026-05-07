from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


WEATHERS = ["FOG", "LOWLIGHT", "RAIN", "SNOW"]
CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC = [
    "person",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "autorickshaw",
    "traffic light",
    "traffic sign",
]
CATEGORY_TO_ID = {
    "person": 0,
    "rider": 0,
    "motorcycle": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "autorickshaw": 5,
    "traffic light": 6,
    "traffic sign": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paired IDD-AW RGB/NIR images and YOLO labels directly from raw IDD-AW annotations."
    )
    parser.add_argument(
        "--source-root",
        default=r"E:\毕设\code\data_src\RgbNirWithAnnotation\IDDAW",
        help="Raw IDD-AW root containing train/val weather folders.",
    )
    parser.add_argument(
        "--output-root",
        default=r"E:\毕设\code\datasets\iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic",
        help="Target YOLO RGB+NIR dataset root.",
    )
    parser.add_argument("--weathers", nargs="*", default=WEATHERS, help="Weather subsets to export.")
    parser.add_argument("--splits", nargs="*", default=["train", "val"], help="Dataset splits to export.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before exporting.")
    parser.add_argument(
        "--image-mode",
        choices=["copy", "skip"],
        default="copy",
        help="Use 'skip' to generate labels/meta only when images already exist elsewhere.",
    )
    return parser.parse_args()


def frame_id_from_name(path: Path, suffix: str) -> str:
    stem = path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected file name: {path}")
    return stem[: -len(suffix)]


def discover_sequences(source_root: Path, split: str, weather: str) -> list[str]:
    rgb_root = source_root / split / weather / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(f"Missing rgb directory: {rgb_root}")
    return sorted(path.name for path in rgb_root.iterdir() if path.is_dir())


def sequence_frames(directory: Path, suffix: str, extension: str) -> dict[str, Path]:
    return {frame_id_from_name(path, suffix): path for path in sorted(directory.glob(f"*{suffix}{extension}"))}


def polygon_to_yolo_box(polygon: list[list[float]], width: int, height: int) -> tuple[float, float, float, float] | None:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1.0 or box_h <= 1.0:
        return None
    return (
        (x1 + box_w / 2.0) / float(width),
        (y1 + box_h / 2.0) / float(height),
        box_w / float(width),
        box_h / float(height),
    )


def parse_yolo_lines(json_path: Path, width: int, height: int) -> tuple[list[str], Counter[str]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for obj in payload.get("objects", []):
        if int(obj.get("deleted", 0)) != 0:
            continue
        label = str(obj.get("label", ""))
        if label not in CATEGORY_TO_ID:
            continue
        polygon = obj.get("polygon", [])
        if len(polygon) < 3:
            continue
        box = polygon_to_yolo_box(polygon, width, height)
        if box is None:
            continue
        class_id = CATEGORY_TO_ID[label]
        line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        counts[CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC[class_id]] += 1
    return lines, counts


def ensure_dirs(root: Path, splits: list[str]) -> None:
    for split in splits:
        (root / "visible" / split).mkdir(parents=True, exist_ok=True)
        (root / "nir" / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def export_split(source_root: Path, output_root: Path, split: str, weathers: list[str], image_mode: str) -> dict[str, object]:
    split_stats: dict[str, object] = {
        "samples": 0,
        "images": 0,
        "labels": 0,
        "annotations": 0,
        "class_counts": Counter(),
        "weather_counts": Counter(),
        "pairing_warnings": [],
    }
    visible_split_dir = output_root / "visible" / split
    nir_split_dir = output_root / "nir" / split

    for weather in weathers:
        split_root = source_root / split / weather
        for seq in discover_sequences(source_root, split, weather):
            rgb_dir = split_root / "rgb" / seq
            nir_dir = split_root / "nir" / seq
            gt_dir = split_root / "gtSeg" / seq
            rgb_map = sequence_frames(rgb_dir, "_rgb", ".png")
            nir_map = sequence_frames(nir_dir, "_nir", ".png")
            json_map = sequence_frames(gt_dir, "_mask", ".json")
            common_keys = sorted(set(rgb_map) & set(nir_map) & set(json_map))
            if set(rgb_map) != set(nir_map) or set(rgb_map) != set(json_map):
                split_stats["pairing_warnings"].append(
                    {
                        "weather": weather.lower(),
                        "sequence": seq,
                        "rgb_count": len(rgb_map),
                        "nir_count": len(nir_map),
                        "json_count": len(json_map),
                        "kept_count": len(common_keys),
                    }
                )

            for frame in common_keys:
                vis_src = rgb_map[frame]
                nir_src = nir_map[frame]
                ann_src = json_map[frame]
                with Image.open(vis_src) as vis_image:
                    width, height = vis_image.size
                sample_stem = f"{split}_{weather}_{seq}_{frame}"
                image_name = f"{sample_stem}{vis_src.suffix.lower()}"
                label_name = f"{sample_stem}.txt"
                if image_mode == "copy":
                    shutil.copy2(vis_src, visible_split_dir / image_name)
                    shutil.copy2(nir_src, nir_split_dir / image_name)
                    split_stats["images"] += 2

                lines, counts = parse_yolo_lines(ann_src, width, height)
                label_text = "\n".join(lines)
                (visible_split_dir / label_name).write_text(label_text, encoding="utf-8")
                (nir_split_dir / label_name).write_text(label_text, encoding="utf-8")
                split_stats["labels"] += 2
                split_stats["samples"] += 1
                split_stats["annotations"] += len(lines)
                split_stats["class_counts"].update(counts)
                split_stats["weather_counts"][weather.lower()] += 1

    split_stats["class_counts"] = dict(split_stats["class_counts"])
    split_stats["weather_counts"] = dict(split_stats["weather_counts"])
    return split_stats


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    weathers = [weather.upper() for weather in args.weathers]
    splits = [split.lower() for split in args.splits]

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dirs(output_root, splits)

    split_reports = {
        split: export_split(source_root, output_root, split, weathers, args.image_mode)
        for split in splits
    }
    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "schema": "8cls_personmerge_traffic",
        "categories": CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC,
        "raw_category_mapping": CATEGORY_TO_ID,
        "image_mode": args.image_mode,
        "splits": split_reports,
    }
    (output_root / "meta" / "export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
