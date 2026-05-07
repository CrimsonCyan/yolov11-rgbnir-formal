from __future__ import annotations

import argparse
import json
import os
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
        choices=["copy", "hardlink", "skip"],
        default="copy",
        help="Use 'skip' to generate labels/meta only; use 'hardlink' to avoid duplicating image bytes.",
    )
    parser.add_argument(
        "--detectable-filter",
        action="store_true",
        help="Drop extreme tiny/unrecognizable instances using resized bbox and polygon-area thresholds.",
    )
    parser.add_argument(
        "--filter-imgsz",
        type=int,
        default=640,
        help="Input image size used to convert raw bbox/polygon sizes to training-scale pixels.",
    )
    parser.add_argument(
        "--min-resized-side",
        type=float,
        default=4.0,
        help="Drop instances whose shortest bbox side is smaller than this at --filter-imgsz.",
    )
    parser.add_argument(
        "--min-resized-area",
        type=float,
        default=16.0,
        help="Drop instances whose bbox area is smaller than this at --filter-imgsz.",
    )
    parser.add_argument(
        "--min-resized-mask-area",
        type=float,
        default=10.0,
        help="Drop instances whose polygon area is smaller than this at --filter-imgsz.",
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


def polygon_area(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(polygon):
        next_point = polygon[(idx + 1) % len(polygon)]
        area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return abs(area) / 2.0


def resized_scale(width: int, height: int, imgsz: int) -> float:
    return float(imgsz) / float(max(width, height))


def filter_reason(
    box: tuple[float, float, float, float],
    polygon: list[list[float]],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> str | None:
    if not args.detectable_filter:
        return None
    scale = resized_scale(width, height, args.filter_imgsz)
    resized_w = box[2] * float(width) * scale
    resized_h = box[3] * float(height) * scale
    resized_bbox_area = resized_w * resized_h
    resized_polygon_area = polygon_area(polygon) * scale * scale

    if min(resized_w, resized_h) < args.min_resized_side:
        return "min_side"
    if resized_bbox_area < args.min_resized_area:
        return "bbox_area"
    if resized_polygon_area < args.min_resized_mask_area:
        return "mask_area"
    return None


def parse_yolo_lines(
    json_path: Path,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[list[str], Counter[str], dict[str, object]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    counts: Counter[str] = Counter()
    filter_stats: dict[str, object] = {
        "kept": 0,
        "removed": 0,
        "removed_by_reason": Counter(),
        "removed_by_class": Counter(),
    }
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
        class_name = CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC[class_id]
        line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        if line in seen:
            continue
        seen.add(line)
        reason = filter_reason(box, polygon, width, height, args)
        if reason is not None:
            filter_stats["removed"] += 1
            filter_stats["removed_by_reason"][reason] += 1
            filter_stats["removed_by_class"][class_name] += 1
            continue
        lines.append(line)
        counts[class_name] += 1
        filter_stats["kept"] += 1
    filter_stats["removed_by_reason"] = dict(filter_stats["removed_by_reason"])
    filter_stats["removed_by_class"] = dict(filter_stats["removed_by_class"])
    return lines, counts, filter_stats


def ensure_dirs(root: Path, splits: list[str]) -> None:
    for split in splits:
        (root / "visible" / split).mkdir(parents=True, exist_ok=True)
        (root / "nir" / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def link_or_copy_image(src: Path, dst: Path, image_mode: str) -> bool:
    if image_mode == "skip":
        return False
    if image_mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return True
        except OSError:
            shutil.copy2(src, dst)
            return True
    shutil.copy2(src, dst)
    return True


def merge_filter_stats(dst: dict[str, object], src: dict[str, object]) -> None:
    dst["kept"] += int(src["kept"])
    dst["removed"] += int(src["removed"])
    dst["removed_by_reason"].update(src["removed_by_reason"])
    dst["removed_by_class"].update(src["removed_by_class"])


def export_split(
    source_root: Path,
    output_root: Path,
    split: str,
    weathers: list[str],
    image_mode: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    split_stats: dict[str, object] = {
        "samples": 0,
        "images": 0,
        "labels": 0,
        "annotations": 0,
        "class_counts": Counter(),
        "weather_counts": Counter(),
        "pairing_warnings": [],
        "detectable_filter": {
            "kept": 0,
            "removed": 0,
            "removed_by_reason": Counter(),
            "removed_by_class": Counter(),
        },
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
                if link_or_copy_image(vis_src, visible_split_dir / image_name, image_mode):
                    link_or_copy_image(nir_src, nir_split_dir / image_name, image_mode)
                    split_stats["images"] += 2

                lines, counts, filter_stats = parse_yolo_lines(ann_src, width, height, args)
                label_text = "\n".join(lines)
                (visible_split_dir / label_name).write_text(label_text, encoding="utf-8")
                (nir_split_dir / label_name).write_text(label_text, encoding="utf-8")
                split_stats["labels"] += 2
                split_stats["samples"] += 1
                split_stats["annotations"] += len(lines)
                split_stats["class_counts"].update(counts)
                split_stats["weather_counts"][weather.lower()] += 1
                merge_filter_stats(split_stats["detectable_filter"], filter_stats)

    split_stats["class_counts"] = dict(split_stats["class_counts"])
    split_stats["weather_counts"] = dict(split_stats["weather_counts"])
    split_stats["detectable_filter"]["removed_by_reason"] = dict(
        split_stats["detectable_filter"]["removed_by_reason"]
    )
    split_stats["detectable_filter"]["removed_by_class"] = dict(
        split_stats["detectable_filter"]["removed_by_class"]
    )
    return split_stats


def write_dataset_info(output_root: Path, report: dict[str, object], args: argparse.Namespace) -> None:
    lines = [
        "# IDD-AW RGB-NIR 8 类 personmerge + traffic 可检测目标数据集说明",
        "",
        "## 数据集口径",
        "",
        "本数据集由 IDD-AW 原始语义/实例标注转换为 YOLO 检测标签，并保留 8 类目标：",
        "",
        "| class id | class name |",
        "| --- | --- |",
    ]
    for idx, name in enumerate(CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC):
        lines.append(f"| {idx} | {name} |")
    lines.extend(
        [
            "",
            "`rider` 合并到 `person`。`traffic light` 与 `traffic sign` 作为独立类别保留。",
            "",
            "## 可检测目标过滤规则",
            "",
            "该版本用于减少由分割标注转检测框时产生的极端不可检测实例。过滤在 train/val 上使用同一规则，避免训练和评估口径不一致。",
            "",
            f"- 参考输入尺度：`{args.filter_imgsz}`。",
            f"- 若 resize 后 bbox 最短边 `< {args.min_resized_side:g}px`，则剔除。",
            f"- 若 resize 后 bbox 面积 `< {args.min_resized_area:g}px^2`，则剔除。",
            f"- 若 resize 后 polygon 面积 `< {args.min_resized_mask_area:g}px^2`，则剔除。",
            "",
            "该规则只过滤极端不可检测目标，不等同于删除全部小目标。",
            "",
            "## 数据规模与类别统计",
            "",
            "| split | paired samples | annotations kept | annotations removed |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for split, stats in report["splits"].items():
        filt = stats["detectable_filter"]
        lines.append(
            f"| {split} | {stats['samples']} | {stats['annotations']} | {filt['removed']} |"
        )
    lines.extend(["", "### 训练集标注数量", "", "| class name | count |", "| --- | ---: |"])
    for name in CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC:
        lines.append(f"| {name} | {report['splits']['train']['class_counts'].get(name, 0)} |")
    lines.extend(["", "### 验证集标注数量", "", "| class name | count |", "| --- | ---: |"])
    for name in CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC:
        lines.append(f"| {name} | {report['splits']['val']['class_counts'].get(name, 0)} |")
    lines.extend(["", "### 剔除实例统计", "", "| split | reason/class | count |", "| --- | --- | ---: |"])
    for split, stats in report["splits"].items():
        filt = stats["detectable_filter"]
        for reason, count in sorted(filt["removed_by_reason"].items()):
            lines.append(f"| {split} | reason: {reason} | {count} |")
        for class_name, count in sorted(filt["removed_by_class"].items()):
            lines.append(f"| {split} | class: {class_name} | {count} |")
    lines.extend(
        [
            "",
            "## 生成命令",
            "",
            "```bash",
            "python tools/export_iddaw_raw_to_yolo_rgbnir.py \\",
            f"  --source-root \"{args.source_root}\" \\",
            f"  --output-root \"{args.output_root}\" \\",
            "  --image-mode hardlink \\",
            "  --detectable-filter \\",
            f"  --filter-imgsz {args.filter_imgsz} \\",
            f"  --min-resized-side {args.min_resized_side:g} \\",
            f"  --min-resized-area {args.min_resized_area:g} \\",
            f"  --min-resized-mask-area {args.min_resized_mask_area:g} \\",
            "  --clean",
            "```",
            "",
            "完整导出统计见 `meta/export_report.json`。",
            "",
        ]
    )
    (output_root / "DATASET_INFO.md").write_text("\n".join(lines), encoding="utf-8")


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
        split: export_split(source_root, output_root, split, weathers, args.image_mode, args)
        for split in splits
    }
    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "schema": "8cls_personmerge_traffic",
        "categories": CATEGORY_NAMES_8_PERSONMERGE_TRAFFIC,
        "raw_category_mapping": CATEGORY_TO_ID,
        "image_mode": args.image_mode,
        "detectable_filter": {
            "enabled": args.detectable_filter,
            "filter_imgsz": args.filter_imgsz,
            "min_resized_side": args.min_resized_side,
            "min_resized_area": args.min_resized_area,
            "min_resized_mask_area": args.min_resized_mask_area,
        },
        "splits": split_reports,
    }
    (output_root / "meta" / "export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.detectable_filter:
        write_dataset_info(output_root, report, args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
