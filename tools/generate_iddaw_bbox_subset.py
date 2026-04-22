from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


YOLO_ROOT = Path(r"E:\毕设\code\datasets\iddaw_all_weather_full_yolov11_rgbnir")
OUTPUT_ROOT = Path(r"E:\毕设\code\datasets\processed")
SPLIT = "val"
WEATHERS = ("FOG", "LOWLIGHT", "RAIN", "SNOW")
SAMPLES_PER_WEATHER = 25
DEFAULT_CLASS_NAMES = [
    "person",
    "rider",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "autorickshaw",
]
CLASS_COLORS = [
    (255, 99, 71),
    (65, 105, 225),
    (60, 179, 113),
    (255, 165, 0),
    (138, 43, 226),
    (220, 20, 60),
    (0, 191, 255),
]
HEADER_HEIGHT = 30
PANEL_GAP = 10
PADDING = 8


@dataclass
class SampleRecord:
    sample_id: str
    weather: str
    split: str
    visible_image: str
    label_file: str
    annotation_count: int
    output_image: str


def load_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def parse_weather(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected sample stem: {stem}")
    return parts[1].upper()


def list_samples() -> dict[str, list[str]]:
    visible_dir = YOLO_ROOT / "visible" / SPLIT
    if not visible_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {visible_dir}")

    grouped: dict[str, list[str]] = defaultdict(list)
    for image_path in sorted(visible_dir.glob("*.png")):
        weather = parse_weather(image_path.stem)
        if weather in WEATHERS:
            grouped[weather].append(image_path.stem)
    return grouped


def load_class_names() -> list[str]:
    report_path = YOLO_ROOT / "meta" / "export_report.json"
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        categories = payload.get("categories")
        if isinstance(categories, list) and categories:
            return [str(item) for item in categories]
    return DEFAULT_CLASS_NAMES


def yolo_to_xyxy(line: str, width: int, height: int, class_names: list[str]) -> tuple[int, int, int, int, int]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label: {line}")
    cls_id = int(parts[0])
    if not (0 <= cls_id < len(class_names)):
        raise ValueError(f"Class id out of range: {cls_id}")
    cx, cy, bw, bh = (float(v) for v in parts[1:])
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
        raise ValueError(f"YOLO values out of range: {line}")

    x1 = max(0, min(width - 1, int(round((cx - bw / 2.0) * width))))
    y1 = max(0, min(height - 1, int(round((cy - bh / 2.0) * height))))
    x2 = max(0, min(width - 1, int(round((cx + bw / 2.0) * width))))
    y2 = max(0, min(height - 1, int(round((cy + bh / 2.0) * height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox after conversion: {line}")
    return cls_id, x1, y1, x2, y2


def load_boxes(label_path: Path, width: int, height: int, class_names: list[str]) -> list[tuple[int, int, int, int, int]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return [yolo_to_xyxy(line.strip(), width, height, class_names) for line in content.splitlines() if line.strip()]


def draw_boxes(image: Image.Image, boxes: list[tuple[int, int, int, int, int]], font: ImageFont.ImageFont, class_names: list[str]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for cls_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{cls_id}:{class_names[cls_id]}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = max(0, min(canvas.width - text_w - 6, x1))
        text_y = max(0, y1 - text_h - 6)
        draw.rectangle([text_x, text_y, text_x + text_w + 6, text_y + text_h + 4], fill=color)
        draw.text((text_x + 3, text_y + 2), label, fill=(255, 255, 255), font=font)
    return canvas


def build_composite(sample_id: str, weather: str, rgb_path: Path, boxes: list[tuple[int, int, int, int, int]], output_path: Path, font: ImageFont.ImageFont, class_names: list[str]) -> None:
    with Image.open(rgb_path) as rgb_image:
        rgb = rgb_image.convert("RGB")

    boxed_rgb = draw_boxes(rgb, boxes, font, class_names)
    width, height = rgb.size
    composite = Image.new("RGB", (width * 2 + PANEL_GAP + PADDING * 2, height + HEADER_HEIGHT + PADDING * 2), color=(255, 255, 255))
    draw = ImageDraw.Draw(composite)
    header = f"{sample_id} | weather={weather} | annotations={len(boxes)}"
    draw.rectangle([0, 0, composite.width, HEADER_HEIGHT + PADDING], fill=(245, 245, 245))
    draw.text((PADDING, 6), header, fill=(20, 20, 20), font=font)
    draw.text((PADDING, HEADER_HEIGHT - 2), "RGB", fill=(50, 50, 50), font=font)
    draw.text((PADDING + width + PANEL_GAP, HEADER_HEIGHT - 2), "RGB + bbox", fill=(50, 50, 50), font=font)
    composite.paste(rgb, (PADDING, HEADER_HEIGHT + PADDING))
    composite.paste(boxed_rgb, (PADDING + width + PANEL_GAP, HEADER_HEIGHT + PADDING))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output_path)


def clean_output_root() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for weather in WEATHERS:
        weather_dir = OUTPUT_ROOT / weather
        if weather_dir.exists():
            shutil.rmtree(weather_dir)
    for name in ("manifest.json", "summary.json"):
        path = OUTPUT_ROOT / name
        if path.exists():
            path.unlink()


def main() -> None:
    font = load_font()
    class_names = load_class_names()
    grouped = list_samples()
    selected: dict[str, list[str]] = {}
    for weather in WEATHERS:
        stems = grouped.get(weather, [])
        if len(stems) < SAMPLES_PER_WEATHER:
            raise ValueError(f"Not enough {weather} samples in {SPLIT}: {len(stems)} < {SAMPLES_PER_WEATHER}")
        selected[weather] = stems[:SAMPLES_PER_WEATHER]

    clean_output_root()

    manifest: list[dict[str, object]] = []
    summary = {
        "dataset_root": str(YOLO_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "split": SPLIT,
        "samples_per_weather": SAMPLES_PER_WEATHER,
        "render_mode": "left_rgb_right_rgb_with_bbox",
        "total_samples": 0,
        "empty_label_samples": 0,
        "weather_counts": {},
        "average_annotations_per_weather": {},
    }

    visible_dir = YOLO_ROOT / "visible" / SPLIT
    for weather in WEATHERS:
        total_annotations = 0
        empty_labels = 0
        weather_records: list[SampleRecord] = []
        for stem in selected[weather]:
            rgb_path = visible_dir / f"{stem}.png"
            label_path = visible_dir / f"{stem}.txt"
            if not rgb_path.exists():
                raise FileNotFoundError(f"Missing image for {stem}")
            with Image.open(rgb_path) as image:
                width, height = image.size
            boxes = load_boxes(label_path, width, height, class_names)
            output_path = OUTPUT_ROOT / weather / f"{stem}.png"
            build_composite(stem, weather, rgb_path, boxes, output_path, font, class_names)
            total_annotations += len(boxes)
            if not boxes:
                empty_labels += 1
            weather_records.append(SampleRecord(
                sample_id=stem,
                weather=weather,
                split=SPLIT,
                visible_image=str(rgb_path),
                label_file=str(label_path),
                annotation_count=len(boxes),
                output_image=str(output_path),
            ))

        summary["weather_counts"][weather] = len(weather_records)
        summary["average_annotations_per_weather"][weather] = round(total_annotations / len(weather_records), 3)
        summary["total_samples"] += len(weather_records)
        summary["empty_label_samples"] += empty_labels
        manifest.extend(record.__dict__ for record in weather_records)

    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
