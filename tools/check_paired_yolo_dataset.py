from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-root", required=True)
    parser.add_argument("--paired-root", default="", help="Optional paired-json dataset root for exact label cross-check.")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def expected_lines_from_sample(sample: dict[str, object]) -> list[str]:
    width = int(sample["width"])
    height = int(sample["height"])
    lines: list[str] = []
    for annotation in sample["annotations"]:
        x, y, w, h = annotation["bbox"]
        cx = (float(x) + float(w) / 2.0) / float(width)
        cy = (float(y) + float(h) / 2.0) / float(height)
        bw = float(w) / float(width)
        bh = float(h) / float(height)
        category = int(annotation["category_id"]) - 1
        lines.append(f"{category} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def unique_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def parse_label_lines(path: Path, num_classes: int) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    line_count = 0
    empty_count = 0
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return line_count, 1
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label format in {path}: {line}")
        cls_id = int(parts[0])
        if cls_id < 0 or cls_id >= num_classes:
            raise ValueError(f"Class id out of range in {path}: {cls_id}")
        cx, cy, bw, bh = (float(item) for item in parts[1:])
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            raise ValueError(f"Center out of range in {path}: {line}")
        if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            raise ValueError(f"Size out of range in {path}: {line}")
        if cx - bw / 2.0 < -1e-6 or cx + bw / 2.0 > 1.0 + 1e-6:
            raise ValueError(f"X box exceeds image range in {path}: {line}")
        if cy - bh / 2.0 < -1e-6 or cy + bh / 2.0 > 1.0 + 1e-6:
            raise ValueError(f"Y box exceeds image range in {path}: {line}")
        line_count += 1
    return line_count, empty_count


def load_reference_samples(paired_root: Path, split: str) -> dict[str, dict[str, object]]:
    annotation_path = paired_root / "annotations" / f"{split}.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    return {str(sample["sample_id"]): sample for sample in payload["samples"]}


def check_split(yolo_root: Path, split: str, num_classes: int, reference_samples: dict[str, dict[str, object]] | None) -> dict[str, object]:
    visible_dir = yolo_root / "visible" / split
    nir_dir = yolo_root / "nir" / split
    if not visible_dir.exists() or not nir_dir.exists():
        raise FileNotFoundError(f"Missing split directory for {split}")

    visible_images = sorted(path.stem for path in visible_dir.glob("*.png"))
    nir_images = sorted(path.stem for path in nir_dir.glob("*.png"))
    visible_labels = sorted(path.stem for path in visible_dir.glob("*.txt"))
    nir_labels = sorted(path.stem for path in nir_dir.glob("*.txt"))

    if visible_images != nir_images:
        raise ValueError(f"Image pairing mismatch for split={split}")
    if visible_labels != nir_labels:
        raise ValueError(f"Label pairing mismatch for split={split}")
    if visible_images != visible_labels:
        raise ValueError(f"Image/label stem mismatch for split={split}")

    total_annotations = 0
    empty_labels = 0
    reference_duplicates_removed = 0
    for stem in visible_images:
        vis_image = visible_dir / f"{stem}.png"
        nir_image = nir_dir / f"{stem}.png"
        vis_label = visible_dir / f"{stem}.txt"
        nir_label = nir_dir / f"{stem}.txt"

        with Image.open(vis_image) as vis_obj:
            vis_size = vis_obj.size
        with Image.open(nir_image) as nir_obj:
            nir_size = nir_obj.size
        if vis_size != nir_size:
            raise ValueError(f"Image size mismatch for split={split}, stem={stem}: {vis_size} vs {nir_size}")

        vis_text = vis_label.read_text(encoding="utf-8")
        nir_text = nir_label.read_text(encoding="utf-8")
        if vis_text != nir_text:
            raise ValueError(f"Visible/NIR label mismatch for split={split}, stem={stem}")

        ann_count, empty_count = parse_label_lines(vis_label, num_classes)
        total_annotations += ann_count
        empty_labels += empty_count

        if reference_samples is not None:
            if stem not in reference_samples:
                raise ValueError(f"Missing reference sample for split={split}, stem={stem}")
            raw_expected_lines = expected_lines_from_sample(reference_samples[stem])
            expected_lines = unique_preserve_order(raw_expected_lines)
            reference_duplicates_removed += len(raw_expected_lines) - len(expected_lines)
            actual_lines = [line.strip() for line in vis_text.splitlines() if line.strip()]
            if actual_lines != expected_lines:
                raise ValueError(f"Export label mismatch against paired reference for split={split}, stem={stem}")

    if reference_samples is not None and set(visible_images) != set(reference_samples):
        missing = sorted(set(reference_samples) - set(visible_images))
        extra = sorted(set(visible_images) - set(reference_samples))
        raise ValueError(f"Reference mismatch for split={split}: missing={missing[:5]}, extra={extra[:5]}")

    return {
        "samples": len(visible_images),
        "images": len(visible_images) * 2,
        "labels": len(visible_labels) * 2,
        "annotations": total_annotations,
        "empty_label_files": empty_labels,
        "reference_verified": reference_samples is not None,
        "reference_duplicates_removed": reference_duplicates_removed,
    }


def main() -> None:
    args = parse_args()
    yolo_root = Path(args.yolo_root).resolve()
    paired_root = Path(args.paired_root).resolve() if args.paired_root else None

    report = {
        "yolo_root": str(yolo_root),
        "paired_root": str(paired_root) if paired_root else "",
        "num_classes": args.num_classes,
        "splits": {},
    }

    for split in ("train", "val"):
        reference_samples = load_reference_samples(paired_root, split) if paired_root else None
        report["splits"][split] = check_split(yolo_root, split, args.num_classes, reference_samples)

    if args.write_report:
        meta_dir = yolo_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "check_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
