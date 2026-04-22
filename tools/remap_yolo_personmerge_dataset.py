from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


SOURCE_NAMES = ["person", "rider", "motorcycle", "car", "truck", "bus", "autorickshaw"]
TARGET_NAMES = ["person", "motorcycle", "car", "truck", "bus", "autorickshaw"]
CLASS_MAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
SPLITS = ("train", "val")
MODALITIES = ("visible", "nir")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for modality in MODALITIES:
        for split in SPLITS:
            (root / modality / split).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def clone_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label_file(src: Path, dst: Path) -> tuple[int, int]:
    content = src.read_text(encoding="utf-8").strip()
    if not content:
        dst.write_text("", encoding="utf-8")
        return 0, 0
    lines = []
    rider_lines = 0
    for raw in content.splitlines():
        parts = raw.split()
        cls_id = int(parts[0])
        if cls_id == 1:
            rider_lines += 1
        parts[0] = str(CLASS_MAP[cls_id])
        lines.append(" ".join(parts))
    dst.write_text("\n".join(lines), encoding="utf-8")
    return len(lines), rider_lines


def copy_meta(source_root: Path, output_root: Path, report: dict[str, object]) -> None:
    meta_dir = output_root / "meta"
    source_report = source_root / "meta" / "export_report.json"
    if source_report.exists():
        payload = json.loads(source_report.read_text(encoding="utf-8"))
        payload["derived_from"] = str(source_root)
        payload["categories"] = TARGET_NAMES
        payload["merge_rider_into_person"] = True
        payload["category_mapping"] = CLASS_MAP
        payload["source_categories"] = SOURCE_NAMES
        payload["remap_report"] = report
        (meta_dir / "export_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        (meta_dir / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dirs(output_root)

    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "merge_rider_into_person": True,
        "source_categories": SOURCE_NAMES,
        "categories": TARGET_NAMES,
        "category_mapping": CLASS_MAP,
        "splits": {},
    }

    for split in SPLITS:
        split_stats = {"images": 0, "labels": 0, "samples": 0, "rider_labels_merged": 0}
        stems: set[str] = set()
        for modality in MODALITIES:
            src_dir = source_root / modality / split
            dst_dir = output_root / modality / split
            for image_path in sorted(src_dir.glob("*.png")):
                clone_image(image_path, dst_dir / image_path.name)
                split_stats["images"] += 1
                stems.add(image_path.stem)
            for label_path in sorted(src_dir.glob("*.txt")):
                line_count, rider_lines = remap_label_file(label_path, dst_dir / label_path.name)
                split_stats["labels"] += 1
                split_stats["rider_labels_merged"] += rider_lines
                stems.add(label_path.stem)
        split_stats["samples"] = len(stems)
        report["splits"][split] = split_stats

    copy_meta(source_root, output_root, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
