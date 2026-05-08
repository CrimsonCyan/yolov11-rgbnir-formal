from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_rgbnir.iddaw import (  # noqa: E402
    build_dataset_yaml,
    category_names_for_mode,
    mode_specific_kwargs,
    model_config_for,
    train_batch_for,
)
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether GT boxes receive enough TaskAlignedAssigner positives during YOLO training. "
            "This follows v8DetectionLoss assignment logic and reports class/scale/P-level positive counts."
        )
    )
    parser.add_argument("--mode", required=True, help="Experiment mode registered in formal_rgbnir.iddaw.")
    parser.add_argument("--weights", default="", help="Optional checkpoint. If omitted, the mode YAML is used.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Dataset split to inspect.")
    parser.add_argument("--loader-mode", choices=["train", "val"], default="", help="Override dataloader mode.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=0, help="Batch size. Defaults to the mode training batch.")
    parser.add_argument("--max-batches", type=int, default=25, help="Limit batches for quick diagnostics; 0 = all.")
    parser.add_argument("--device", default="0", help="Device for forward pass, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers for deterministic diagnostics.")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic probability used when loader mode is train.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/analysis/assigner_positives", help="Output directory.")
    parser.add_argument(
        "--small-side-ratio",
        type=float,
        default=0.05,
        help="Small-object equivalent side threshold relative to resized train image max side.",
    )
    parser.add_argument(
        "--medium-side-ratio",
        type=float,
        default=0.15,
        help="Medium-object equivalent side threshold relative to resized train image max side.",
    )
    parser.add_argument(
        "--min-pos-per-gt",
        type=int,
        default=3,
        help="GTs with fewer final positives than this are marked positive-starved.",
    )
    parser.add_argument("--tal-topk", type=int, default=10, help="TaskAlignedAssigner top-k used by YOLO loss.")
    return parser.parse_args()


def normalize_path(path: str | Path) -> str:
    return Path(path).expanduser().resolve().as_posix()


def p_level_name(stride: float) -> str:
    if stride > 0:
        level = math.log2(float(stride))
        if abs(level - round(level)) < 1e-6:
            return f"P{int(round(level))}"
    return f"s{int(stride)}"


def bucket_for_side(equiv_side: float, image_side: float, small_ratio: float, medium_ratio: float) -> str:
    small_thr = image_side * small_ratio
    medium_thr = image_side * medium_ratio
    if equiv_side <= small_thr:
        return "small"
    if equiv_side <= medium_thr:
        return "medium"
    return "large"


def build_trainer(args: argparse.Namespace) -> DetectionTrainer:
    data_yaml = build_dataset_yaml(args.mode)
    model_path = Path(args.weights).expanduser().resolve() if args.weights else Path(model_config_for(args.mode))
    batch = args.batch if args.batch > 0 else train_batch_for(args.mode)
    overrides = {
        "model": normalize_path(model_path),
        "data": normalize_path(data_yaml),
        "imgsz": args.imgsz,
        "batch": batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "cache": False,
        "plots": False,
        "save": False,
        "project": str((Path(args.out) / "_loader").resolve()),
        "name": args.mode,
        "exist_ok": True,
        "epochs": 1,
        "optimizer": "Adam",
        "lr0": 0.01,
        "mosaic": args.mosaic,
        "close_mosaic": 0,
        "pretrained": False,
        **mode_specific_kwargs(args.mode),
    }
    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.model.eval()
    return trainer


def get_feats(preds, nl: int) -> list[torch.Tensor]:
    feats = preds[1] if isinstance(preds, tuple) else preds
    if isinstance(feats, dict):
        feats = feats.get("one2many", next(iter(feats.values())))
    return list(feats[:nl])


def level_slices(feats: list[torch.Tensor], strides: torch.Tensor) -> list[tuple[str, int, int, int]]:
    slices: list[tuple[str, int, int, int]] = []
    start = 0
    for feat, stride in zip(feats, strides.detach().cpu().tolist()):
        count = int(feat.shape[2] * feat.shape[3])
        slices.append((p_level_name(float(stride)), start, start + count, int(stride)))
        start += count
    return slices


def count_by_level(mask: torch.Tensor, slices: list[tuple[str, int, int, int]]) -> dict[str, torch.Tensor]:
    counts = {}
    for level, start, end, _ in slices:
        counts[level] = mask[..., start:end].sum(dim=-1)
    return counts


@torch.no_grad()
def analyze_batch(
    trainer: DetectionTrainer,
    batch: dict,
    criterion: v8DetectionLoss,
    class_names: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[tuple[str, int, int, int]]]:
    batch = trainer.preprocess_batch(batch)
    preds = trainer.model(batch["img"])
    feats = get_feats(preds, criterion.stride.size(0))

    pred_distri, pred_scores = torch.cat(
        [xi.view(feats[0].shape[0], criterion.no, -1) for xi in feats], 2
    ).split((criterion.reg_max * 4, criterion.nc), 1)
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=trainer.device, dtype=dtype) * criterion.stride[0]
    image_side = float(imgsz.max().detach().cpu())
    anchor_points, stride_tensor = make_anchors(feats, criterion.stride, 0.5)

    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(targets.to(trainer.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
    level_info = level_slices(feats, criterion.stride)

    if gt_bboxes.shape[1] == 0 or not mask_gt.any():
        return [], level_info

    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    assigner = criterion.assigner
    if not isinstance(assigner, TaskAlignedAssigner):
        raise TypeError(f"Expected TaskAlignedAssigner, got {type(assigner).__name__}")
    assigner.bs = pred_scores.shape[0]
    assigner.n_max_boxes = gt_bboxes.shape[1]

    anchor_px = anchor_points * stride_tensor
    pred_bboxes_px = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
    scores = pred_scores.detach().sigmoid()
    inside_mask = assigner.select_candidates_in_gts(anchor_px, gt_bboxes).bool() & mask_gt.bool()
    topk_mask, _, overlaps = assigner.get_pos_mask(scores, pred_bboxes_px, gt_labels, gt_bboxes, anchor_px, mask_gt)
    _, _, final_mask = assigner.select_highest_overlaps(topk_mask.clone(), overlaps, assigner.n_max_boxes)

    inside_counts = count_by_level(inside_mask, level_info)
    topk_counts = count_by_level(topk_mask.bool(), level_info)
    final_counts = count_by_level(final_mask.bool(), level_info)
    inside_total = inside_mask.sum(dim=-1)
    topk_total = topk_mask.sum(dim=-1)
    final_total = final_mask.sum(dim=-1)

    im_files = batch.get("im_file", [""] * batch_size)
    records: list[dict[str, object]] = []
    valid = mask_gt.squeeze(-1)
    for bi in range(batch_size):
        for gi in torch.nonzero(valid[bi], as_tuple=False).flatten().tolist():
            label = int(gt_labels[bi, gi, 0].detach().cpu())
            box = gt_bboxes[bi, gi].detach().cpu()
            width = max(float(box[2] - box[0]), 0.0)
            height = max(float(box[3] - box[1]), 0.0)
            area = width * height
            equiv_side = math.sqrt(max(area, 0.0))
            row: dict[str, object] = {
                "image": str(im_files[bi]),
                "class_id": label,
                "class": class_names[label] if 0 <= label < len(class_names) else str(label),
                "bucket": bucket_for_side(equiv_side, image_side, args.small_side_ratio, args.medium_side_ratio),
                "width_px": width,
                "height_px": height,
                "area_px": area,
                "equiv_side_px": equiv_side,
                "inside_total": int(inside_total[bi, gi].detach().cpu()),
                "topk_total": int(topk_total[bi, gi].detach().cpu()),
                "final_total": int(final_total[bi, gi].detach().cpu()),
                "zero_final": int(final_total[bi, gi].item() == 0),
                "starved_final": int(final_total[bi, gi].item() < args.min_pos_per_gt),
            }
            for level, _, _, stride in level_info:
                row[f"stride_{level}"] = stride
                row[f"inside_{level}"] = int(inside_counts[level][bi, gi].detach().cpu())
                row[f"topk_{level}"] = int(topk_counts[level][bi, gi].detach().cpu())
                row[f"final_{level}"] = int(final_counts[level][bi, gi].detach().cpu())
            records.append(row)
    return records, level_info


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(records: list[dict[str, object]], levels: list[str], keys: tuple[str, ...], min_pos: int) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in keys)].append(row)

    out_rows: list[dict[str, object]] = []
    for group_key, rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        totals = [float(row["final_total"]) for row in rows]
        out: dict[str, object] = {key: value for key, value in zip(keys, group_key)}
        out.update(
            {
                "gt": len(rows),
                "avg_inside_total": mean([float(row["inside_total"]) for row in rows]),
                "avg_topk_total": mean([float(row["topk_total"]) for row in rows]),
                "avg_final_total": mean(totals),
                "median_final_total": median(totals) if totals else 0.0,
                "zero_final_gt": sum(int(row["final_total"]) == 0 for row in rows),
                "zero_final_rate": mean([float(int(row["final_total"]) == 0) for row in rows]),
                f"lt{min_pos}_final_gt": sum(int(row["final_total"]) < min_pos for row in rows),
                f"lt{min_pos}_final_rate": mean([float(int(row["final_total"]) < min_pos) for row in rows]),
            }
        )
        for level in levels:
            out[f"avg_inside_{level}"] = mean([float(row[f"inside_{level}"]) for row in rows])
            out[f"avg_topk_{level}"] = mean([float(row[f"topk_{level}"]) for row in rows])
            out[f"avg_final_{level}"] = mean([float(row[f"final_{level}"]) for row in rows])
        out_rows.append(out)
    return out_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    class_names = category_names_for_mode(args.mode)
    trainer = build_trainer(args)
    split_path = trainer.trainset if args.split == "train" else trainer.testset
    loader_mode = args.loader_mode or args.split
    loader = trainer.get_dataloader(split_path, batch_size=trainer.args.batch, rank=-1, mode=loader_mode)
    criterion = v8DetectionLoss(trainer.model, tal_topk=args.tal_topk)

    all_records: list[dict[str, object]] = []
    level_info: list[tuple[str, int, int, int]] = []
    batches_seen = 0
    for batch_i, batch in enumerate(loader):
        if args.max_batches and batch_i >= args.max_batches:
            break
        records, level_info = analyze_batch(trainer, batch, criterion, class_names, args)
        all_records.extend(records)
        batches_seen += 1

    levels = [level for level, _, _, _ in level_info]
    out_dir = Path(args.out) / args.mode
    if args.weights:
        out_dir = out_dir / Path(args.weights).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    by_class = aggregate(all_records, levels, ("class_id", "class"), args.min_pos_per_gt)
    by_class_bucket = aggregate(all_records, levels, ("class_id", "class", "bucket"), args.min_pos_per_gt)
    by_bucket = aggregate(all_records, levels, ("bucket",), args.min_pos_per_gt)
    write_csv(out_dir / "assigner_gt_records.csv", all_records)
    write_csv(out_dir / "assigner_by_class.csv", by_class)
    write_csv(out_dir / "assigner_by_class_bucket.csv", by_class_bucket)
    write_csv(out_dir / "assigner_by_bucket.csv", by_bucket)

    summary = {
        "mode": args.mode,
        "weights": normalize_path(args.weights) if args.weights else "",
        "dataset_yaml": normalize_path(build_dataset_yaml(args.mode)),
        "split": args.split,
        "loader_mode": loader_mode,
        "imgsz": args.imgsz,
        "batch": int(trainer.args.batch),
        "batches_seen": batches_seen,
        "gt_seen": len(all_records),
        "tal_topk": args.tal_topk,
        "min_pos_per_gt": args.min_pos_per_gt,
        "small_side_ratio": args.small_side_ratio,
        "medium_side_ratio": args.medium_side_ratio,
        "levels": [{"name": name, "anchors": end - start, "stride": stride} for name, start, end, stride in level_info],
        "by_class": by_class,
        "by_bucket": by_bucket,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved assigner diagnostics to {out_dir}")
    print(f"batches={batches_seen}, gt={len(all_records)}, levels={levels}")
    for row in by_class:
        if row.get("class") in {"traffic light", "traffic sign", "person", "motorcycle"}:
            print(
                f"{row['class']}: gt={row['gt']}, avg_final={row['avg_final_total']:.2f}, "
                f"zero={row['zero_final_rate']:.3f}, lt{args.min_pos_per_gt}={row[f'lt{args.min_pos_per_gt}_final_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
