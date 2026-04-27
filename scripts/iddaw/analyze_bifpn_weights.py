from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.conv import WeightedFeatureFusion


EDGE_LABELS = {
    "p4_td_fuse": ["P4_in", "P5_up"],
    "p3_td_fuse": ["P3_in", "P4_td_up"],
    "p2_out_fuse": ["P2_in", "P3_td_up"],
    "p3_out_fuse": ["P3_in", "P3_td", "P2_out_down"],
    "p4_out_fuse": ["P4_in", "P4_td", "P3_out_down"],
    "p5_out_fuse": ["P5_in", "P4_out_down"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learned BiFPN edge weights from a YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to best.pt/last.pt.")
    parser.add_argument(
        "--out",
        default="runs/analysis/bifpn_weights",
        help="Output directory for bifpn_weights.json/csv/png.",
    )
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation.")
    parser.add_argument("--wandb", action="store_true", help="Log the analysis table, plot, and files to W&B.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "iddaw-rgbnir-formal"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_GROUP", "analysis"))
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-tags", default="bifpn-weights,analysis")
    parser.add_argument("--wandb-artifact-name", default="")
    return parser.parse_args()


def normalized_weights(module: WeightedFeatureFusion) -> tuple[list[float], list[float], list[float]]:
    raw = module.weights.detach().float().cpu()
    relu = torch.relu(raw)
    norm = relu / (relu.sum() + module.eps)
    return raw.tolist(), relu.tolist(), norm.tolist()


def module_parts(name: str) -> tuple[int | None, str]:
    match = re.search(r"blocks\.(\d+)\.([^.]+)$", name)
    if match:
        return int(match.group(1)), match.group(2)
    return None, name.rsplit(".", 1)[-1]


def collect_weights(weights_path: str) -> list[dict[str, object]]:
    model = YOLO(weights_path).model
    rows: list[dict[str, object]] = []
    for name, module in model.named_modules():
        if not isinstance(module, WeightedFeatureFusion):
            continue
        block, node = module_parts(name)
        raw, relu, norm = normalized_weights(module)
        labels = EDGE_LABELS.get(node, [f"input_{idx}" for idx in range(len(norm))])
        for idx, value in enumerate(norm):
            rows.append(
                {
                    "module": name,
                    "block": block,
                    "node": node,
                    "edge_index": idx,
                    "edge_label": labels[idx] if idx < len(labels) else f"input_{idx}",
                    "raw_weight": raw[idx],
                    "relu_weight": relu[idx],
                    "normalized_weight": value,
                }
            )
    return rows


def write_json(path: Path, rows: list[dict[str, object]], weights_path: str) -> None:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row["module"])
        node = grouped.setdefault(
            key,
            {
                "module": row["module"],
                "block": row["block"],
                "node": row["node"],
                "edges": [],
            },
        )
        node["edges"].append(
            {
                "edge_index": row["edge_index"],
                "edge_label": row["edge_label"],
                "raw_weight": row["raw_weight"],
                "relu_weight": row["relu_weight"],
                "normalized_weight": row["normalized_weight"],
            }
        )
    payload = {
        "weights": str(weights_path),
        "num_fusion_edges": len(rows),
        "num_fusion_nodes": len(grouped),
        "fusion_nodes": list(grouped.values()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "module",
        "block",
        "node",
        "edge_index",
        "edge_label",
        "raw_weight",
        "relu_weight",
        "normalized_weight",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, object]], title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [
        f"b{row['block']}:{row['node']}:{row['edge_label']}"
        if row["block"] is not None
        else f"{row['node']}:{row['edge_label']}"
        for row in rows
    ]
    values = [float(row["normalized_weight"]) for row in rows]
    height = max(6, min(18, len(labels) * 0.32))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(range(len(values)), values, color="#2f6f73")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized BiFPN fusion weight")
    ax.set_xlim(0, max(1.0, max(values, default=1.0) * 1.1))
    ax.set_title(title or "Learned BiFPN Edge Weights")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def safe_wandb_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "bifpn-weights"


def log_to_wandb(args: argparse.Namespace, out_dir: Path, rows: list[dict[str, object]], plot_written: bool) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--wandb was set, but the 'wandb' package is not installed.") from exc

    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=args.wandb_run_name or None,
        tags=tags,
        job_type="bifpn_weight_analysis",
        config={
            "weights": args.weights,
            "fusion_nodes": len({row["module"] for row in rows}),
            "fusion_edges": len(rows),
        },
    )

    columns = [
        "module",
        "block",
        "node",
        "edge_index",
        "edge_label",
        "raw_weight",
        "relu_weight",
        "normalized_weight",
    ]
    table = wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in rows])
    log_payload = {
        "bifpn_weight_table": table,
        "fusion_nodes": len({row["module"] for row in rows}),
        "fusion_edges": len(rows),
    }
    if plot_written:
        log_payload["bifpn_weights_plot"] = wandb.Image(str(out_dir / "bifpn_weights.png"))
    wandb.log(log_payload)

    artifact_name = args.wandb_artifact_name or f"bifpn-weights-{Path(args.weights).parent.parent.name}"
    artifact = wandb.Artifact(safe_wandb_name(artifact_name), type="analysis")
    for filename in ("bifpn_weights.json", "bifpn_weights.csv", "bifpn_weights.png"):
        path = out_dir / filename
        if path.exists():
            artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_weights(args.weights)
    if not rows:
        raise RuntimeError(f"No WeightedFeatureFusion modules found in {args.weights}")

    write_json(out_dir / "bifpn_weights.json", rows, args.weights)
    write_csv(out_dir / "bifpn_weights.csv", rows)
    plot_written = False if args.no_plots else write_plot(out_dir / "bifpn_weights.png", rows, args.title)

    print(f"fusion_nodes={len({row['module'] for row in rows})}")
    print(f"fusion_edges={len(rows)}")
    print(f"json={out_dir / 'bifpn_weights.json'}")
    print(f"csv={out_dir / 'bifpn_weights.csv'}")
    if plot_written:
        print(f"plot={out_dir / 'bifpn_weights.png'}")
    elif not args.no_plots:
        print("plot=skipped (matplotlib is not installed)")
    if args.wandb:
        log_to_wandb(args, out_dir, rows, plot_written)
        print("wandb=logged")


if __name__ == "__main__":
    main()
