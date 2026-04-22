from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR, YOLO
from ultralytics.utils import SETTINGS

from formal_rgbnir.decision_fusion import run_decision_fusion, save_decision_fusion_outputs
from formal_rgbnir.iddaw_fog import (
    build_dataset_yaml,
    common_predict_kwargs,
    common_train_kwargs,
    common_val_kwargs,
    experiment_name,
    experiment_project_dir,
    mode_specific_kwargs,
    model_config_for,
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_wandb(mode: str) -> None:
    enabled = env_flag("WANDB_ENABLED", default=False)
    SETTINGS.update({"wandb": enabled})
    if not enabled:
        return

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "WANDB_ENABLED=1 but the 'wandb' package is not installed in the current environment."
        ) from exc

    os.environ.setdefault("WANDB_PROJECT", "iddaw-rgbnir-formal")
    os.environ.setdefault("WANDB_GROUP", "iddaw_all_weather")
    dataset_tag = "6-class-personmerge" if mode.endswith("_6cls_personmerge") else "7-class"
    os.environ.setdefault("WANDB_TAGS", f"{mode},all-weather,{dataset_tag}")


def completed_epochs_from_checkpoint(checkpoint_path: str) -> int:
    run_dir = Path(checkpoint_path).resolve().parents[1]
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8") as fh:
            line_count = sum(1 for _ in fh)
        return max(line_count - 1, 0)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "rgb",
            "rgb_yolo11s",
            "rgb_yolo11s_6cls_personmerge",
            "rgb_rtdetr",
            "nir",
            "rgbnir",
            "input_fusion",
            "light_gate",
            "bifpn_only",
            "bifpn_only_yolo11s",
            "bifpn_only_yolo11s_6cls_personmerge",
            "attention_only",
            "full_proposed",
            "full_proposed_residual",
            "full_proposed_residual_v2",
            "full_proposed_residual_v2_yolo11s",
            "full_proposed_residual_v2_yolo11s_6cls_personmerge",
            "decision_fusion",
        ],
        required=True,
    )
    parser.add_argument("--task", choices=["train", "val", "predict"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=1, help="Run validation every N epochs during training.")
    parser.add_argument("--resume", default="", help="Checkpoint path to continue training from.")
    parser.add_argument("--weights", default="", help="Checkpoint path for val/predict.")
    parser.add_argument("--rgb-weights", default="", help="RGB checkpoint for decision fusion.")
    parser.add_argument("--nir-weights", default="", help="NIR checkpoint for decision fusion.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split for decision fusion.")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "decision_fusion":
        if args.task == "train":
            raise ValueError("decision_fusion is an offline baseline and does not support training")
        payload = run_decision_fusion(
            split=args.split,
            device=args.device,
            rgb_weights=args.rgb_weights or None,
            nir_weights=args.nir_weights or None,
        )
        output_dir = experiment_project_dir() / experiment_name(args.mode)
        save_decision_fusion_outputs(output_dir, payload)
        print(payload["metrics"])
        return

    data_yaml = str(build_dataset_yaml(args.mode))
    mode_kwargs = mode_specific_kwargs(args.mode)
    model_cls = RTDETR if args.mode == "rgb_rtdetr" else YOLO

    if args.task == "train":
        configure_wandb(args.mode)
        if args.resume:
            completed_epochs = completed_epochs_from_checkpoint(args.resume)
            extra_epochs = args.epochs - completed_epochs
            if extra_epochs <= 0:
                raise ValueError(
                    f"Checkpoint already covers {completed_epochs} epochs, target total {args.epochs} is not larger"
                )
            print(
                f"Continuing from checkpoint {args.resume}: completed_epochs={completed_epochs}, "
                f"extra_epochs={extra_epochs}, target_total_epochs={args.epochs}"
            )
            model = model_cls(args.resume)
            model.train(
                data=data_yaml,
                **common_train_kwargs(args.mode, extra_epochs, args.device, args.val_interval),
                **mode_kwargs,
            )
            return

        model = model_cls(model_config_for(args.mode))
        model.train(
            data=data_yaml,
            **common_train_kwargs(args.mode, args.epochs, args.device, args.val_interval),
            **mode_kwargs,
        )
        return

    if not args.weights:
        raise ValueError("--weights is required for val and predict")

    model = model_cls(args.weights)
    if args.task == "val":
        model.val(data=data_yaml, **common_val_kwargs(args.mode), **mode_kwargs)
        return

    model.predict(**common_predict_kwargs(args.mode), **mode_kwargs)


if __name__ == "__main__":
    main()
