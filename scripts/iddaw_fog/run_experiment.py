from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from formal_rgbnir.iddaw_fog import (
    build_dataset_yaml,
    common_predict_kwargs,
    common_train_kwargs,
    common_val_kwargs,
    mode_specific_kwargs,
    model_config_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rgb", "nir", "rgbnir"], required=True)
    parser.add_argument("--task", choices=["train", "val", "predict"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weights", default="", help="Checkpoint path for val/predict.")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = str(build_dataset_yaml(args.mode))
    mode_kwargs = mode_specific_kwargs(args.mode)

    if args.task == "train":
        model = YOLO(model_config_for(args.mode))
        model.train(data=data_yaml, **common_train_kwargs(args.mode, args.epochs, args.device), **mode_kwargs)
        return

    if not args.weights:
        raise ValueError("--weights is required for val and predict")

    model = YOLO(args.weights)
    if args.task == "val":
        model.val(data=data_yaml, **common_val_kwargs(args.mode), **mode_kwargs)
        return

    model.predict(**common_predict_kwargs(args.mode), **mode_kwargs)


if __name__ == "__main__":
    main()
