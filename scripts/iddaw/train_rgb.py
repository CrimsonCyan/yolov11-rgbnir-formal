from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from formal_rgbnir.iddaw import build_dataset_yaml, common_train_kwargs, mode_specific_kwargs, model_config_for


if __name__ == "__main__":
    mode = "rgb"
    model = YOLO(model_config_for(mode))
    model.train(
        data=str(build_dataset_yaml(mode)),
        **common_train_kwargs(mode),
        **mode_specific_kwargs(mode),
    )
