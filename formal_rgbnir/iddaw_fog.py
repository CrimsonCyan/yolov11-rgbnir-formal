from __future__ import annotations

import os
from pathlib import Path


CATEGORY_NAMES = ["person", "rider", "motorcycle", "car"]
DEFAULT_PAIRS = ["visible", "infrared"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_root() -> Path:
    env_root = os.getenv("IDDAW_FOG_YOLO_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"IDDAW_FOG_YOLO_ROOT does not exist: {root}")
        return root

    candidates = [
        repo_root().parent / "datasets" / "iddaw_fog_yolov11_rgbnir",
        repo_root() / "datasets" / "iddaw_fog_yolov11_rgbnir",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Unable to locate iddaw_fog_yolov11_rgbnir dataset root.\n"
        "Set IDDAW_FOG_YOLO_ROOT or place the dataset under one of:\n"
        f"{searched}"
    )


def build_dataset_yaml(mode: str) -> Path:
    dataset_root = resolve_dataset_root()
    if mode == "rgb":
        train = "visible/train"
        val = "visible/val"
    elif mode == "nir":
        train = "infrared/train"
        val = "infrared/val"
    elif mode == "rgbnir":
        train = "visible/train"
        val = "visible/val"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    runtime_dir = repo_root() / "runtime_cfg"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_dir / f"iddaw_fog_{mode}.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                f"train: {train}",
                f"val: {val}",
                "nc: 4",
                'names: ["person", "rider", "motorcycle", "car"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return runtime_yaml


def experiment_name(mode: str) -> str:
    names = {
        "rgb": "iddaw-fog-yolo11n-rgb",
        "nir": "iddaw-fog-yolo11n-nir",
        "rgbnir": "iddaw-fog-yolo11n-rgbnir-plain",
    }
    if mode not in names:
        raise ValueError(f"Unsupported mode: {mode}")
    return names[mode]


def model_config_for(mode: str) -> str:
    root = repo_root()
    if mode == "rgb":
        return str((root / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml").resolve())
    if mode == "nir":
        return str((root / "ultralytics" / "cfg" / "models" / "11" / "yolo11-gray.yaml").resolve())
    if mode == "rgbnir":
        return str((root / "configs" / "models" / "yolo11_rgbnir_midfusion_plain.yaml").resolve())
    raise ValueError(f"Unsupported mode: {mode}")


def mode_specific_kwargs(mode: str) -> dict[str, object]:
    if mode == "rgb":
        return {"use_simotm": "BGR", "channels": 3}
    if mode == "nir":
        return {"use_simotm": "Gray", "channels": 1}
    if mode == "rgbnir":
        return {"use_simotm": "RGBT", "channels": 4, "pairs_rgb_ir": DEFAULT_PAIRS}
    raise ValueError(f"Unsupported mode: {mode}")


def common_train_kwargs(mode: str, epochs: int = 50, device: str = "0") -> dict[str, object]:
    return {
        "cache": False,
        "imgsz": 640,
        "epochs": epochs,
        "batch": 8,
        "close_mosaic": 5,
        "workers": 2,
        "device": device,
        "optimizer": "SGD",
        "project": "runs/IDD_AW_FOG",
        "name": experiment_name(mode),
    }


def common_val_kwargs(mode: str) -> dict[str, object]:
    return {
        "split": "val",
        "imgsz": 640,
        "batch": 8,
        "project": "runs/IDD_AW_FOG_VAL",
        "name": experiment_name(mode),
    }


def common_predict_kwargs(mode: str) -> dict[str, object]:
    dataset_root = resolve_dataset_root()
    source_subdir = "visible/val" if mode in {"rgb", "rgbnir"} else "infrared/val"
    return {
        "source": str((dataset_root / source_subdir).resolve()),
        "imgsz": 640,
        "project": "runs/IDD_AW_FOG_PRED",
        "name": experiment_name(mode),
        "save": True,
    }
