from __future__ import annotations

import os
from pathlib import Path


CATEGORY_NAMES = ["person", "rider", "motorcycle", "car", "truck", "bus", "autorickshaw"]
DEFAULT_PAIRS = ["visible", "nir"]
TRAINABLE_MODES = {"rgb", "nir", "rgbnir", "input_fusion", "light_gate", "bifpn_only"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_root() -> Path:
    env_root = os.getenv("IDDAW_YOLO_ROOT") or os.getenv("IDDAW_FOG_YOLO_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"IDDAW YOLO dataset root does not exist: {root}")
        return root

    candidates = [
        repo_root().parent / "datasets" / "iddaw_all_weather_full_yolov11_rgbnir",
        repo_root().parent / "datasets" / "iddaw_fog_full_yolov11_rgbnir",
        repo_root().parent / "datasets" / "iddaw_fog_yolov11_rgbnir",
        repo_root() / "datasets" / "iddaw_all_weather_full_yolov11_rgbnir",
        repo_root() / "datasets" / "iddaw_fog_full_yolov11_rgbnir",
        repo_root() / "datasets" / "iddaw_fog_yolov11_rgbnir",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Unable to locate an IDD-AW YOLO RGBNIR dataset root.\n"
        "Set IDDAW_YOLO_ROOT or place the dataset under one of:\n"
        f"{searched}"
    )


def build_dataset_yaml(mode: str) -> Path:
    dataset_root = resolve_dataset_root()
    if mode == "rgb":
        train = "visible/train"
        val = "visible/val"
    elif mode == "nir":
        train = "nir/train"
        val = "nir/val"
    elif mode in {"rgbnir", "input_fusion", "light_gate", "bifpn_only", "decision_fusion"}:
        train = "visible/train"
        val = "visible/val"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    runtime_dir = repo_root() / "runtime_cfg"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_dir / f"iddaw_{mode}.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                f"train: {train}",
                f"val: {val}",
                "nc: 7",
                'names: ["person", "rider", "motorcycle", "car", "truck", "bus", "autorickshaw"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return runtime_yaml


def experiment_name(mode: str) -> str:
    names = {
        "rgb": "iddaw-yolo11n-rgb",
        "nir": "iddaw-yolo11n-nir",
        "rgbnir": "iddaw-yolo11n-rgbnir-plain",
        "input_fusion": "iddaw-yolo11n-input-fusion",
        "light_gate": "iddaw-yolo11n-rgbnir-light-gate",
        "bifpn_only": "iddaw-yolo11n-rgbnir-bifpn-only",
        "decision_fusion": "iddaw-yolo11n-decision-fusion",
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
        return str((root / "configs" / "models" / "yolo11n_rgbnir_midfusion_plain.yaml").resolve())
    if mode == "input_fusion":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_input_fusion.yaml").resolve())
    if mode == "light_gate":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_midfusion_gate.yaml").resolve())
    if mode == "bifpn_only":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_bifpn_only.yaml").resolve())
    raise ValueError(f"Unsupported mode: {mode}")


def mode_specific_kwargs(mode: str) -> dict[str, object]:
    if mode == "rgb":
        return {"use_simotm": "BGR", "channels": 3}
    if mode == "nir":
        return {"use_simotm": "Gray", "channels": 1}
    if mode in {"rgbnir", "input_fusion", "light_gate", "bifpn_only", "decision_fusion"}:
        return {"use_simotm": "RGBNIR", "channels": 4, "pairs_rgb_ir": DEFAULT_PAIRS}
    raise ValueError(f"Unsupported mode: {mode}")


def train_batch_for(mode: str) -> int:
    batches = {
        "rgb": 96,
        "nir": 96,
        "rgbnir": 48,
        "input_fusion": 96,
        "light_gate": 48,
        "bifpn_only": 48,
    }
    if mode not in batches:
        raise ValueError(f"Unsupported mode: {mode}")
    return batches[mode]


def workers_for(mode: str) -> int:
    workers = {
        "rgb": 12,
        "nir": 12,
        "rgbnir": 10,
        "input_fusion": 12,
        "light_gate": 10,
        "bifpn_only": 10,
    }
    if mode not in workers:
        raise ValueError(f"Unsupported mode: {mode}")
    return workers[mode]


def common_train_kwargs(mode: str, epochs: int = 50, device: str = "0") -> dict[str, object]:
    if mode not in TRAINABLE_MODES:
        raise ValueError(f"Mode does not support training: {mode}")
    return {
        "cache": "ram",
        "imgsz": 640,
        "epochs": epochs,
        "batch": train_batch_for(mode),
        "close_mosaic": 5,
        "workers": workers_for(mode),
        "device": device,
        "optimizer": "SGD",
        "project": "runs/IDD_AW",
        "name": experiment_name(mode),
    }


def common_val_kwargs(mode: str) -> dict[str, object]:
    batch = 16 if mode == "decision_fusion" else train_batch_for(mode)
    return {
        "split": "val",
        "imgsz": 640,
        "batch": batch,
        "project": "runs/IDD_AW_VAL",
        "name": experiment_name(mode),
    }


def common_predict_kwargs(mode: str) -> dict[str, object]:
    dataset_root = resolve_dataset_root()
    source_subdir = "visible/val" if mode in {"rgb", "rgbnir", "input_fusion", "light_gate", "bifpn_only", "decision_fusion"} else "nir/val"
    return {
        "source": str((dataset_root / source_subdir).resolve()),
        "imgsz": 640,
        "project": "runs/IDD_AW_PRED",
        "name": experiment_name(mode),
        "save": True,
    }


def experiment_project_dir() -> Path:
    return repo_root() / "runs" / "IDD_AW"


def latest_run_dir(mode: str) -> Path:
    prefix = experiment_name(mode)
    project_dir = experiment_project_dir()
    candidates = [path for path in project_dir.glob(f"{prefix}*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory found for mode '{mode}' under {project_dir}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def latest_weights_for(mode: str, weight_name: str = "best.pt") -> Path:
    weight_path = latest_run_dir(mode) / "weights" / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    return weight_path
