from __future__ import annotations

import os
import re
from pathlib import Path


CATEGORY_NAMES_7 = ["person", "rider", "motorcycle", "car", "truck", "bus", "autorickshaw"]
CATEGORY_NAMES_6_PERSONMERGE = ["person", "motorcycle", "car", "truck", "bus", "autorickshaw"]
DEFAULT_PAIRS = ["visible", "nir"]
PERSONMERGE_MODES = {
    "rgb_yolo11s_6cls_personmerge",
    "bifpn_only_yolo11s_6cls_personmerge",
    "full_proposed_residual_v2_yolo11s_6cls_personmerge",
}
TRAINABLE_MODES = {
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
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def category_names_for_mode(mode: str) -> list[str]:
    return CATEGORY_NAMES_6_PERSONMERGE if mode in PERSONMERGE_MODES else CATEGORY_NAMES_7


def resolve_dataset_root(mode: str) -> Path:
    if mode in PERSONMERGE_MODES:
        env_root = os.getenv("IDDAW_YOLO_ROOT_6CLS_PERSONMERGE")
        candidates = [
            repo_root().parent / "datasets" / "iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge",
            repo_root() / "datasets" / "iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge",
        ]
    else:
        env_root = os.getenv("IDDAW_YOLO_ROOT") or os.getenv("IDDAW_FOG_YOLO_ROOT")
        candidates = [
            repo_root().parent / "datasets" / "iddaw_all_weather_full_yolov11_rgbnir",
            repo_root().parent / "datasets" / "iddaw_fog_full_yolov11_rgbnir",
            repo_root().parent / "datasets" / "iddaw_fog_yolov11_rgbnir",
            repo_root() / "datasets" / "iddaw_all_weather_full_yolov11_rgbnir",
            repo_root() / "datasets" / "iddaw_fog_full_yolov11_rgbnir",
            repo_root() / "datasets" / "iddaw_fog_yolov11_rgbnir",
        ]
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"IDDAW YOLO dataset root does not exist: {root}")
        return root
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Unable to locate an IDD-AW YOLO RGBNIR dataset root.\n"
        f"Mode: {mode}\n"
        "Set the matching dataset root env var or place the dataset under one of:\n"
        f"{searched}"
    )


def build_dataset_yaml(mode: str) -> Path:
    dataset_root = resolve_dataset_root(mode)
    if mode in {"rgb", "rgb_yolo11s", "rgb_yolo11s_6cls_personmerge", "rgb_rtdetr"}:
        train = "visible/train"
        val = "visible/val"
    elif mode == "nir":
        train = "nir/train"
        val = "nir/val"
    elif mode in TRAINABLE_MODES | {"decision_fusion"}:
        train = "visible/train"
        val = "visible/val"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    names = category_names_for_mode(mode)
    runtime_dir = repo_root() / "runtime_cfg"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_dir / f"iddaw_{mode}.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                f"train: {train}",
                f"val: {val}",
                f"nc: {len(names)}",
                f"names: {names!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return runtime_yaml


def experiment_name(mode: str) -> str:
    names = {
        "rgb": "iddaw-yolo11n-rgb",
        "rgb_yolo11s": "iddaw-yolo11s-rgb",
        "rgb_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgb-6cls-personmerge",
        "rgb_rtdetr": "iddaw-rtdetr-r18-rgb",
        "nir": "iddaw-yolo11n-nir",
        "rgbnir": "iddaw-yolo11n-rgbnir-plain",
        "input_fusion": "iddaw-yolo11n-input-fusion",
        "light_gate": "iddaw-yolo11n-rgbnir-light-gate",
        "bifpn_only": "iddaw-yolo11n-rgbnir-bifpn-only",
        "bifpn_only_yolo11s": "iddaw-yolo11s-rgbnir-bifpn-only",
        "bifpn_only_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge",
        "attention_only": "iddaw-yolo11n-rgbnir-attention-only",
        "full_proposed": "iddaw-yolo11n-rgbnir-full-proposed",
        "full_proposed_residual": "iddaw-yolo11n-rgbnir-full-proposed-residual",
        "full_proposed_residual_v2": "iddaw-yolo11n-rgbnir-full-proposed-residual-v2",
        "full_proposed_residual_v2_yolo11s": "iddaw-yolo11s-rgbnir-full-proposed-residual-v2",
        "full_proposed_residual_v2_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-full-proposed-residual-v2-6cls-personmerge",
        "decision_fusion": "iddaw-yolo11n-decision-fusion",
    }
    if mode not in names:
        raise ValueError(f"Unsupported mode: {mode}")
    return names[mode]


def model_config_for(mode: str) -> str:
    root = repo_root()
    if mode == "rgb":
        return str((root / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml").resolve())
    if mode in {"rgb_yolo11s", "rgb_yolo11s_6cls_personmerge"}:
        return str((root / "ultralytics" / "cfg" / "models" / "11" / "yolo11s.yaml").resolve())
    if mode == "rgb_rtdetr":
        return str((root / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-r18.yaml").resolve())
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
    if mode == "bifpn_only_yolo11s":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_bifpn_only.yaml").resolve())
    if mode == "bifpn_only_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_bifpn_only_6cls_personmerge.yaml").resolve())
    if mode == "attention_only":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_attention_only.yaml").resolve())
    if mode == "full_proposed":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_full_proposed.yaml").resolve())
    if mode == "full_proposed_residual":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_full_proposed_residual.yaml").resolve())
    if mode == "full_proposed_residual_v2":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_full_proposed_residual_v2.yaml").resolve())
    if mode == "full_proposed_residual_v2_yolo11s":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_full_proposed_residual_v2.yaml").resolve())
    if mode == "full_proposed_residual_v2_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_full_proposed_residual_v2_6cls_personmerge.yaml").resolve())
    raise ValueError(f"Unsupported mode: {mode}")


def mode_specific_kwargs(mode: str) -> dict[str, object]:
    if mode in {"rgb", "rgb_yolo11s", "rgb_yolo11s_6cls_personmerge", "rgb_rtdetr"}:
        return {"use_simotm": "BGR", "channels": 3}
    if mode == "nir":
        return {"use_simotm": "Gray", "channels": 1}
    if mode in (TRAINABLE_MODES - {"rgb", "rgb_yolo11s", "rgb_yolo11s_6cls_personmerge", "rgb_rtdetr", "nir"}) | {"decision_fusion"}:
        return {"use_simotm": "RGBNIR", "channels": 4, "pairs_rgb_ir": DEFAULT_PAIRS}
    raise ValueError(f"Unsupported mode: {mode}")


def train_batch_for(mode: str) -> int:
    batches = {
        "rgb": 96,
        "rgb_yolo11s": 48,
        "rgb_yolo11s_6cls_personmerge": 48,
        "rgb_rtdetr": 32,
        "nir": 96,
        "rgbnir": 48,
        "input_fusion": 96,
        "light_gate": 48,
        "bifpn_only": 48,
        "bifpn_only_yolo11s": 24,
        "bifpn_only_yolo11s_6cls_personmerge": 24,
        "attention_only": 48,
        "full_proposed": 48,
        "full_proposed_residual": 48,
        "full_proposed_residual_v2": 48,
        "full_proposed_residual_v2_yolo11s": 24,
        "full_proposed_residual_v2_yolo11s_6cls_personmerge": 24,
    }
    if mode not in batches:
        raise ValueError(f"Unsupported mode: {mode}")
    return batches[mode]


def workers_for(mode: str) -> int:
    workers = {
        "rgb": 12,
        "rgb_yolo11s": 12,
        "rgb_yolo11s_6cls_personmerge": 12,
        "rgb_rtdetr": 10,
        "nir": 12,
        "rgbnir": 10,
        "input_fusion": 12,
        "light_gate": 10,
        "bifpn_only": 10,
        "bifpn_only_yolo11s": 10,
        "bifpn_only_yolo11s_6cls_personmerge": 10,
        "attention_only": 10,
        "full_proposed": 10,
        "full_proposed_residual": 10,
        "full_proposed_residual_v2": 10,
        "full_proposed_residual_v2_yolo11s": 10,
        "full_proposed_residual_v2_yolo11s_6cls_personmerge": 10,
    }
    if mode not in workers:
        raise ValueError(f"Unsupported mode: {mode}")
    return workers[mode]


def common_train_kwargs(mode: str, epochs: int = 50, device: str = "0", val_interval: int = 1) -> dict[str, object]:
    if mode not in TRAINABLE_MODES:
        raise ValueError(f"Mode does not support training: {mode}")
    return {
        "cache": "ram",
        "imgsz": 640,
        "epochs": epochs,
        "val_interval": max(int(val_interval), 1),
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
    dataset_root = resolve_dataset_root(mode)
    source_subdir = "nir/val" if mode == "nir" else "visible/val"
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
    pattern = re.compile(rf"^{re.escape(prefix)}(?:\d+)?$")
    candidates = [path for path in project_dir.iterdir() if path.is_dir() and pattern.fullmatch(path.name)]
    if not candidates:
        raise FileNotFoundError(f"No run directory found for mode '{mode}' under {project_dir}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def latest_weights_for(mode: str, weight_name: str = "best.pt") -> Path:
    weight_path = latest_run_dir(mode) / "weights" / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    return weight_path
