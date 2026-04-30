from __future__ import annotations

import os
import re
from pathlib import Path


CATEGORY_NAMES_7 = ["person", "rider", "motorcycle", "car", "truck", "bus", "autorickshaw"]
CATEGORY_NAMES_6_PERSONMERGE = ["person", "motorcycle", "car", "truck", "bus", "autorickshaw"]
CATEGORY_NAMES = CATEGORY_NAMES_6_PERSONMERGE
DEFAULT_PAIRS = ["visible", "nir"]
DEFAULT_CLASS_SCHEMA = "6cls_personmerge"
LEGACY_CLASS_SCHEMA = "7cls"
PERSONMERGE_MODES = {
    "rgb_yolo11s_6cls_personmerge",
    "nir_yolo11s_6cls_personmerge",
    "early_fusion_yolo11s_6cls_personmerge",
    "rgbnir_yolo11s_6cls_personmerge",
    "bifpn_only_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_floor005_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_headp2_smallprior_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_gatefloor_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_twofloor_c256_yolo11s_6cls_personmerge",
    "oa_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge",
    "oa_yolo_pan_light_nir_p2p5_c256_yolo11s_6cls_personmerge",
    "rgbnir_light_nir_yolo11s_6cls_personmerge",
    "full_proposed_residual_v2_yolo11s_6cls_personmerge",
    "proposed_lite_yolo11s_6cls_personmerge",
    "proposed_lite_light_nir_yolo11s_6cls_personmerge",
}
TRAINABLE_MODES = {
    "rgb",
    "rgb_yolo11s",
    "rgb_yolo11s_6cls_personmerge",
    "rgb_rtdetr",
    "nir",
    "nir_yolo11s_6cls_personmerge",
    "early_fusion_yolo11s_6cls_personmerge",
    "rgbnir",
    "rgbnir_yolo11s_6cls_personmerge",
    "input_fusion",
    "light_gate",
    "bifpn_only",
    "bifpn_only_yolo11s",
    "bifpn_only_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_floor005_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_headp2_smallprior_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_gatefloor_c256_yolo11s_6cls_personmerge",
    "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_twofloor_c256_yolo11s_6cls_personmerge",
    "oa_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge",
    "oa_yolo_pan_light_nir_p2p5_c256_yolo11s_6cls_personmerge",
    "rgbnir_light_nir_yolo11s_6cls_personmerge",
    "attention_only",
    "full_proposed",
    "full_proposed_residual",
    "full_proposed_residual_v2",
    "full_proposed_residual_v2_yolo11s",
    "full_proposed_residual_v2_yolo11s_6cls_personmerge",
    "proposed_lite_yolo11s_6cls_personmerge",
    "proposed_lite_light_nir_yolo11s_6cls_personmerge",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def class_schema_for_mode(mode: str) -> str:
    if mode in PERSONMERGE_MODES:
        return DEFAULT_CLASS_SCHEMA
    return os.getenv("IDDAW_CLASS_SCHEMA", DEFAULT_CLASS_SCHEMA).strip().lower()


def use_personmerge_schema(mode: str) -> bool:
    return class_schema_for_mode(mode) != LEGACY_CLASS_SCHEMA


def category_names_for_mode(mode: str) -> list[str]:
    return CATEGORY_NAMES_6_PERSONMERGE if use_personmerge_schema(mode) else CATEGORY_NAMES_7


def resolve_dataset_root(mode: str = "rgbnir") -> Path:
    if use_personmerge_schema(mode):
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
    elif mode in {"nir", "nir_yolo11s_6cls_personmerge"}:
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
        "nir_yolo11s_6cls_personmerge": "iddaw-yolo11s-nir-6cls-personmerge",
        "early_fusion_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-early-fusion-6cls-personmerge",
        "rgbnir": "iddaw-yolo11n-rgbnir-plain",
        "rgbnir_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-plain-6cls-personmerge",
        "input_fusion": "iddaw-yolo11n-input-fusion",
        "light_gate": "iddaw-yolo11n-rgbnir-light-gate",
        "bifpn_only": "iddaw-yolo11n-rgbnir-bifpn-only",
        "bifpn_only_yolo11s": "iddaw-yolo11s-rgbnir-bifpn-only",
        "bifpn_only_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge",
        "bifpn_only_light_nir_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge",
        "bifpn_only_light_nir_p2_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_floor005_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-floor005-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-reflect-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-fg-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-ms-softprior-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-ms-softprior-p2only-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-ms-softprior-resreflect-p2only-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-resreflect-p2only-p3plain-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-smallprior-p2only-p3plain-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_headp2_smallprior_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-headp2-smallprior-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_gatefloor_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-ms-softprior-p2only-gatefloor-c256-6cls-personmerge",
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_twofloor_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-ms-softprior-p2only-twofloor-c256-6cls-personmerge",
        "oa_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-oa-only-light-nir-p2p5-c256-6cls-personmerge",
        "oa_yolo_pan_light_nir_p2p5_c256_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-oa-yolo-pan-light-nir-p2p5-c256-6cls-personmerge",
        "rgbnir_light_nir_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-light-nir-6cls-personmerge",
        "attention_only": "iddaw-yolo11n-rgbnir-attention-only",
        "full_proposed": "iddaw-yolo11n-rgbnir-full-proposed",
        "full_proposed_residual": "iddaw-yolo11n-rgbnir-full-proposed-residual",
        "full_proposed_residual_v2": "iddaw-yolo11n-rgbnir-full-proposed-residual-v2",
        "full_proposed_residual_v2_yolo11s": "iddaw-yolo11s-rgbnir-full-proposed-residual-v2",
        "full_proposed_residual_v2_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-full-proposed-residual-v2-6cls-personmerge",
        "proposed_lite_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-proposed-lite-p34-6cls-personmerge",
        "proposed_lite_light_nir_yolo11s_6cls_personmerge": "iddaw-yolo11s-rgbnir-proposed-lite-light-nir-p34-6cls-personmerge",
        "decision_fusion": "iddaw-yolo11n-decision-fusion",
    }
    if mode not in names:
        raise ValueError(f"Unsupported mode: {mode}")
    name = names[mode]
    if use_personmerge_schema(mode) and "6cls-personmerge" not in name:
        return f"{name}-6cls-personmerge"
    return name


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
    if mode == "nir_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_nir_6cls_personmerge.yaml").resolve())
    if mode == "early_fusion_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_early_fusion_6cls_personmerge.yaml").resolve())
    if mode == "rgbnir":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_midfusion_plain.yaml").resolve())
    if mode == "rgbnir_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_midfusion_plain_6cls_personmerge.yaml").resolve())
    if mode == "input_fusion":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_input_fusion.yaml").resolve())
    if mode == "light_gate":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_midfusion_gate.yaml").resolve())
    if mode == "bifpn_only":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_bifpn_only.yaml").resolve())
    if mode == "bifpn_only_yolo11s":
        config = "yolo11s_rgbnir_bifpn_only_6cls_personmerge.yaml" if use_personmerge_schema(mode) else "yolo11s_rgbnir_bifpn_only.yaml"
        return str((root / "configs" / "models" / config).resolve())
    if mode == "bifpn_only_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_bifpn_only_6cls_personmerge.yaml").resolve())
    if mode == "bifpn_only_light_nir_yolo11s_6cls_personmerge":
        return str(
            (root / "configs" / "models" / "yolo11s_rgbnir_bifpn_only_light_nir_6cls_personmerge.yaml").resolve()
        )
    if mode == "bifpn_only_light_nir_p2_yolo11s_6cls_personmerge":
        return str(
            (root / "configs" / "models" / "yolo11s_rgbnir_bifpn_only_light_nir_p2_6cls_personmerge.yaml").resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge":
        return str(
            (root / "configs" / "models" / "yolo11s_rgbnir_bifpn_p2p5_light_nir_6cls_personmerge.yaml").resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge":
        return str(
            (root / "configs" / "models" / "yolo11s_rgbnir_bifpn_p2p5_light_nir_c256_6cls_personmerge.yaml").resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_floor005_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_floor005_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oagate_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oagate_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_reflect_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_fg_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_ms_softprior_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_ms_softprior_p2only_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_ms_softprior_resreflect_p2only_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_resreflect_p2only_p3plain_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_smallprior_p2only_p3plain_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_headp2_smallprior_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_headp2_smallprior_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_gatefloor_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_ms_softprior_p2only_gatefloor_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_twofloor_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_ms_softprior_p2only_twofloor_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "oa_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_oa_only_light_nir_p2p5_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "oa_yolo_pan_light_nir_p2p5_c256_yolo11s_6cls_personmerge":
        return str(
            (
                root
                / "configs"
                / "models"
                / "yolo11s_rgbnir_oa_yolo_pan_light_nir_p2p5_c256_6cls_personmerge.yaml"
            ).resolve()
        )
    if mode == "rgbnir_light_nir_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_light_nir_6cls_personmerge.yaml").resolve())
    if mode == "attention_only":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_attention_only.yaml").resolve())
    if mode == "full_proposed":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_full_proposed.yaml").resolve())
    if mode == "full_proposed_residual":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_full_proposed_residual.yaml").resolve())
    if mode == "full_proposed_residual_v2":
        return str((root / "configs" / "models" / "yolo11n_rgbnir_full_proposed_residual_v2.yaml").resolve())
    if mode == "full_proposed_residual_v2_yolo11s":
        config = (
            "yolo11s_rgbnir_full_proposed_residual_v2_6cls_personmerge.yaml"
            if use_personmerge_schema(mode)
            else "yolo11s_rgbnir_full_proposed_residual_v2.yaml"
        )
        return str((root / "configs" / "models" / config).resolve())
    if mode == "full_proposed_residual_v2_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_full_proposed_residual_v2_6cls_personmerge.yaml").resolve())
    if mode == "proposed_lite_yolo11s_6cls_personmerge":
        return str((root / "configs" / "models" / "yolo11s_rgbnir_proposed_lite_p34_6cls_personmerge.yaml").resolve())
    if mode == "proposed_lite_light_nir_yolo11s_6cls_personmerge":
        return str(
            (root / "configs" / "models" / "yolo11s_rgbnir_proposed_lite_light_nir_p34_6cls_personmerge.yaml").resolve()
        )
    raise ValueError(f"Unsupported mode: {mode}")


def mode_specific_kwargs(mode: str) -> dict[str, object]:
    if mode in {"rgb", "rgb_yolo11s", "rgb_yolo11s_6cls_personmerge", "rgb_rtdetr"}:
        return {"use_simotm": "BGR", "channels": 3}
    if mode in {"nir", "nir_yolo11s_6cls_personmerge"}:
        return {"use_simotm": "Gray", "channels": 1}
    if mode in (
        TRAINABLE_MODES
        - {"rgb", "rgb_yolo11s", "rgb_yolo11s_6cls_personmerge", "rgb_rtdetr", "nir", "nir_yolo11s_6cls_personmerge"}
    ) | {"decision_fusion"}:
        return {"use_simotm": "RGBNIR", "channels": 4, "pairs_rgb_ir": DEFAULT_PAIRS}
    raise ValueError(f"Unsupported mode: {mode}")


def train_batch_for(mode: str) -> int:
    batches = {
        "rgb": 96,
        "rgb_yolo11s": 48,
        "rgb_yolo11s_6cls_personmerge": 48,
        "rgb_rtdetr": 32,
        "nir": 96,
        "nir_yolo11s_6cls_personmerge": 20,
        "early_fusion_yolo11s_6cls_personmerge": 20,
        "rgbnir": 48,
        "rgbnir_yolo11s_6cls_personmerge": 20,
        "input_fusion": 96,
        "light_gate": 48,
        "bifpn_only": 48,
        "bifpn_only_yolo11s": 24,
        "bifpn_only_yolo11s_6cls_personmerge": 24,
        "bifpn_only_light_nir_yolo11s_6cls_personmerge": 24,
        "bifpn_only_light_nir_p2_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_floor005_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_headp2_smallprior_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_gatefloor_c256_yolo11s_6cls_personmerge": 20,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_twofloor_c256_yolo11s_6cls_personmerge": 20,
        "oa_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge": 20,
        "oa_yolo_pan_light_nir_p2p5_c256_yolo11s_6cls_personmerge": 20,
        "rgbnir_light_nir_yolo11s_6cls_personmerge": 24,
        "attention_only": 48,
        "full_proposed": 48,
        "full_proposed_residual": 48,
        "full_proposed_residual_v2": 48,
        "full_proposed_residual_v2_yolo11s": 24,
        "full_proposed_residual_v2_yolo11s_6cls_personmerge": 24,
        "proposed_lite_yolo11s_6cls_personmerge": 24,
        "proposed_lite_light_nir_yolo11s_6cls_personmerge": 24,
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
        "nir_yolo11s_6cls_personmerge": 10,
        "early_fusion_yolo11s_6cls_personmerge": 10,
        "rgbnir": 10,
        "rgbnir_yolo11s_6cls_personmerge": 10,
        "input_fusion": 12,
        "light_gate": 10,
        "bifpn_only": 10,
        "bifpn_only_yolo11s": 10,
        "bifpn_only_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_floor005_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_headp2_smallprior_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_gatefloor_c256_yolo11s_6cls_personmerge": 10,
        "bifpn_only_light_nir_p2p5_oa_ms_softprior_p2only_twofloor_c256_yolo11s_6cls_personmerge": 10,
        "oa_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge": 10,
        "oa_yolo_pan_light_nir_p2p5_c256_yolo11s_6cls_personmerge": 10,
        "rgbnir_light_nir_yolo11s_6cls_personmerge": 10,
        "attention_only": 10,
        "full_proposed": 10,
        "full_proposed_residual": 10,
        "full_proposed_residual_v2": 10,
        "full_proposed_residual_v2_yolo11s": 10,
        "full_proposed_residual_v2_yolo11s_6cls_personmerge": 10,
        "proposed_lite_yolo11s_6cls_personmerge": 10,
        "proposed_lite_light_nir_yolo11s_6cls_personmerge": 10,
    }
    if mode not in workers:
        raise ValueError(f"Unsupported mode: {mode}")
    return workers[mode]


def common_train_kwargs(
    mode: str,
    epochs: int = 50,
    device: str = "0",
    val_interval: int = 1,
    imgsz: int = 640,
    optimizer: str | None = None,
    batch: int | None = None,
    lr0: float | None = None,
    cos_lr: bool = False,
) -> dict[str, object]:
    if mode not in TRAINABLE_MODES:
        raise ValueError(f"Mode does not support training: {mode}")
    optimizer_name = optimizer or os.getenv("OPTIMIZER", "SGD")
    train_batch = batch if batch and batch > 0 else train_batch_for(mode)
    close_mosaic = int(os.getenv("CLOSE_MOSAIC", "20"))
    kwargs = {
        "cache": "ram",
        "imgsz": imgsz,
        "epochs": epochs,
        "val_interval": max(int(val_interval), 1),
        "batch": train_batch,
        "close_mosaic": close_mosaic,
        "workers": workers_for(mode),
        "device": device,
        "optimizer": optimizer_name,
        "project": "runs/IDD_AW",
        "name": experiment_name(mode),
    }
    if cos_lr:
        kwargs["cos_lr"] = True
    if lr0 is not None:
        kwargs["lr0"] = lr0
    return kwargs


def common_val_kwargs(mode: str, imgsz: int = 640, batch: int | None = None) -> dict[str, object]:
    val_batch = batch if batch and batch > 0 else (16 if mode == "decision_fusion" else train_batch_for(mode))
    return {
        "split": "val",
        "imgsz": imgsz,
        "batch": val_batch,
        "project": "runs/IDD_AW_VAL",
        "name": experiment_name(mode),
    }


def common_predict_kwargs(mode: str, imgsz: int = 640) -> dict[str, object]:
    dataset_root = resolve_dataset_root(mode)
    source_subdir = "nir/val" if mode in {"nir", "nir_yolo11s_6cls_personmerge"} else "visible/val"
    return {
        "source": str((dataset_root / source_subdir).resolve()),
        "imgsz": imgsz,
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
