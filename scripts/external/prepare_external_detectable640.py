from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


CLASS_NAMES = [
    "person",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "autorickshaw",
    "traffic_light",
    "traffic_sign",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare IDD-AW detectable640 data adapters for external methods: "
            "RT-DETRv2, ICAFusion, and Efficient RGB-T Early Fusion."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_8cls_personmerge_traffic_detectable640",
        help="YOLO-format detectable640 dataset root.",
    )
    parser.add_argument(
        "--external-root",
        default="/data1/lvyanhu/code/externalmethod",
        help="Root directory that contains external method repositories.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing generated files/symlinks when needed.")
    return parser.parse_args()


def ensure_clean_link(link: Path, target: Path, force: bool) -> None:
    target = target.resolve()
    if link.is_symlink():
        current = link.resolve()
        if current == target:
            return
        if not force:
            raise FileExistsError(f"Symlink already exists with different target: {link} -> {current}")
        link.unlink()
    elif link.exists():
        if not force:
            raise FileExistsError(f"Path exists and is not a symlink: {link}")
        if link.is_dir():
            raise IsADirectoryError(f"Refusing to remove real directory: {link}")
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    return images


def yolo_xywhn_to_coco_bbox(values: list[float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = values
    box_w = max(0.0, bw * width)
    box_h = max(0.0, bh * height)
    x = (cx - bw / 2.0) * width
    y = (cy - bh / 2.0) * height
    x = min(max(x, 0.0), float(width))
    y = min(max(y, 0.0), float(height))
    box_w = min(box_w, float(width) - x)
    box_h = min(box_h, float(height) - y)
    return [float(x), float(y), float(max(0.0, box_w)), float(max(0.0, box_h))]


def convert_split_to_coco(dataset_root: Path, split: str) -> dict[str, object]:
    images_dir = dataset_root / "visible" / split
    images = list_images(images_dir)
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    ann_id = 1
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": int(width),
                "height": int(height),
            }
        )
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Expected YOLO bbox label with 5 columns at {label_path}:{line_no}, got {len(parts)}")
            cls = int(float(parts[0]))
            if not 0 <= cls < len(CLASS_NAMES):
                raise ValueError(f"Class id {cls} out of range at {label_path}:{line_no}")
            bbox = yolo_xywhn_to_coco_bbox([float(value) for value in parts[1:]], width, height)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return {
        "info": {"description": f"IDD-AW detectable640 {split} annotations converted from YOLO bbox labels"},
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(CLASS_NAMES)],
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_common_coco(dataset_root: Path, external_root: Path, force: bool) -> tuple[Path, Path]:
    coco_root = external_root / "datasets" / "iddaw_detectable640_coco"
    rgbt_root = external_root / "datasets" / "iddaw_detectable640_coco_rgbt"
    for split, coco_name in [("train", "train2017"), ("val", "val2017")]:
        coco = convert_split_to_coco(dataset_root, split)
        for root in [coco_root, rgbt_root]:
            write_json(root / "annotations" / f"instances_{coco_name}.json", coco)
        ensure_clean_link(coco_root / coco_name, dataset_root / "visible" / split, force)
    ensure_clean_link(rgbt_root / "visible", dataset_root / "visible", force)
    ensure_clean_link(rgbt_root / "infrared", dataset_root / "nir", force)
    return coco_root, rgbt_root


def prepare_icafusion(dataset_root: Path, external_root: Path, force: bool) -> list[Path]:
    generated: list[Path] = []
    ensure_clean_link(dataset_root / "infrared", dataset_root / "nir", force)
    labels_root = dataset_root / "labels"
    labels_root.mkdir(exist_ok=True)
    for split in ["train", "val"]:
        ensure_clean_link(labels_root / split, dataset_root / "visible" / split, force)
    # ICAFusion train.py hard-codes labels/test while val_rgb/val_ir are read
    # from the YAML. Point it to the validation labels without duplicating data.
    ensure_clean_link(labels_root / "test", dataset_root / "visible" / "val", force)

    icafusion_root = external_root / "ICAFusion"
    data_dir = icafusion_root / "data" / "multispectral"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = data_dir / "iddaw_detectable640_8cls.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                f"train_rgb: {dataset_root / 'visible' / 'train'}",
                f"val_rgb: {dataset_root / 'visible' / 'val'}",
                f"train_ir: {dataset_root / 'infrared' / 'train'}",
                f"val_ir: {dataset_root / 'infrared' / 'val'}",
                "nc: 8",
                f"names: {CLASS_NAMES}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    generated.append(data_yaml)

    hyp_path = icafusion_root / "data" / "hyp.iddaw_detectable640.yaml"
    hyp_path.parent.mkdir(parents=True, exist_ok=True)
    hyp_path.write_text(
        """# IDD-AW detectable640 hyperparameters for ICAFusion/yolov5s Transfusion.
lr0: 0.01
lrf: 0.1
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
""",
        encoding="utf-8",
    )
    generated.append(hyp_path)
    return generated


def prepare_rtdetr(external_root: Path, coco_root: Path) -> list[Path]:
    rtdetr_root = external_root / "RT-DETRv2-S"
    method_coco = rtdetr_root / "rt_dataset" / "dataset" / "coco"
    method_coco.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for name in ["train2017", "val2017", "annotations"]:
        ensure_clean_link(method_coco / name, coco_root / name, force=True)
    cfg = rtdetr_root / "train_config_iddaw_detectable640_8cls.yml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        """# RT-DETRv2-S RGB-only config for IDD-AW detectable640.
__include__: [
  'rt_detr/rtdetrv2_pytorch/configs/runtime.yml',
  'rt_detr/rtdetrv2_pytorch/configs/rtdetrv2/include/dataloader.yml',
  'rt_detr/rtdetrv2_pytorch/configs/rtdetrv2/include/optimizer.yml',
  'rt_detr/rtdetrv2_pytorch/configs/rtdetrv2/include/rtdetrv2_r50vd.yml',
]

task: detection
output_dir: ./output/rtdetrv2_s_rgb_iddaw_detectable640_8cls_100e
num_classes: 8
remap_mscoco_category: False

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']

train_dataloader:
  dataset:
    type: CocoDetection
    img_folder: ../../rt_dataset/dataset/coco/train2017/
    ann_file: ../../rt_dataset/dataset/coco/annotations/instances_train2017.json
    return_masks: False
    transforms:
      ops:
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
      policy:
        name: stop_epoch
        epoch: 97
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
  shuffle: True
  total_batch_size: 16
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction
    scales: ~

val_dataloader:
  dataset:
    type: CocoDetection
    img_folder: ../../rt_dataset/dataset/coco/val2017/
    ann_file: ../../rt_dataset/dataset/coco/annotations/instances_val2017.json
    return_masks: False
    transforms:
      ops:
        - {type: Resize, size: [640, 640]}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
  shuffle: False
  total_batch_size: 32
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction

PResNet:
  depth: 18
  freeze_at: -1
  freeze_norm: False
  pretrained: True

HybridEncoder:
  in_channels: [128, 256, 512]
  hidden_dim: 256
  expansion: 0.5

RTDETRTransformerv2:
  num_layers: 3

epoches: 100
use_amp: True
use_ema: True

optimizer:
  type: AdamW
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.0001
  params:
    -
      params: '^(?=.*backbone)(?!.*norm).*$'
      lr: 0.00001
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
      weight_decay: 0.

lr_scheduler:
  type: MultiStepLR
  milestones: [60, 80]
  gamma: 0.1

lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: 1000
""",
        encoding="utf-8",
    )
    generated.append(cfg)
    return generated


def prepare_efficient(external_root: Path, rgbt_root: Path) -> list[Path]:
    config_dir = external_root / "Efficient-RGB-T-Early-Fusion-Detection" / "mmdetection" / "configs" / "gfl"
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_dir / "gfl_r50_fpn_1x_iddaw_detectable640_shape.py"
    cfg.write_text(
        f"""_base_ = ['./gfl_r50_fpn_1x_coco.py']

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
dataset_type = 'M3FDDataset'
data_root = '{rgbt_root}/'
backend_args = None

classes = {tuple(CLASS_NAMES)!r}
metainfo = dict(classes=classes)

model = dict(
    type='GFLCLIP',
    backbone=dict(type='ResNetRGBTEarlyModifiedStem', in_channels=3, frozen_stages=-1),
    data_preprocessor=dict(
        _delete_=True,
        type='RGBTDetDataPreprocessor',
        thermal_mean=[84.1, 84.1, 84.1],
        thermal_std=[50.6, 50.6, 50.6],
        rgb_mean=[128.2, 129.3, 125.3],
        rgb_std=[49.1, 50.2, 53.5],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    bbox_head=dict(num_classes=8),
)

train_pipeline = [
    dict(type='LoadRGBTImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ResizeRGBT', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlipRGBT', prob=0.5),
    dict(type='PackDetRGBTInputs')
]
test_pipeline = [
    dict(type='LoadRGBTImageFromFile', backend_args=backend_args),
    dict(type='ResizeRGBT', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetRGBTInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        # LoadRGBTImageFromFile reads the thermal/NIR image from img_path and
        # obtains RGB by replacing 'infrared' with 'visible'. Therefore the
        # primary COCO image prefix must be infrared, not visible.
        data_prefix=dict(img='infrared/train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=4),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='infrared/val/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=3, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=47, by_epoch=True, begin=3, end=50),
]

train_cfg = dict(max_epochs=50, val_interval=1)
auto_scale_lr = dict(base_batch_size=64)
""",
        encoding="utf-8",
    )
    return [cfg]


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    external_root = Path(args.external_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    external_root.mkdir(parents=True, exist_ok=True)

    coco_root, rgbt_root = prepare_common_coco(dataset_root, external_root, args.force)
    generated = []
    generated.extend(prepare_icafusion(dataset_root, external_root, args.force))
    generated.extend(prepare_rtdetr(external_root, coco_root))
    generated.extend(prepare_efficient(external_root, rgbt_root))

    summary = {
        "dataset_root": str(dataset_root),
        "external_root": str(external_root),
        "coco_root": str(coco_root),
        "rgbt_coco_root": str(rgbt_root),
        "class_names": CLASS_NAMES,
        "generated_files": [str(path) for path in generated],
    }
    summary_path = external_root / "datasets" / "iddaw_detectable640_external_adapters.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
