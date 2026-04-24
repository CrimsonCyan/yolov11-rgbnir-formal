# 当前模型训练与状态总表

本文档整理当前 `yolov11-rgbnir-formal` 工程下所有已接入模型的：

1. 训练命令
2. 模型的具体实现方法
3. 训练参数与运行设置
4. 训练日志与结果存放位置
5. 当前最新训练状态及结果

更新时间：`2026-04-23`

## 1. 工程与数据根

- formal 工程目录：`E:\毕设\code\yolov11-rgbnir-formal`
- 远端工程目录：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal`
- 当前正式数据根：`/home/lym/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir`
- 并行 6 类数据根：`/home/lym/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge`
- 数据协议：
  - 数据集：`IDD-AW all-weather`
  - 模态：`RGB / NIR`
  - 类别数：`7`
  - 类别：`person, rider, motorcycle, car, truck, bus, autorickshaw`
- 并行 6 类口径：
  - 类别：`person, motorcycle, car, truck, bus, autorickshaw`
  - 映射：`rider -> person`
- 当前默认训练口径：
  - 默认 `IDDAW_CLASS_SCHEMA=6cls_personmerge`
  - 后续新训练若未显式指定 7 类，将默认走 `6 类 person+rider 合并` 方案

## 2. 统一训练入口

### 2.1 本地/远端统一 Python 入口

```bash
python scripts/iddaw/run_experiment.py --mode <mode> --task train --epochs <epochs> --device 0
```

### 2.2 远端后台训练入口

```bash
bash scripts/iddaw/launch_nohup_train.sh <mode> <epochs> 0
```

通过脚本后台训练时，当前默认行为为：

- `WANDB_ENABLED` 会按训练类型自动切换：
  - `epochs <= 1` 的冒烟训练默认 `0`
  - `epochs > 1` 的正式长训练默认 `1`
  - 若显式传入环境变量 `WANDB_ENABLED=0/1`，则以显式设置为准
- `WANDB_CONSOLE=off`，默认不上传高频 stdout/console 流
- `VAL_INTERVAL=1`，默认每个 epoch 验证一次
- `IDDAW_CLASS_SCHEMA=6cls_personmerge`，默认训练使用 6 类口径
- `OPTIMIZER=SGD`，默认使用 SGD；可通过环境变量覆盖为 `Adam/AdamW` 等 Ultralytics 支持的优化器
- 自动生成 `stdout.log / pid / meta` 三类远端日志文件
- 自动维护 `latest_<mode>.*` 软链接，便于追踪当前最新 run

### 2.3 从已完成 checkpoint 补训练到目标总 epoch

当前 formal 工程已支持“从 `last.pt` 继续补足到目标总 epoch”的模式。示例：

```bash
bash scripts/iddaw/launch_nohup_train.sh rgbnir 70 0 /home/lym/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/iddaw-yolo11n-rgbnir-plain2/weights/last.pt
```

说明：

- 这里的 `70` 表示目标总轮数。
- 若 checkpoint 已完成 `50` 轮，则脚本会自动再训练 `20` 轮。
- 当前不是严格恢复优化器状态的 Ultralytics `resume`，而是基于 `last.pt` 继续训练剩余轮数。

### 2.4 Decision-Fusion 离线评测入口

```bash
python scripts/iddaw/run_experiment.py --mode decision_fusion --task val --device 0
```

如需显式指定权重：

```bash
python scripts/iddaw/run_experiment.py --mode decision_fusion --task val --device 0 --rgb-weights <rgb_best.pt> --nir-weights <nir_best.pt>
```

## 3. 模型实现方法总览

| 模式 | 实现/配置文件 | 具体实现方法 |
| --- | --- | --- |
| `rgb` | `ultralytics/cfg/models/11/yolo11.yaml` | 官方 Ultralytics `YOLO11` 单模 RGB 检测器 |
| `rgb_yolo11s` | `ultralytics/cfg/models/11/yolo11s.yaml` | 官方 Ultralytics `YOLO11s` 单模 RGB 检测器 |
| `rgb_yolo11s_6cls_personmerge` | `ultralytics/cfg/models/11/yolo11s.yaml` | 官方 `YOLO11s` 单模 RGB 检测器，6 类口径并将 `rider` 合并到 `person` |
| `rgb_rtdetr` | `ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml` | 官方 Ultralytics `RT-DETR-R18` 单模 RGB 检测器 |
| `nir` | `ultralytics/cfg/models/11/yolo11-gray.yaml` | 官方 `YOLO11` 灰度单模检测器，输入为单通道 NIR |
| `rgbnir` | `configs/models/yolo11n_rgbnir_midfusion_plain.yaml` | 双流 RGB/NIR backbone，`P3/P4/P5` 同尺度 `Concat`，之后 `SPPF + C2PSA + YOLO head` |
| `input_fusion` | `configs/models/yolo11n_rgbnir_input_fusion.yaml` | 4 通道输入级融合，`RGB(3)+NIR(1)` 直接输入单 backbone |
| `light_gate` | `configs/models/yolo11n_rgbnir_midfusion_gate.yaml` | 双流 backbone，`P3/P4/P5` 用 `ConcatGate` 代替 plain concat |
| `bifpn_only` | `configs/models/yolo11n_rgbnir_bifpn_only.yaml` | 双流 backbone + plain mid-fusion，neck 换成 `BiFPN(256, repeat=2)` |
| `attention_only` | `configs/models/yolo11n_rgbnir_attention_only.yaml` | 双流 backbone，`P3/P4/P5` 用 `QualityAwareFusion`，neck/head 保持 plain |
| `full_proposed` | `configs/models/yolo11n_rgbnir_full_proposed.yaml` | `QualityAwareFusion + BiFPN` 组合版历史路线 |
| `full_proposed_residual` | `configs/models/yolo11n_rgbnir_full_proposed_residual.yaml` | `ResidualQualityAwareFusion + BiFPN` 组合版历史路线 |
| `full_proposed_residual_v2` | `configs/models/yolo11n_rgbnir_full_proposed_residual_v2.yaml` | 当前正式 `Proposed`：`ResidualQualityAwareFusionV2 + BiFPN` |
| `bifpn_only_yolo11s` | `configs/models/yolo11s_rgbnir_bifpn_only.yaml` | `YOLO11s` 版双流 `BiFPN-only` |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_only_light_nir_6cls_personmerge.yaml` | `YOLO11s` 版 `BiFPN-only + Light NIR branch`：保留 `P3` NIR 分支，压缩 `P4/P5` NIR 语义通道并投影回融合尺度后做 plain concat |
| `full_proposed_residual_v2_yolo11s` | `configs/models/yolo11s_rgbnir_full_proposed_residual_v2.yaml` | `YOLO11s` 版 `ResidualQualityAwareFusionV2 + BiFPN` |
| `proposed_lite_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_proposed_lite_p34_6cls_personmerge.yaml` | `YOLO11s` 版 `Proposed-Lite`：`P3/P4` 用 `ResidualQualityAwareFusionV2`，`P5` 回退为 `Concat`，之后进入 `BiFPN` |
| `proposed_lite_light_nir_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_proposed_lite_light_nir_p34_6cls_personmerge.yaml` | `YOLO11s` 版 `Proposed-Lite + Light NIR branch`：保持 `P3` NIR 分支，压缩 `P4/P5` NIR 语义通道后再投影回融合尺度 |
| `decision_fusion` | `formal_rgbnir/decision_fusion.py` | 离线结果级融合：读取 `rgb` 与 `nir` 权重，推理后做 batched NMS 融合 |

## 4. 统一训练参数与设置

### 4.1 通用正式设置

- 输入尺寸：`640`
- 优化器：`SGD`
- 数据缓存：`cache=ram`
- `close_mosaic=10`
- 设备：`device=0`
- 项目目录：`runs/IDD_AW`
- 脚本默认验证频率：`val_interval=1`
- 脚本默认 W\&B console：`WANDB_CONSOLE=off`
- 统一入口支持可选覆盖：`--optimizer`、`--batch`

### 4.2 模式级差异

| 模式 | 输入模态 | `use_simotm` | `channels` | batch | workers | 正式主口径 |
| --- | --- | --- | --- | --- | --- | --- |
| `rgb` | `visible/train,val` | `BGR` | `3` | `96` | `12` | `50 epoch` |
| `rgb_yolo11s` | `visible/train,val` | `BGR` | `3` | `48` | `12` | 已完成 `80 epoch` |
| `rgb_yolo11s_6cls_personmerge` | `visible/train,val` | `BGR` | `3` | `48` | `12` | 已完成 `80 epoch` |
| `rgb_rtdetr` | `visible/train,val` | `BGR` | `3` | `32` | `10` | `50 epoch`，现补到 `70` 中 |
| `nir` | `nir/train,val` | `Gray` | `1` | `96` | `12` | `50 epoch` |
| `rgbnir` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | `50 epoch`，现补到 `70` |
| `input_fusion` | `visible/train,val` + paired nir | `RGBNIR` | `4` | `96` | `12` | `50 epoch` |
| `light_gate` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | 当前无 all-weather 正式长训 |
| `bifpn_only` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | `50 epoch`，现补到 `70` |
| `attention_only` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | `50 epoch` |
| `full_proposed` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | 历史 `25 epoch` 路线 |
| `full_proposed_residual` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | 历史 `25 epoch` 路线 |
| `full_proposed_residual_v2` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | 当前 `YOLO11n Proposed`，已完成 `80 epoch` |
| `full_proposed_residual_v2_yolo11s` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | `YOLO11s Proposed`，已完成 `70 epoch` |
| `full_proposed_residual_v2_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | 待启动 `70 epoch` |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `32` | `10` | 已完成 `100 epoch` |
| `proposed_lite_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | 已完成 `70 epoch` |
| `proposed_lite_light_nir_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | 当前不在主线，保留为备选结构 |
| `bifpn_only_yolo11s` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | `YOLO11s BiFPN-only`，已完成 `70 epoch` |
| `bifpn_only_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | 已完成 `50 epoch` |

补充：

- `decision_fusion` 不训练，只做离线融合评测。
- 双模态模式通过 `pairs_rgb_ir=["visible", "nir"]` 读取配对图像。

## 5. 训练日志与结果存放位置

### 5.1 远端训练日志

统一目录：

```text
/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw
```

每次训练会生成：

- `<mode>_e<epochs>_<timestamp>.stdout.log`
- `<mode>_e<epochs>_<timestamp>.pid`
- `<mode>_e<epochs>_<timestamp>.meta`

同时维护最新软链接：

- `latest_<mode>.stdout.log`
- `latest_<mode>.pid`
- `latest_<mode>.meta`

### 5.2 结果目录

统一目录：

```text
/home/lym/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW
```

其中每个 run 一般包含：

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `confusion_matrix*.png`
- `PR_curve.png / P_curve.png / R_curve.png / F1_curve.png`

### 5.3 常用查看命令

查看 RGBNIR plain 最新日志：

```bash
ssh 4_3090 "tail -f /home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/latest_rgbnir.stdout.log"
```

查看 BiFPN-only 最新日志：

```bash
ssh 4_3090 "tail -f /home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/latest_bifpn_only.stdout.log"
```

查看 RT-DETR 最新日志：

```bash
ssh 4_3090 "tail -f /home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/latest_rgb_rtdetr.stdout.log"
```

查看 70 epoch 续训队列总日志：

```bash
ssh 4_3090 "tail -f /home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/resume70_queue.stdout.log"
```

## 6. 当前最新训练状态与结果

### 6.1 已完成的正式主线结果

| 模式 | 最新正式 run | 训练口径 | 当前状态 | P | R | mAP50 | mAP50-95 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rgb` | `iddaw-yolo11n-rgb2` | `50 epoch` | 已完成 | `0.58182` | `0.41056` | `0.43404` | `0.27339` | 官方 YOLO11 RGB-only 基线 |
| `rgb_yolo11s` | `iddaw-yolo11s-rgb7` | `80 epoch` | 已完成 | `0.69330` | `0.48334` | `0.53782` | `0.36051` | 官方 YOLO11s RGB-only 基线 |
| `rgb_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgb-6cls-personmerge2` | `80 epoch` | 已完成 | `0.66588` | `0.53652` | `0.57578` | `0.39398` | 6 类口径，`rider -> person` 后的 YOLO11s RGB-only |
| `bifpn_only_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge2` | `50 epoch` | 已完成 | `0.69297` | `0.51787` | `0.58269` | `0.39173` | 6 类口径下当前已完成的 YOLO11s BiFPN-only 对照 |
| `proposed_lite_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-proposed-lite-p34-6cls-personmerge3` | `70 epoch` | 已完成 | `0.76323` | `0.50914` | `0.59353` | `0.39896` | 6 类口径下 `P3/P4` 质量感知 + `P5 plain concat` 的 Proposed-Lite |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge4` | `100 epoch, batch=32` | 已完成 | `0.71556` | `0.55095` | `0.60467` | `0.41390` | 当前 `YOLO11s + RGB-NIR + 640 + SGD` 最强结果，Light NIR branch |
| `nir` | `iddaw-yolo11n-nir2` | `50 epoch` | 已完成 | `0.57803` | `0.36977` | `0.40328` | `0.24597` | 官方 YOLO11 Gray/NIR 基线 |
| `rgbnir` | `iddaw-yolo11n-rgbnir-plain2` | `50 epoch` | 已完成 | `0.63680` | `0.44948` | `0.48136` | `0.30476` | 7 类双流 plain baseline |
| `rgbnir (6cls personmerge)` | `iddaw-yolo11n-rgbnir-plain-6cls-personmerge2` | `100 epoch, imgsz=800, Adam` | 已完成 | `0.71500` | `0.54400` | `0.61200` | `0.41500` | 默认 6 类口径，`person+rider` 合并后的双流 plain，当前最新为 Adam 版 |
| `input_fusion` | `iddaw-yolo11n-input-fusion` | `50 epoch` | 已完成 | `0.55391` | `0.42494` | `0.44575` | `0.27876` | 4 通道输入级融合 |
| `attention_only` | `iddaw-yolo11n-rgbnir-attention-only3` | `50 epoch` | 已完成 | `0.65444` | `0.43345` | `0.48363` | `0.30789` | `QualityAwareFusion` |
| `bifpn_only` | `iddaw-yolo11n-rgbnir-bifpn-only3` | `50 epoch` | 已完成 | `0.68079` | `0.46012` | `0.51515` | `0.33616` | 当前 50 epoch 最强基线 |
| `full_proposed_residual_v2` | `iddaw-yolo11n-rgbnir-full-proposed-residual-v23` | `80 epoch` | 已完成 | `0.69012` | `0.47603` | `0.52312` | `0.34447` | 当前正式 `YOLO11n Proposed` |
| `full_proposed_residual_v2_yolo11s` | `iddaw-yolo11s-rgbnir-full-proposed-residual-v22` | `70 epoch` | 已完成 | `0.67454` | `0.50943` | `0.54620` | `0.35699` | `YOLO11s` 版 RGB-NIR Proposed |
| `bifpn_only_yolo11s` | `iddaw-yolo11s-rgbnir-bifpn-only2` | `70 epoch` | 已完成 | `0.68281` | `0.48188` | `0.53924` | `0.35857` | `YOLO11s` 版 BiFPN-only |
| `rgb_rtdetr` | `iddaw-rtdetr-r18-rgb2` | `50 epoch` | 已完成 | `0.42766` | `0.33811` | `0.32081` | `0.18771` | 外部现成 RGB 单模基线 |

### 6.2 历史路线结果

| 模式 | 最新历史 run | 训练口径 | 当前状态 | P | R | mAP50 | mAP50-95 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_proposed` | `iddaw-yolo11n-rgbnir-full-proposed2` | `25 epoch` | 历史完成 | `0.62252` | `0.43418` | `0.46426` | `0.29263` | `QualityAwareFusion + BiFPN` |
| `full_proposed_residual` | `iddaw-yolo11n-rgbnir-full-proposed-residual2` | `25 epoch` | 历史完成 | `0.67632` | `0.42003` | `0.47187` | `0.29707` | `ResidualQualityAwareFusion + BiFPN` |

### 6.3 当前未进入 all-weather 正式主线的模式

| 模式 | 当前状态 | 说明 |
| --- | --- | --- |
| `light_gate` | 当前无 all-weather 正式长训 | 历史上已做过 FOG 子集验证，但当前 `IDD-AW all-weather / 50 epoch` 主口径下未正式重跑 |

### 6.4 Decision-Fusion 当前状态

- 结果目录：`runs/IDD_AW/iddaw-yolo11n-decision-fusion`
- 当前状态：已完成离线融合评测
- 当前使用权重：
  - RGB：`runs/IDD_AW/iddaw-yolo11n-rgb2/weights/best.pt`
  - NIR：`runs/IDD_AW/iddaw-yolo11n-nir2/weights/best.pt`
- 当前指标：
  - `AP50 = 0.4349117256474419`
  - `mAP@0.5:0.95 = 0.27518500903758975`
  - `AP_S = 0.0003632636341665473`
  - `AP_M = 0.07798446116368306`
  - `AP_L = 0.5591813057122286`

### 6.5 70 epoch 补训练测试状态

当前只把 3 条线从 `50` 补到 `70` 做趋势测试：

1. `rgbnir plain`
2. `bifpn_only`
3. `rgb_rtdetr`

#### `rgbnir plain`

- 续训命令：

```bash
bash scripts/iddaw/launch_nohup_train.sh rgbnir 70 0 /home/lym/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/iddaw-yolo11n-rgbnir-plain2/weights/last.pt
```

- 当前状态：已完成 `50 -> 70` 补训练
- latest log 给出的最终验证结果：
  - `P = 0.641`
  - `R = 0.458`
  - `mAP50 = 0.501`
  - `mAP50-95 = 0.320`
- 结论：相对 `50 epoch` 的 `0.48136 / 0.30476`，继续上升

#### `bifpn_only`

- 续训命令：

```bash
bash scripts/iddaw/launch_nohup_train.sh bifpn_only 70 0 /home/lym/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/iddaw-yolo11n-rgbnir-bifpn-only3/weights/last.pt
```

- 当前状态：已完成 `50 -> 70` 补训练
- latest log 给出的最终验证结果：
  - `P = 0.708`
  - `R = 0.488`
  - `mAP50 = 0.540`
  - `mAP50-95 = 0.354`
- 结论：相对 `50 epoch` 的 `0.51515 / 0.33616`，继续上升，仍是当前最强路线

#### `rgb_rtdetr`

- 续训命令：

```bash
bash scripts/iddaw/launch_nohup_train.sh rgb_rtdetr 70 0 /home/lym/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/iddaw-rtdetr-r18-rgb2/weights/last.pt
```

- 当前状态：已完成 `50 -> 70` 补训练
- latest log 给出的最终验证结果：
  - `P = 0.474`
  - `R = 0.391`
  - `mAP50 = 0.369`
  - `mAP50-95 = 0.224`
- 结论：相对 `50 epoch` 的 `0.32081 / 0.18771`，继续训练后有明显提升，但整体仍弱于当前 YOLO11 RGB-only 与主要 RGB-NIR 路线

### 6.6 `rgb_yolo11s` 80 epoch 从零重训练结果

- 当前正式完成 run：`iddaw-yolo11s-rgb7`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgb7`
- W&B run：`36oh1gse`
- 最终指标：
  - `Precision = 0.69330`
  - `Recall = 0.48334`
  - `mAP50 = 0.53782`
  - `mAP50-95 = 0.36051`
- 相比 `50 epoch` 的 `0.50990 / 0.33412`，提升为：
  - `mAP50 +0.02792`
  - `mAP50-95 +0.02639`
- 结论：`YOLO11s RGB-only` 已成为当前更强的单模 RGB 基线，并且当前结果已经高于 `YOLO11n` 版 Proposed 的 80 epoch 结果。这直接说明后续需要验证：在更强 backbone/scale 下，`RGB-NIR Proposed` 是否仍能保持增益。

### 6.7 `YOLO11s` 版 BiFPN-only 70 epoch 结果

- 当前正式完成 run：`iddaw-yolo11s-rgbnir-bifpn-only2`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only2`
- W&B run：`mdndl274`
- 当前模型规模：
  - `16,476,153` parameters
  - `54.82 GFLOPs`
- 最终指标：
  - `Precision = 0.68281`
  - `Recall = 0.48188`
  - `mAP50 = 0.53924`
  - `mAP50-95 = 0.35857`
- 与当前 `YOLO11s RGB-only` 基线对比：
  - `YOLO11s RGB-only`：`0.53782 / 0.36051`
  - `YOLO11s RGB-NIR BiFPN-only`：`0.53924 / 0.35857`
  - 差值：`mAP50 +0.00142`，`mAP50-95 -0.00194`
- 结果解读：
  - 在更强 `YOLO11s` 尺度下，单独加入 `BiFPN` 后仍能把 `mAP50` 推到略高于 `RGB-only` 的水平。
  - 但 `mAP50-95` 仍略低于 `RGB-only`，说明更严格 IoU 口径下的框质量优势还不稳定。
  - 当前它和 `YOLO11s Proposed` 都落在与 `YOLO11s RGB-only` 非常接近的区间，说明尺度增强后，纯结构增益被明显压缩。

### 6.8 `YOLO11s` 版 Proposed 70 epoch 结果


- 当前正式完成 run：`iddaw-yolo11s-rgbnir-full-proposed-residual-v22`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-full-proposed-residual-v22`
- W&B run：`xtarzf9w`
- 当前模型规模：
  - `17,466,395` parameters
  - `55.64 GFLOPs`
- 最终 best.pt 验证指标：
  - `Precision = 0.67454`
  - `Recall = 0.50943`
  - `mAP50 = 0.54620`
  - `mAP50-95 = 0.35699`
- 与当前 `YOLO11s RGB-only` 基线对比：
  - `YOLO11s RGB-only`：`0.53782 / 0.36051`
  - `YOLO11s RGB-NIR Proposed`：`0.54620 / 0.35699`
  - 差值：`mAP50 +0.00838`，`mAP50-95 -0.00352`
- 结果解读：
  - RGB-NIR Proposed 在 `mAP50` 上略高于 `YOLO11s RGB-only`，同时 `Recall` 也更高（`+0.02609`）。
  - 但 `mAP50-95` 略低于 `YOLO11s RGB-only`，说明在更严格 IoU 口径下，当前 `YOLO11s` 版 Proposed 还不足以证明对更强 RGB-only 基线形成了稳定全面优势。
  - 因此这条结果更适合作为“RGB-NIR 在更强 backbone 下仍具备一定增益潜力”的证据，而不是直接替代现有最佳方案。

### 6.9 6 类 `person+rider` 并行方案阶段性结果

- 当前并行数据口径：`iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge`
- 类别定义：`person, motorcycle, car, truck, bus, autorickshaw`
- 合并规则：旧 `rider` 统一并入 `person`
- 已完成关键模型：
  - `rgb_yolo11s_6cls_personmerge`：`80 epoch`
  - `rgbnir`（默认 6 类口径）：`100 epoch, imgsz=800`
  - `bifpn_only_yolo11s_6cls_personmerge`：`50 epoch`
  - `proposed_lite_yolo11s_6cls_personmerge`：`70 epoch`
- 已完成 run：
  - `iddaw-yolo11s-rgb-6cls-personmerge2`
  - `iddaw-yolo11n-rgbnir-plain-6cls-personmerge`
  - `iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge2`
  - `iddaw-yolo11s-rgbnir-proposed-lite-p34-6cls-personmerge3`
- W\&B：
  - `https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/ou6tiufd`
  - `https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/5es3st3o`
  - `https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/6b8alwxd`
  - `https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/3ai9kzz3`
- 正式结果（`best.pt`）：
  - `Precision = 0.66588`
  - `Recall = 0.53652`
  - `mAP50 = 0.57578`
  - `mAP50-95 = 0.39398`
- 与 7 类 `YOLO11s RGB-only`（`0.53782 / 0.36051`）对比：
  - `mAP50 +0.03796`
  - `mAP50-95 +0.03347`
  - `Recall +0.05318`
  - `Precision -0.02742`
- 结果解读：
  - 将 `rider` 并入 `person` 后，总体 `Recall` 与两档 AP 都有明确提升，说明原 7 类口径下 `person/rider` 的类别边界确实在污染监督质量。
  - 但按类结果看，`person`（`mAP50=0.370`）和 `motorcycle`（`mAP50=0.397`）仍明显弱于 `car` 等中大目标，说明仅靠类别合并还不足以彻底解决小目标检测问题。
  - 因此 6 类口径已经足够作为下一阶段 `YOLO11s RGB-NIR` 主线，后续重点不再是标签合并本身，而是结构是否能把增益真正送到 `person/motorcycle` 这类小目标。
- 已补齐关键模型：
  - `bifpn_only_light_nir_yolo11s_6cls_personmerge`：`100 epoch`
- 仍可补齐的对照模型：
  - `full_proposed_residual_v2_yolo11s_6cls_personmerge`：`70 epoch`
- 当前对应日志：
  - `latest_bifpn_only_yolo11s_6cls_personmerge.stdout.log`
  - `latest_proposed_lite_yolo11s_6cls_personmerge.stdout.log`
  - `latest_bifpn_only_light_nir_yolo11s_6cls_personmerge.stdout.log`

### 6.10 `YOLO11s Proposed-Lite` 6 类 70 epoch 结果

- 当前正式完成 run：`iddaw-yolo11s-rgbnir-proposed-lite-p34-6cls-personmerge3`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-proposed-lite-p34-6cls-personmerge3`
- W&B run：`3ai9kzz3`
- 当前模型规模：
  - `16,787,554` parameters
  - `54.68 GFLOPs`
- `best.pt` 最终验证指标：
  - `Precision = 0.76323`
  - `Recall = 0.50914`
  - `mAP50 = 0.59353`
  - `mAP50-95 = 0.39896`
- `last` epoch 指标：
  - `Precision = 0.70273`
  - `Recall = 0.53047`
  - `mAP50 = 0.58620`
  - `mAP50-95 = 0.39439`
- `best.pt` 按类结果：
  - `person`: `mAP50 = 0.409`, `mAP50-95 = 0.193`
  - `motorcycle`: `mAP50 = 0.417`, `mAP50-95 = 0.189`
  - `car`: `mAP50 = 0.840`, `mAP50-95 = 0.613`
  - `truck`: `mAP50 = 0.571`, `mAP50-95 = 0.403`
  - `bus`: `mAP50 = 0.635`, `mAP50-95 = 0.487`
  - `autorickshaw`: `mAP50 = 0.690`, `mAP50-95 = 0.508`
- 与 `rgb_yolo11s_6cls_personmerge` 对比：
  - `YOLO11s RGB-only 6 类`：`0.57578 / 0.39398`
  - `YOLO11s RGB-NIR Proposed-Lite`：`0.59353 / 0.39896`
  - 差值：`mAP50 +0.01775`，`mAP50-95 +0.00498`
- 与当前已完成的 `bifpn_only_yolo11s_6cls_personmerge` 对比：
  - `YOLO11s RGB-NIR BiFPN-only 6 类`：`0.58269 / 0.39173`
  - `YOLO11s RGB-NIR Proposed-Lite`：`0.59353 / 0.39896`
  - 差值：`mAP50 +0.01084`，`mAP50-95 +0.00723`
- 与 `bifpn_only_yolo11s_6cls_personmerge` 的关键类别对比：
  - `person`: `0.443 / 0.212 -> 0.409 / 0.193`
  - `motorcycle`: `0.427 / 0.190 -> 0.417 / 0.189`
  - `car`: `0.853 / 0.627 -> 0.840 / 0.613`
  - `truck`: `0.543 / 0.383 -> 0.571 / 0.403`
  - `bus`: `0.559 / 0.432 -> 0.635 / 0.487`
  - `autorickshaw`: `0.671 / 0.506 -> 0.690 / 0.508`
- 结果解读：
  - `Proposed-Lite` 已经证明“`P3/P4` 保留质量感知、`P5` 回退 plain concat”这条方向是成立的。至少在当前 `YOLO11s + 6 类 + 640 + SGD` 配方下，它整体上已经同时超过 `RGB-only` 和当前已完成的 `BiFPN-only` 6 类对照。
  - 但这次提升主要来自 `truck / bus / autorickshaw` 等中大目标类别，`person / motorcycle` 两个更关键的小目标类别没有同步提升，甚至相对 `BiFPN-only` 还有轻微回落。
  - 这说明当前问题已经不再是 `P5` 是否过度融合，而是 `NIR` 深层语义分支仍可能过重，导致它在高层语义上带来的噪声大于收益。
  - 基于这个结果，当前主线不再继续沿残差质量感知方向加结构，而是切到 `BiFPN-only + Light NIR branch`：保留 plain concat 与 `BiFPN` 骨架，只压缩 `NIR` 的深层语义容量，再测试是否能把整体收益转移到 `person / motorcycle`。

### 6.11 `YOLO11s BiFPN-only + Light NIR branch` 当前状态

- 当前主线 mode：`bifpn_only_light_nir_yolo11s_6cls_personmerge`
- 结构定义：
  - RGB 分支保持完整 `YOLO11s` backbone
  - NIR 分支保留 `P3` 路径
  - `P4/P5` 的 NIR 深层语义通道压缩后，再通过 `1x1 Conv` 投影回 RGB 对齐尺度
  - 融合方式保持 `plain Concat + BiFPN`，不再引入残差质量感知融合

#### `1 epoch` 冒烟结果

- 当前 run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge`
- 日志：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_yolo11s_6cls_personmerge_e1_20260424_001758.stdout.log`
- meta：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_yolo11s_6cls_personmerge_e1_20260424_001758.meta`
- 运行配置：
  - `imgsz=640`
  - `optimizer=SGD`
  - `batch=24`
  - `WANDB_ENABLED=0`
- 模型规模：
  - build summary：`14,167,414` parameters / `51.72 GFLOPs`
  - fused summary：`14,148,598` parameters / `51.30 GFLOPs`
- `best.pt` 验证指标：
  - `Precision = 0.000860`
  - `Recall = 0.0474`
  - `mAP50 = 0.000519`
  - `mAP50-95 = 0.000158`
- 冒烟结论：
  - 训练、验证、`best.pt/last.pt` 导出均正常完成
  - mode 注册、数据集 YAML 生成、日志与 `meta` 落盘都正常
  - 指标仍是随机初始化后一轮训练的正常低水平，可作为结构可跑通验证

#### `100 epoch` 正式训练结果

- 当前 run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge4`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge4`
- 日志：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_yolo11s_6cls_personmerge_e100_20260424_145406.stdout.log`
- meta：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_yolo11s_6cls_personmerge_e100_20260424_145406.meta`
- W&B run：`v4ip3znv`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/v4ip3znv`
- 运行配置：
  - `imgsz=640`
  - `optimizer=SGD`
  - `batch=32`
  - `epochs=100`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 模型规模：
  - build summary：`14,167,414` parameters / `51.72 GFLOPs`
  - fused summary：`14,148,598` parameters / `51.30 GFLOPs`
- 训练完成状态：
  - `100 epochs completed in 1.000 hours`
  - `results.csv` 共 `101` 行，含表头和 `100` 个 epoch
  - `best.pt` 与 `last.pt` 均已导出并完成 optimizer strip

- `best.pt` 复验指标：
  - `Precision = 0.71556`
  - `Recall = 0.55095`
  - `mAP50 = 0.60467`
  - `mAP50-95 = 0.41390`
- `results.csv` 最优 epoch：
  - epoch `99`
  - `Precision = 0.71627`
  - `Recall = 0.55000`
  - `mAP50 = 0.60428`
  - `mAP50-95 = 0.41362`
- epoch `100` 指标：
  - `Precision = 0.71643`
  - `Recall = 0.55472`
  - `mAP50 = 0.60367`
  - `mAP50-95 = 0.41336`

- `best.pt` 主要类别表现：
  - `person`: `mAP50 = 0.431`, `mAP50-95 = 0.207`
  - `motorcycle`: `mAP50 = 0.434`, `mAP50-95 = 0.183`
  - `car`: `mAP50 = 0.852`, `mAP50-95 = 0.628`
  - `truck`: `mAP50 = 0.575`, `mAP50-95 = 0.422`
  - `bus`: `mAP50 = 0.637`, `mAP50-95 = 0.523`
  - `autorickshaw`: `mAP50 = 0.698`, `mAP50-95 = 0.522`

- 与 6 类 `YOLO11s RGB-only`（`0.57578 / 0.39398`）对比：
  - `mAP50 +0.02889`
  - `mAP50-95 +0.01992`
- 与 `YOLO11s BiFPN-only 6cls`（`0.58269 / 0.39173`）对比：
  - `mAP50 +0.02198`
  - `mAP50-95 +0.02217`
- 与 `Proposed-Lite`（`0.59353 / 0.39896`）对比：
  - `mAP50 +0.01114`
  - `mAP50-95 +0.01494`
- 与 `RGB-NIR plain + Adam + 800`（`0.612 / 0.415`）对比：
  - `mAP50 -0.00733`
  - `mAP50-95 -0.00110`

- 结果解读：
  - `BiFPN-only + Light NIR branch` 已成为当前 `YOLO11s + RGB-NIR + 6 类 + 640 + SGD` 下最强结构，整体超过 `RGB-only`、原始 `BiFPN-only` 和 `Proposed-Lite`。
  - 轻量 NIR 分支的收益比残差质量感知融合更稳定，说明当前阶段继续压缩 NIR 深层语义比继续加重跨模态注意力更有效。
  - `person` 相对 `Proposed-Lite` 从 `0.409 / 0.193` 回升到 `0.431 / 0.207`，但仍低于原始 `BiFPN-only` 的 `0.443 / 0.212`。
  - `motorcycle` 的 `mAP50` 从 `0.417` 回升到 `0.434`，但 `mAP50-95` 仍只有 `0.183`，低于原始 `BiFPN-only` 的 `0.190` 和 plain Adam 的 `0.201`。
  - 当前最主要的剩余问题不是整体框架有效性，而是小目标高 IoU 定位质量仍偏弱。

#### `100 epoch` 高配方确认结果（`imgsz=800`, `Adam`）

- 当前 run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge5`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge5`
- 日志：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_yolo11s_6cls_personmerge_e100_20260424_165228.stdout.log`
- meta：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_yolo11s_6cls_personmerge_e100_20260424_165228.meta`
- W&B run：`78q6hoxl`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/78q6hoxl`
- 运行配置：
  - `imgsz=800`
  - `optimizer=Adam`
  - `batch=20`
  - `epochs=100`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 模型规模：
  - fused summary：`14,148,598` parameters / `80.15 GFLOPs`
- 训练完成状态：
  - `100 epochs completed in 1.575 hours`
  - `results.csv` 共 `100` 个 epoch
  - `best.pt` 与 `last.pt` 均已导出并完成 optimizer strip

- `best.pt` 复验指标：
  - `Precision = 0.76181`
  - `Recall = 0.60168`
  - `mAP50 = 0.66055`
  - `mAP50-95 = 0.47087`
- `results.csv` 最优 epoch：
  - epoch `99`
  - `Precision = 0.76375`
  - `Recall = 0.59856`
  - `mAP50 = 0.66068`
  - `mAP50-95 = 0.47102`
- epoch `100` 指标：
  - `Precision = 0.76175`
  - `Recall = 0.59842`
  - `mAP50 = 0.66027`
  - `mAP50-95 = 0.47022`

- `best.pt` 主要类别表现：
  - `person`: `mAP50 = 0.502`, `mAP50-95 = 0.255`
  - `motorcycle`: `mAP50 = 0.522`, `mAP50-95 = 0.243`
  - `car`: `mAP50 = 0.884`, `mAP50-95 = 0.679`
  - `truck`: `mAP50 = 0.630`, `mAP50-95 = 0.474`
  - `bus`: `mAP50 = 0.715`, `mAP50-95 = 0.611`
  - `autorickshaw`: `mAP50 = 0.709`, `mAP50-95 = 0.564`

- 与同结构 `640 + SGD` 正式结果（`0.60467 / 0.41390`）对比：
  - `mAP50 +0.05588`
  - `mAP50-95 +0.05697`
  - `person`: `0.431 / 0.207 -> 0.502 / 0.255`
  - `motorcycle`: `0.434 / 0.183 -> 0.522 / 0.243`
- 与 `RGB-NIR plain + Adam + 800`（`0.612 / 0.415`）对比：
  - `mAP50 +0.04855`
  - `mAP50-95 +0.05587`
  - `person`: `0.468 / 0.227 -> 0.502 / 0.255`
  - `motorcycle`: `0.444 / 0.201 -> 0.522 / 0.243`
- 与当前 6 类 `YOLO11s RGB-only`（`0.57578 / 0.39398`）对比：
  - `mAP50 +0.08477`
  - `mAP50-95 +0.07689`
  - 注意：该对比仍不是最终公平主表，因为 RGB-only 还未按 `800 + Adam + 100 epoch + batch=20` 同配方重跑。

- 结果解读：
  - `800 + Adam` 高配方显著放大了 Light NIR 结构的收益，且提升不再只集中在中大目标。
  - `person` 与 `motorcycle` 同时超过 `RGB-NIR plain + Adam + 800`，说明 `BiFPN + Light NIR branch` 对小目标定位质量是正向的。
  - 当前主线可以暂定为 `YOLO11s + BiFPN-only + Light NIR branch`，不需要回到 `ResidualQualityAwareFusionV2` 方向。
  - 下一步必须补同配方 `YOLO11s RGB-only 6cls`，否则不能把 `RGB-NIR 优于 RGB-only` 写成论文主表结论。

#### 6 类 `rgbnir plain`（`imgsz=800`, `100 epoch`）结果

- SGD 版 run：`iddaw-yolo11n-rgbnir-plain-6cls-personmerge`
- SGD 版结果目录：`runs/IDD_AW/iddaw-yolo11n-rgbnir-plain-6cls-personmerge`
- SGD 版 W\&B run：`5es3st3o`
- SGD 版 `best.pt`：
  - `Precision = 0.643`
  - `Recall = 0.539`
  - `mAP50 = 0.572`
  - `mAP50-95 = 0.385`

- Adam 版 run：`iddaw-yolo11n-rgbnir-plain-6cls-personmerge2`
- Adam 版结果目录：`runs/IDD_AW/iddaw-yolo11n-rgbnir-plain-6cls-personmerge2`
- Adam 版 W\&B run：`jvhe9jh7`
- Adam 版 `best.pt`：
  - `Precision = 0.715`
  - `Recall = 0.544`
  - `mAP50 = 0.612`
  - `mAP50-95 = 0.415`
- Adam 版主要类别表现：
  - `person`: `mAP50 = 0.468`, `mAP50-95 = 0.227`
  - `motorcycle`: `mAP50 = 0.444`, `mAP50-95 = 0.201`
  - `car`: `mAP50 = 0.865`, `mAP50-95 = 0.646`
  - `truck`: `mAP50 = 0.584`, `mAP50-95 = 0.427`
  - `bus`: `mAP50 = 0.638`, `mAP50-95 = 0.488`
  - `autorickshaw`: `mAP50 = 0.673`, `mAP50-95 = 0.498`

- Adam 相对 SGD 的变化：
  - `Precision +0.072`
  - `Recall +0.005`
  - `mAP50 +0.040`
  - `mAP50-95 +0.030`

- Adam 与 6 类 `YOLO11s RGB-only`（`0.57578 / 0.39398`）对比：
  - `mAP50 +0.03622`
  - `mAP50-95 +0.02102`

- 结果解读：
  - 在默认 6 类口径和 `800×800` 输入下，将优化器从 `SGD` 切换为 `Adam` 后，双流 plain 的总体指标出现了明确提升，并首次稳定超过当前 6 类 `YOLO11s RGB-only`。
  - 这次提升主要来自整体优化质量改善和中大目标类提升，其中 `truck`、`bus`、`car` 的提升更明显。
  - 对最关键的小目标类别，`person` 相比 SGD 仍有小幅提升（`0.450 -> 0.468`），但 `motorcycle` 基本持平（`0.447 -> 0.444`），说明 `Adam` 并没有从根本上解决最小目标类别的检测短板。
  - 当前结论是：`6 类 + 800×800 + RGBNIR plain + Adam` 已经足以把 plain 路线推到一个比 6 类强 RGB-only 更高的水平，但如果目标是继续改善 `motorcycle` 等小目标，后续仍需要更强结构或更高分辨率策略。

## 7. 当前可直接引用的结论

- 当前 RGB-only 基线中，`YOLO11s RGB-only` 已成为更强单模基线：
  - `YOLO11s RGB-only`（80 epoch）：`mAP50 = 0.53782`，`mAP50-95 = 0.36051`
  - `YOLO11n RGB-only`（50 epoch）：`mAP50 = 0.43404`，`mAP50-95 = 0.27339`
- 在 6 类 `person+rider` 合并口径下，`YOLO11s RGB-only` 进一步提升到：
  - `mAP50 = 0.57578`
  - `mAP50-95 = 0.39398`
- 在当前已完成的 `YOLO11s RGB-NIR` 6 类对照里，`BiFPN-only + Light NIR branch` 是目前整体最强的一条已完成路线：
  - `BiFPN-only + Light NIR branch`（100 epoch, `800 + Adam + batch=20`）：`mAP50 = 0.66055`，`mAP50-95 = 0.47087`
  - `BiFPN-only + Light NIR branch`（100 epoch, `640 + SGD + batch=32`）：`mAP50 = 0.60467`，`mAP50-95 = 0.41390`
  - `Proposed-Lite`（70 epoch）：`mAP50 = 0.59353`，`mAP50-95 = 0.39896`
  - `BiFPN-only`（50 epoch）：`mAP50 = 0.58269`，`mAP50-95 = 0.39173`
  - `YOLO11s RGB-only 6 类`（80 epoch）：`mAP50 = 0.57578`，`mAP50-95 = 0.39398`
- `BiFPN-only + Light NIR branch` 相对 `Proposed-Lite` 的主要变化：
  - 总体 `mAP50 +0.01114`
  - 总体 `mAP50-95 +0.01494`
  - `person` 从 `0.409 / 0.193` 回升到 `0.431 / 0.207`
  - `motorcycle` 从 `0.417 / 0.189` 变为 `0.434 / 0.183`
- 当前小目标结论：
  - 轻量 NIR 分支能恢复 `person / motorcycle` 的部分 `mAP50`，但 `motorcycle` 的 `mAP50-95` 仍偏弱。
  - 当前瓶颈已经从“是否要复杂跨模态注意力”转向“小目标高 IoU 定位质量”。
- 在相同 6 类口径和 `800×800` 输入下，`RGB-NIR plain + Adam` 当前达到：
  - `mAP50 = 0.612`
  - `mAP50-95 = 0.415`
  - `BiFPN-only + Light NIR branch` 在同为 `800 + Adam` 的高配方下达到 `mAP50 = 0.66055`、`mAP50-95 = 0.47087`
  - 相对 `RGB-NIR plain + Adam + 800`：`mAP50 +0.04855`，`mAP50-95 +0.05587`
- 当前正式 `YOLO11n Proposed` 为 `full_proposed_residual_v2`：
  - `mAP50 = 0.52312`
  - `mAP50-95 = 0.34447`
- 当前 `YOLO11s` 版 RGB-NIR 结果中：
  - `YOLO11s BiFPN-only`：`mAP50 = 0.53924`，`mAP50-95 = 0.35857`
  - `YOLO11s Proposed`：`mAP50 = 0.54620`，`mAP50-95 = 0.35699`
- 与 `YOLO11s RGB-only` 对比：
  - 二者都在 `mAP50` 上略高于 `RGB-only`，但 `mAP50-95` 都略低
  - 当前尚不足以证明在更强 RGB-only 基线下形成全面稳定优势
- 但就当前 6 类并行实验来看：
  - `RGB-NIR plain` 在 `Adam + 800×800` 下已经超过 6 类 `YOLO11s RGB-only`
  - `BiFPN-only + Light NIR branch` 在 `Adam + 800×800` 下进一步扩大了 RGB-NIR 优势
  - 当前唯一缺口是同配方 `YOLO11s RGB-only 6cls` 尚未补齐，因此最终主表仍需等待公平基线结果
- 外部 RGB 单模基线 `RT-DETR-R18 RGB-only` 已成功接入并完成 `50 epoch`：
  - `mAP50 = 0.32081`
  - `mAP50-95 = 0.18771`
- 当前 70 epoch 补训练趋势表明：
  - `rgbnir plain` 继续上升
  - `bifpn_only` 继续上升
  - `RT-DETR-R18` 补到 `70 epoch` 后也有明显提升，但仍未超过 `YOLO11 RGB-only`

## 8. 下一步执行方案

- 第一优先级：补一条同配方 `YOLO11s RGB-only 6cls` 高配方基线，固定 `imgsz=800`、`Adam`、`100 epoch`、`batch=20`，用于论文主表公平比较。
- 外部中断记录：
  - mode：`rgb_yolo11s_6cls_personmerge`
  - run：`iddaw-yolo11s-rgb-6cls-personmerge3`
  - 远端 pid：`5406`
  - 日志：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/rgb_yolo11s_6cls_personmerge_e100_20260424_190154.stdout.log`
  - meta：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/rgb_yolo11s_6cls_personmerge_e100_20260424_190154.meta`
  - W&B run：`7o4m6xnq`
  - W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/7o4m6xnq`
  - 启动配方：`imgsz=800`、`optimizer=Adam`、`batch=20`、`epochs=100`、`WANDB_ENABLED=1`
  - 状态：外部中断，远端已无该训练进程；日志最后可见约 `16/100`，不纳入公平主表。
- 外部中断记录：
  - mode：`rgb_yolo11s_6cls_personmerge`
  - run：`iddaw-yolo11s-rgb-6cls-personmerge4`
  - 远端 pid：`6536`
  - 日志：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/rgb_yolo11s_6cls_personmerge_e100_20260424_195713.stdout.log`
  - meta：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/rgb_yolo11s_6cls_personmerge_e100_20260424_195713.meta`
  - W&B run：`20zojg1d`
  - W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/20zojg1d`
  - 启动配方：`imgsz=800`、`optimizer=Adam`、`batch=20`、`epochs=100`、`WANDB_ENABLED=1`
  - 状态：外部中断，远端已无该训练进程；日志最后可见约 `14/100`，不纳入公平主表。
- 外部中断/失败记录：
  - mode：`rgb_yolo11s_6cls_personmerge`
  - run：`iddaw-yolo11s-rgb-6cls-personmerge5`
  - 远端 pid：`6662`
  - 日志：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/rgb_yolo11s_6cls_personmerge_e100_20260424_224647.stdout.log`
  - meta：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/rgb_yolo11s_6cls_personmerge_e100_20260424_224647.meta`
  - W&B run：`a6ve6pep`
  - W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/a6ve6pep`
  - 启动配方：`imgsz=800`、`optimizer=Adam`、`batch=20`、`epochs=100`、`WANDB_ENABLED=1`
  - 状态：运行到 `5/100` 后失败，不纳入公平主表。
  - 错误：`RuntimeError: CUDA error: unspecified launch failure`
  - 伴随现象：随后 `nvidia-smi` 报 `Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error`，判断为 GPU/驱动层异常，不是单纯 batch/OOM 问题。
- 当前处理建议：
  - 先恢复 `4_3090` 的 GPU 可见性，至少要求 `nvidia-smi` 正常显示 RTX 3090 后再重启训练。
  - GPU 恢复后继续同配方从头跑：`rgb_yolo11s_6cls_personmerge`, `imgsz=800`, `Adam`, `batch=20`, `100 epoch`。
  - 如果 GPU 恢复后仍复现 launch failure，再把 batch 同步降到 `16` 作为稳定性排查，而不是直接改变优化器或输入尺寸。
- 第二优先级：如果 RGB-only 高配方仍低于 `BiFPN-only + Light NIR branch`，则把这两条作为最终主表核心对照，并围绕 `person / motorcycle` 展开小目标分析。
- 第三优先级：如果显存和时间允许，再补 `bifpn_only_light_nir_yolo11s_6cls_personmerge` 的 `imgsz=800 + SGD` 消融，用来区分收益来自结构、分辨率还是优化器。
- 暂不建议继续推进 `ResidualQualityAwareFusionV2` 或 `Proposed-Lite + light NIR`，因为当前结果已经说明更轻的 NIR 深层分支比更复杂的残差质量感知更稳。
- 后续论文主线建议写成：`YOLO11s + BiFPN + Light NIR branch`，贡献点聚焦于多尺度融合与 NIR 深层语义轻量化，而不是残差质量感知注意力。
