# 当前模型训练与状态总表

本文档整理当前 `yolov11-rgbnir-formal` 工程下所有已接入模型的：

1. 训练命令
2. 模型的具体实现方法
3. 训练参数与运行设置
4. 训练日志与结果存放位置
5. 当前最新训练状态及结果

更新时间：`2026-04-22`

## 1. 工程与数据根

- formal 工程目录：`E:\毕设\code\yolov11-rgbnir-formal`
- 远端工程目录：`/home/lym/lvyanhu/code/yolov11-rgbnir-formal`
- 当前正式数据根：`/home/lym/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir`
- 数据协议：
  - 数据集：`IDD-AW all-weather`
  - 模态：`RGB / NIR`
  - 类别数：`7`
  - 类别：`person, rider, motorcycle, car, truck, bus, autorickshaw`

## 2. 统一训练入口

### 2.1 本地/远端统一 Python 入口

```bash
python scripts/iddaw/run_experiment.py --mode <mode> --task train --epochs <epochs> --device 0
```

### 2.2 远端后台训练入口

```bash
bash scripts/iddaw/launch_nohup_train.sh <mode> <epochs> 0
```

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
| `decision_fusion` | `formal_rgbnir/decision_fusion.py` | 离线结果级融合：读取 `rgb` 与 `nir` 权重，推理后做 batched NMS 融合 |

## 4. 统一训练参数与设置

### 4.1 通用正式设置

- 输入尺寸：`640`
- 优化器：`SGD`
- 数据缓存：`cache=ram`
- `close_mosaic=5`
- 设备：`device=0`
- 项目目录：`runs/IDD_AW`

### 4.2 模式级差异

| 模式 | 输入模态 | `use_simotm` | `channels` | batch | workers | 正式主口径 |
| --- | --- | --- | --- | --- | --- | --- |
| `rgb` | `visible/train,val` | `BGR` | `3` | `96` | `12` | `50 epoch` |
| `rgb_yolo11s` | `visible/train,val` | `BGR` | `3` | `48` | `12` | 已完成 `80 epoch` |
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
| `bifpn_only_yolo11s` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | `YOLO11s BiFPN-only`，已完成 `70 epoch` |

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
| `nir` | `iddaw-yolo11n-nir2` | `50 epoch` | 已完成 | `0.57803` | `0.36977` | `0.40328` | `0.24597` | 官方 YOLO11 Gray/NIR 基线 |
| `rgbnir` | `iddaw-yolo11n-rgbnir-plain2` | `50 epoch` | 已完成 | `0.63680` | `0.44948` | `0.48136` | `0.30476` | 双流 plain baseline |
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

## 7. 当前可直接引用的结论

- 当前 `70 epoch` 趋势测试下，`bifpn_only` 仍是现有最强内部路线：
  - `mAP50 = 0.540`
  - `mAP50-95 = 0.354`
- 当前 RGB-only 基线中，`YOLO11s RGB-only` 已成为更强单模基线：
  - `YOLO11s RGB-only`（80 epoch）：`mAP50 = 0.53782`，`mAP50-95 = 0.36051`
  - `YOLO11n RGB-only`（50 epoch）：`mAP50 = 0.43404`，`mAP50-95 = 0.27339`
- 当前正式 `YOLO11n Proposed` 为 `full_proposed_residual_v2`：
  - `mAP50 = 0.52312`
  - `mAP50-95 = 0.34447`
- 当前 `YOLO11s` 版 RGB-NIR 结果中：
  - `YOLO11s BiFPN-only`：`mAP50 = 0.53924`，`mAP50-95 = 0.35857`
  - `YOLO11s Proposed`：`mAP50 = 0.54620`，`mAP50-95 = 0.35699`
- 与 `YOLO11s RGB-only` 对比：
  - 二者都在 `mAP50` 上略高于 `RGB-only`，但 `mAP50-95` 都略低
  - 当前尚不足以证明在更强 RGB-only 基线下形成全面稳定优势
- 外部 RGB 单模基线 `RT-DETR-R18 RGB-only` 已成功接入并完成 `50 epoch`：
  - `mAP50 = 0.32081`
  - `mAP50-95 = 0.18771`
- 当前 70 epoch 补训练趋势表明：
  - `rgbnir plain` 继续上升
  - `bifpn_only` 继续上升
  - `RT-DETR-R18` 补到 `70 epoch` 后也有明显提升，但仍未超过 `YOLO11 RGB-only`
