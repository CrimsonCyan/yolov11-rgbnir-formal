# 当前模型训练与状态总表

本文档整理当前 `yolov11-rgbnir-formal` 工程下所有已接入模型的：

1. 训练命令
2. 模型的具体实现方法
3. 训练参数与运行设置
4. 训练日志与结果存放位置
5. 当前最新训练状态及结果

更新时间：`2026-04-27`

## 1. 工程与数据根

- formal 工程目录：`E:\毕设\code\yolov11-rgbnir-formal`
- 远端工程目录：`/data1/lvyanhu/code/yolov11-rgbnir-formal`
- 当前正式数据根：`/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge`
- 7 类历史数据根：`/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir`
- 6 类数据根：`/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge`
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
- `LR0` 可选覆盖初始学习率；若不设置则使用 Ultralytics 默认学习率
- `COS_LR=1` 可选启用 Ultralytics cosine learning rate schedule；默认关闭
- `CLOSE_MOSAIC=20`，默认在最后 20 个 epoch 关闭 mosaic；可通过环境变量覆盖
- 自动生成 `stdout.log / pid / meta` 三类远端日志文件
- 自动维护 `latest_<mode>.*` 软链接，便于追踪当前最新 run

### 2.3 从中断 checkpoint 原样恢复训练

当前 formal 工程的 `resume_ckpt` 使用 Ultralytics 原生 `resume=True`，用于异常中断后的原样恢复。示例：

```bash
bash scripts/iddaw/launch_nohup_train.sh rgbnir 70 0 /data1/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/iddaw-yolo11n-rgbnir-plain2/weights/last.pt
```

说明：

- 这里的 `resume_ckpt` 必须指向待恢复 run 的 `weights/last.pt`。
- 原生 resume 会恢复 checkpoint 中保存的 optimizer、EMA、start epoch、原始训练参数和 mosaic 关闭阶段。
- 入口传入的 `epochs / optimizer / lr0 / cos_lr` 不用于重新定义训练计划；Ultralytics 原生 resume 只允许覆盖 `imgsz / batch / device / close_mosaic` 等少量参数。
- 若要在已完成 checkpoint 上额外微调若干 epoch，应另开新训练/微调流程，不再用异常恢复入口混用。

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
| `nir_yolo11s_6cls_personmerge` | `configs/models/yolo11s_nir_6cls_personmerge.yaml` | `YOLO11s` 单模 NIR 检测器，6 类 personmerge 口径，用于与 RGB-only / RGB-NIR 统一规模比较 |
| `rgbnir` | `configs/models/yolo11n_rgbnir_midfusion_plain.yaml` | 双流 RGB/NIR backbone，`P3/P4/P5` 同尺度 `Concat`，之后 `SPPF + C2PSA + YOLO head` |
| `rgbnir_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_midfusion_plain_6cls_personmerge.yaml` | `YOLO11s` 双流 RGB/NIR plain mid-fusion，`P3/P4/P5` 同尺度 `Concat`，作为不带 BiFPN/Light NIR/OA 的统一 plain baseline |
| `input_fusion` | `configs/models/yolo11n_rgbnir_input_fusion.yaml` | 4 通道输入级融合，`RGB(3)+NIR(1)` 直接输入单 backbone |
| `light_gate` | `configs/models/yolo11n_rgbnir_midfusion_gate.yaml` | 双流 backbone，`P3/P4/P5` 用 `ConcatGate` 代替 plain concat |
| `bifpn_only` | `configs/models/yolo11n_rgbnir_bifpn_only.yaml` | 双流 backbone + plain mid-fusion，neck 换成 `BiFPN(256, repeat=2)` |
| `attention_only` | `configs/models/yolo11n_rgbnir_attention_only.yaml` | 双流 backbone，`P3/P4/P5` 用 `QualityAwareFusion`，neck/head 保持 plain |
| `full_proposed` | `configs/models/yolo11n_rgbnir_full_proposed.yaml` | `QualityAwareFusion + BiFPN` 组合版历史路线 |
| `full_proposed_residual` | `configs/models/yolo11n_rgbnir_full_proposed_residual.yaml` | `ResidualQualityAwareFusion + BiFPN` 组合版历史路线 |
| `full_proposed_residual_v2` | `configs/models/yolo11n_rgbnir_full_proposed_residual_v2.yaml` | 当前正式 `Proposed`：`ResidualQualityAwareFusionV2 + BiFPN` |
| `bifpn_only_yolo11s` | `configs/models/yolo11s_rgbnir_bifpn_only.yaml` | `YOLO11s` 版双流 `BiFPN-only` |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_only_light_nir_6cls_personmerge.yaml` | `YOLO11s` 版 `BiFPN-only + Light NIR branch`：保留 `P3` NIR 分支，压缩 `P4/P5` NIR 语义通道并投影回融合尺度后做 plain concat |
| `bifpn_only_light_nir_p2_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_only_light_nir_p2_6cls_personmerge.yaml` | `YOLO11s` 版 `BiFPN-only + Light NIR branch + P2 head`：保持三尺度 BiFPN 不变，额外加入 stride=4 的 P2 小目标检测分支 |
| `bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_6cls_personmerge.yaml` | `YOLO11s` 版 `BiFPN-only + Light NIR branch + true P2-P5 BiFPN`：将 P2 输入纳入 BiFPN 双向融合，检测头只保留每尺度直接细化 |
| `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_c256_6cls_personmerge.yaml` | `YOLO11s` 版 `true P2-P5 BiFPN c256`：基于 true P2-P5，将 `BiFPNP2P5` 宽度设为 `256`，并将四个 Detect 前 refine head 的通道字面量设为 `256` |
| `bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oagate_6cls_personmerge.yaml` | `YOLO11s` 版 `BiFPN-only + Light NIR branch + true P2-P5 BiFPN + Object-aware NIR gate`：仅在 `P2/P3` RGB-NIR 融合处用轻量 OA gate 调制 NIR，`P4/P5` 保持 plain concat |
| `bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oagate_c256_6cls_personmerge.yaml` | `YOLO11s` 版 `true P2-P5 BiFPN + OA gate c256`：保留 `P2/P3` OA gate，同时将 `BiFPNP2P5` 宽度和四个 refine head 通道字面量设为 `256` |
| `bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_reflect_c256_6cls_personmerge.yaml` | `YOLO11s` 版 `true P2-P5 BiFPN + OA-Reflect gate c256`：以 plain c256 为基线，仅在 `P2/P3` 用 luminance/reflection/object-prior 三路 gate 调制 NIR，`P4/P5` 保持 plain concat |
| `bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_fg_c256_6cls_personmerge.yaml` | `YOLO11s` 版 `true P2-P5 BiFPN + foreground-supervised OA-Reflect c256`：沿用 OA-Reflect 结构，仅在训练时用 GT box 栅格化前景 mask 对 P2/P3 object-prior gate 加轻量 BCE 约束 |
| `bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_ms_softprior_c256_6cls_personmerge.yaml` | `YOLO11s` 版 `true P2-P5 BiFPN + multi-scale OA soft-prior c256`：P2/P3 使用 `3/5/7` 多尺度 NIR smooth 做 luminance/reflectance 分解，并对 object-prior gate 加中心高边缘低的 soft box 约束 |
| `rgbnir_light_nir_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_light_nir_6cls_personmerge.yaml` | `YOLO11s` 版 `RGB-NIR + Light NIR branch`：复用 Light NIR 分支，保留普通 YOLO neck/head，不使用 BiFPN，用于隔离 BiFPN 增益 |
| `full_proposed_residual_v2_yolo11s` | `configs/models/yolo11s_rgbnir_full_proposed_residual_v2.yaml` | `YOLO11s` 版 `ResidualQualityAwareFusionV2 + BiFPN` |
| `proposed_lite_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_proposed_lite_p34_6cls_personmerge.yaml` | `YOLO11s` 版 `Proposed-Lite`：`P3/P4` 用 `ResidualQualityAwareFusionV2`，`P5` 回退为 `Concat`，之后进入 `BiFPN` |
| `proposed_lite_light_nir_yolo11s_6cls_personmerge` | `configs/models/yolo11s_rgbnir_proposed_lite_light_nir_p34_6cls_personmerge.yaml` | `YOLO11s` 版 `Proposed-Lite + Light NIR branch`：保持 `P3` NIR 分支，压缩 `P4/P5` NIR 语义通道后再投影回融合尺度 |
| `decision_fusion` | `formal_rgbnir/decision_fusion.py` | 离线结果级融合：读取 `rgb` 与 `nir` 权重，推理后做 batched NMS 融合 |

## 4. 统一训练参数与设置

### 4.1 通用正式设置

- 输入尺寸：`640`
- 优化器：`SGD`
- 数据缓存：`cache=ram`
- `close_mosaic=20`
- 设备：`device=0`
- 项目目录：`runs/IDD_AW`
- 脚本默认验证频率：`val_interval=1`
- 脚本默认 W\&B console：`WANDB_CONSOLE=off`
- 统一入口支持可选覆盖：`--optimizer`、`--batch`、`--lr0`、`--cos-lr`
- 历史结果若未单独说明，默认仍按当时配置 `close_mosaic=10` 解释；`2026-04-25` 之后的新训练默认切到 `20`

### 4.2 模式级差异

| 模式 | 输入模态 | `use_simotm` | `channels` | batch | workers | 正式主口径 |
| --- | --- | --- | --- | --- | --- | --- |
| `rgb` | `visible/train,val` | `BGR` | `3` | `96` | `12` | `50 epoch` |
| `rgb_yolo11s` | `visible/train,val` | `BGR` | `3` | `48` | `12` | 已完成 `80 epoch` |
| `rgb_yolo11s_6cls_personmerge` | `visible/train,val` | `BGR` | `3` | `48` | `12` | 已完成 `80 epoch` |
| `rgb_rtdetr` | `visible/train,val` | `BGR` | `3` | `32` | `10` | `50 epoch`，现补到 `70` 中 |
| `nir` | `nir/train,val` | `Gray` | `1` | `96` | `12` | `50 epoch` |
| `nir_yolo11s_6cls_personmerge` | `nir/train,val` | `Gray` | `1` | `20` | `10` | 新增待训练；YOLO11s NIR-only 统一基线 |
| `rgbnir` | paired `visible + nir` | `RGBNIR` | `4` | `48` | `10` | `50 epoch`，现补到 `70` |
| `rgbnir_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 新增待训练；YOLO11s RGB-NIR plain 统一基线 |
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
| `bifpn_only_light_nir_p2_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 已通过冒烟；小目标 P2 四尺度检测头 |
| `bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 已完成 `100 epoch`；true P2-P5 BiFPN |
| `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 已完成 `100 epoch`；true P2-P5 BiFPN c256 |
| `bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 已完成 `100 epoch`；基于 true P2-P5 BiFPN，仅在 `P2/P3` 加 OA gate |
| `bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 已通过 Ultralytics 原生 `resume=True` 完成 `100 epoch`；true P2-P5 BiFPN + P2/P3 OA gate c256 |
| `bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 已完成 `100 epoch`；true P2-P5 BiFPN + P2/P3 OA-Reflect c256 |
| `bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 已完成 `100 epoch`；foreground-supervised OA-Reflect c256 |
| `bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `20` | `10` | 新增待冒烟；multi-scale OA soft-prior c256 |
| `rgbnir_light_nir_yolo11s_6cls_personmerge` | paired `visible + nir` | `RGBNIR` | `4` | `24` | `10` | 已完成 `100 epoch`；Light NIR plain baseline |
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
/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw
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
/data1/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW
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
ssh lyh "tail -f /data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/latest_rgbnir.stdout.log"
```

查看 BiFPN-only 最新日志：

```bash
ssh lyh "tail -f /data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/latest_bifpn_only.stdout.log"
```

查看 RT-DETR 最新日志：

```bash
ssh lyh "tail -f /data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/latest_rgb_rtdetr.stdout.log"
```

查看 70 epoch 续训队列总日志：

```bash
ssh lyh "tail -f /data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/resume70_queue.stdout.log"
```

## 6. 当前最新训练状态与结果

### 6.1 已完成的正式主线结果

| 模式 | 最新正式 run | 训练口径 | 当前状态 | P | R | mAP50 | mAP50-95 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rgb` | `iddaw-yolo11n-rgb2` | `50 epoch` | 已完成 | `0.58182` | `0.41056` | `0.43404` | `0.27339` | 官方 YOLO11 RGB-only 基线 |
| `rgb_yolo11s` | `iddaw-yolo11s-rgb7` | `80 epoch` | 已完成 | `0.69330` | `0.48334` | `0.53782` | `0.36051` | 官方 YOLO11s RGB-only 基线 |
| `rgb_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgb-6cls-personmerge8` | `100 epoch, imgsz=800, Adam, batch=20` | 已完成 | `0.70245` | `0.55279` | `0.61056` | `0.42368` | 6 类口径高配方 RGB-only 公平基线，`rider -> person` |
| `bifpn_only_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge2` | `50 epoch` | 已完成 | `0.69297` | `0.51787` | `0.58269` | `0.39173` | 6 类口径下当前已完成的 YOLO11s BiFPN-only 对照 |
| `bifpn_only_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge4` | `100 epoch, imgsz=800, Adam, batch=20, close_mosaic=20` | 已完成 | `0.76313` | `0.58431` | `0.65825` | `0.47101` | 原始对称双分支 BiFPN-only 高配方，close_mosaic=20 后总体 `mAP50-95` 追平 Light NIR |
| `proposed_lite_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-proposed-lite-p34-6cls-personmerge3` | `70 epoch` | 已完成 | `0.76323` | `0.50914` | `0.59353` | `0.39896` | 6 类口径下 `P3/P4` 质量感知 + `P5 plain concat` 的 Proposed-Lite |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge4` | `100 epoch, batch=32` | 已完成 | `0.71556` | `0.55095` | `0.60467` | `0.41390` | 当前 `YOLO11s + RGB-NIR + 640 + SGD` 最强结果，Light NIR branch |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge5` | `100 epoch, imgsz=800, Adam, batch=20` | 已完成 | `0.76181` | `0.60168` | `0.66055` | `0.47087` | 高配方 Light NIR branch，效率更优的主线候选 |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge6` | `100 epoch, imgsz=800, AdamW, lr0=0.001, batch=20, close_mosaic=20` | 已完成 | `0.75772` | `0.57149` | `0.64335` | `0.46029` | AdamW/lr0 消融；低于 Adam 主线 |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge7` | `100 epoch, imgsz=800, AdamW, lr0=0.001, batch=20, close_mosaic=10` | 已完成 | `0.72128` | `0.58693` | `0.63484` | `0.44939` | close_mosaic=10 消融；低于 close_mosaic=20 |
| `rgbnir_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-light-nir-6cls-personmerge` | `100 epoch, imgsz=800, AdamW, lr0=0.001, batch=20, close_mosaic=10` | 已完成 | `0.73076` | `0.55552` | `0.61771` | `0.42634` | Light NIR plain baseline，不带 BiFPN |
| `bifpn_only_light_nir_p2_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge4` | `100 epoch, imgsz=800, Adam, batch=20, device=0,1` | 已完成 | `0.75923` | `0.60991` | `0.67259` | `0.47049` | P2 检测头小目标增强，`person/motorcycle` 提升，但总体 `mAP50-95` 与无 P2 持平略低 |
| `bifpn_only_light_nir_p2_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge5` | `100 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.72686` | `0.58771` | `0.64462` | `0.44502` | P2 head 同配方复核；明显低于 close_mosaic=20 版本，不作为主结果 |
| `bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge3` | `100 epoch, imgsz=640, Adam, batch=20, close_mosaic=20` | 已完成 | `0.76438` | `0.58118` | `0.66597` | `0.45832` | true P2-P5 BiFPN 单卡结果；`800 + batch20` 单卡 OOM，因此不能与 800 高配方直接公平比较 |
| `bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge5` | `150 epoch, imgsz=800, AdamW, lr0=0.001, cos_lr=True, batch=20, device=0,1` | 已完成 | `0.71407` | `0.56201` | `0.61474` | `0.42033` | true P2-P5 BiFPN 高分辨率稳定配方；明显低于 P2 head / Light NIR 主线，暂不晋级 |
| `bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge6` | `150 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.73999` | `0.55773` | `0.63204` | `0.44113` | true P2-P5 BiFPN 的 Adam/lr0/close_mosaic=15 复核；优于 AdamW+cos，但仍低于 P2 head / Light NIR 主线 |
| `bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-6cls-personmerge2` | `100 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.70032` | `0.57634` | `0.63013` | `0.43517` | true P2-P5 + P2/P3 OA gate 诊断实验；小目标未改善，不建议继续强化 true P2-P5 |
| `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge` | `100 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.68729` | `0.59268` | `0.63579` | `0.43961` | true P2-P5 BiFPN c256；mAP50 略升但 mAP50-95 低于 c192 Adam 版，未证明加宽有效 |
| `bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge` | `100 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.72876` | `0.56793` | `0.63061` | `0.43168` | 原生 `resume=True` 从 epoch 91 恢复到 100；低于 plain c256，OA gate c256 未显示优势 |
| `bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge3` | `100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.74477` | `0.62225` | `0.68654` | `0.47814` | `lr0=0.01` 复训显著提升，曾为阶段最强；后续 plain c256 同配方略高 |
| `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge2` | `100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.72568` | `0.62905` | `0.69135` | `0.47976` | plain c256 同配方对照；略优于 OA gate c256 且更轻，说明当前主要收益来自 true P2-P5 c256 + lr0=0.01 |
| `bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-fg-c256-6cls-personmerge3` | `100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1` | 已完成 | `0.71942` | `0.61375` | `0.67895` | `0.47658` | foreground mask 弱监督能让 `fg_gate_loss` 下降，但整体低于 plain c256，暂不晋级 |
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
- 与同配方 6 类 `YOLO11s RGB-only` 高配方基线（`0.61056 / 0.42368`）对比：
  - `mAP50 +0.04999`
  - `mAP50-95 +0.04719`
  - `Precision +0.05936`
  - `Recall +0.04889`

- 结果解读：
  - `800 + Adam` 高配方显著放大了 Light NIR 结构的收益，且提升不再只集中在中大目标。
  - `person` 与 `motorcycle` 同时超过 `RGB-NIR plain + Adam + 800`，说明 `BiFPN + Light NIR branch` 对小目标定位质量是正向的。
  - 当前主线可以暂定为 `YOLO11s + BiFPN-only + Light NIR branch`，不需要回到 `ResidualQualityAwareFusionV2` 方向。
  - 同配方 `YOLO11s RGB-only 6cls` 已补齐，当前可以把 `BiFPN-only + Light NIR branch` 与 RGB-only 的公平高配方对比作为论文主表候选。

#### 6 类 `YOLO11s RGB-only` 高配方确认结果（`imgsz=800`, `Adam`）

- 当前 run：`iddaw-yolo11s-rgb-6cls-personmerge8`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgb-6cls-personmerge8`
- 日志：`/home/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/rgb_yolo11s_6cls_personmerge_e100_20260425_161538.stdout.log`
- W&B run：`0qo46dus`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/0qo46dus`
- 运行配置：
  - `imgsz=800`
  - `optimizer=Adam`
  - `batch=20`
  - `epochs=100`
  - `close_mosaic=10`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 训练完成状态：
  - `results.csv` 共 `100` 个 epoch
  - `best.pt` 与 `last.pt` 均已导出
  - `results.csv` 最优 epoch 为 `100`
- `best.pt` / epoch `100` 指标：
  - `Precision = 0.70245`
  - `Recall = 0.55279`
  - `mAP50 = 0.61056`
  - `mAP50-95 = 0.42368`
- 与同配方 `BiFPN-only + Light NIR branch`（`0.66055 / 0.47087`）对比：
  - `mAP50 -0.04999`
  - `mAP50-95 -0.04719`
- 结果解读：
  - 该结果补齐了此前缺失的公平高配方 RGB-only 基线。
  - 在相同 `6cls personmerge + YOLO11s + imgsz=800 + Adam + batch=20 + 100 epoch` 下，当前 RGB-NIR Light NIR 主线已经稳定超过 RGB-only。
  - 需要注意这条 RGB-only 仍使用历史默认 `close_mosaic=10`，而后续新训练默认切到 `close_mosaic=20`；如果 close_mosaic 增益继续明确，最终主表应以同一 close_mosaic 设置重新确认。

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

### 6.12 `YOLO11s BiFPN-only + Light NIR branch + P2 head` 待验证方向

- 新增 mode：`bifpn_only_light_nir_p2_yolo11s_6cls_personmerge`
- 新增配置：`configs/models/yolo11s_rgbnir_bifpn_only_light_nir_p2_6cls_personmerge.yaml`
- 结构目的：
  - 针对当前 `person / motorcycle` 小目标高 IoU 定位质量偏弱的问题，新增 stride=4 的 `P2` 检测分支
  - 保持现有 `P3/P4/P5 -> BiFPN(256, repeat=2)` 不变，避免同时改动 BiFPN 算子
  - 在 head 侧将 `P3` top-down 特征上采样到 `P2`，再与 RGB/NIR 浅层 `P2` 特征拼接细化
  - Detect 从原三尺度 `P3/P4/P5` 改为四尺度 `P2/P3/P4/P5`
- 实现边界：
  - 不回到 `ResidualQualityAwareFusionV2`
  - 不新增更深 backbone
  - 不改数据导出和类别映射
- 预期收益：
  - 提升 `person`、`motorcycle` 的召回和高 IoU 定位质量
  - 代价是 `imgsz=800` 下显存和训练时间上升；为保持高配方公平比较，正式训练 batch 固定为 `20`
- 当前正式 run：
  - run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge3`
  - 远端 pid：`12954`
  - 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2_yolo11s_6cls_personmerge_e100_20260425_183313.stdout.log`
  - meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2_yolo11s_6cls_personmerge_e100_20260425_183313.meta`
  - W&B run：`hdgwzcf4`
  - W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/hdgwzcf4`
  - 配方：`imgsz=800`、`optimizer=Adam`、`batch=20`、`epochs=100`、`close_mosaic=20`、`WANDB_ENABLED=1`
  - 启动检查：已进入 `2/100`，训练峰值显存约 `22.5G/24G`，当前未见 OOM、shape mismatch、NaN 或 W&B 登录问题
- 废弃 run：
  - `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge2` 使用 `batch=18` 启动后被手动停止，不纳入对照结果

### 6.13 `YOLO11s BiFPN-only + Light NIR branch + true P2-P5 BiFPN` 完成结果

- mode：`bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge`
- 配置：`configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_6cls_personmerge.yaml`
- 结构目的：
  - 将 `P2` 不只放到检测头，而是作为第四个尺度输入纳入 BiFPN 双向融合
  - BiFPN 输入为 `P2/P3/P4/P5`，Detect 输出 stride 为 `[4, 8, 16, 32]`
  - 保持 `Light NIR branch`，不回到残差质量感知融合路线
- 训练 run：
  - run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge3`
  - 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge_e100_20260426_142953.stdout.log`
  - W&B run：`nei0onrv`
  - W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/nei0onrv`
  - 配方：`imgsz=640`、`optimizer=Adam`、`lr0=0.01`、`batch=20`、`epochs=100`、`close_mosaic=20`、`device=0`
  - 备注：单卡尝试 `imgsz=800 + batch=20` 时 OOM，因此本轮实际降为 `640`；该结果不能与 `800` 高配方结果做严格公平比较
- best epoch：
  - epoch：`96`
  - `Precision = 0.76438`
  - `Recall = 0.58118`
  - `mAP50 = 0.66597`
  - `mAP50-95 = 0.45832`
- final epoch：
  - epoch：`100`
  - `Precision = 0.74373`
  - `Recall = 0.59813`
  - `mAP50 = 0.66329`
  - `mAP50-95 = 0.45457`
- best.pt 最终验证类别指标：
  - `person`: `mAP50 = 0.524`, `mAP50-95 = 0.264`
  - `motorcycle`: `mAP50 = 0.523`, `mAP50-95 = 0.233`
  - `car`: `mAP50 = 0.884`, `mAP50-95 = 0.669`
  - `truck`: `mAP50 = 0.596`, `mAP50-95 = 0.438`
  - `bus`: `mAP50 = 0.721`, `mAP50-95 = 0.561`
  - `autorickshaw`: `mAP50 = 0.750`, `mAP50-95 = 0.580`
- 结果分析：
  - 在 `640` 输入下，true P2-P5 BiFPN 已达到 `mAP50 = 0.66597`，接近 `800 + Adam` 的 Light NIR 主线 `0.66055`，说明 P2-P5 融合对目标发现能力有实际帮助。
  - `mAP50-95 = 0.45832` 低于 `800 + Adam` 的 Light NIR 主线 `0.47087` 和 P2 head 版本 `0.47049`，但这里分辨率不同，不能直接判定结构更弱。
  - 小目标类别上，`person/motorcycle` 的 `mAP50-95` 分别为 `0.264 / 0.233`，明显高于早期 Light NIR 主线记录中的 `0.207 / 0.183`，说明加入 P2 路线对小目标定位质量是有效方向。
  - 这轮训练暴露的主要问题不是结构是否可用，而是配方不统一：`Adam + lr0=0.01` 对 Adam 偏高，且 `640` 分辨率与此前高配方 `800` 不可比。
- 当前结论：
  - true P2-P5 BiFPN 值得继续验证，但下一轮必须使用双卡跑 `imgsz=800`，否则无法与现有高配方主线公平比较。
  - 下一轮不建议继续沿 `Adam + lr0=0.01`，应切到 `AdamW + lr0=0.001 + cos_lr=True`，降低高分辨率长训的震荡风险。

#### 双卡 `800 + AdamW + cos_lr` 完成结果

- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge5`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge_e150_20260426_161524.stdout.log`
- W&B run：`cj80como`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/cj80como`
- 配方：`imgsz=800`、`optimizer=AdamW`、`lr0=0.001`、`cos_lr=True`、`batch=20`、`epochs=150`、`close_mosaic=20`、`device=0,1`
- best epoch：
  - epoch：`115`
  - `Precision = 0.71407`
  - `Recall = 0.56201`
  - `mAP50 = 0.61474`
  - `mAP50-95 = 0.42033`
- final epoch：
  - epoch：`150`
  - `Precision = 0.76579`
  - `Recall = 0.53244`
  - `mAP50 = 0.61269`
  - `mAP50-95 = 0.41880`
- best.pt 最终验证类别指标：
  - `person`: `mAP50 = 0.475`, `mAP50-95 = 0.232`
  - `motorcycle`: `mAP50 = 0.483`, `mAP50-95 = 0.219`
  - `car`: `mAP50 = 0.856`, `mAP50-95 = 0.635`
  - `truck`: `mAP50 = 0.549`, `mAP50-95 = 0.392`
  - `bus`: `mAP50 = 0.619`, `mAP50-95 = 0.510`
  - `autorickshaw`: `mAP50 = 0.707`, `mAP50-95 = 0.528`
- 对比结论：
  - 相比 true P2-P5 的 `640 + Adam` 结果（`0.66597 / 0.45832`），本轮 `800 + AdamW + cos_lr` 反而下降到 `0.61474 / 0.42033`。
  - 相比 `P2 head + 800 + Adam`（`0.67259 / 0.47049`），本轮下降 `mAP50 -0.05785`、`mAP50-95 -0.05016`。
  - 相比 `Light NIR + 800 + Adam`（`0.66055 / 0.47087`），本轮下降 `mAP50 -0.04581`、`mAP50-95 -0.05054`。
  - 小目标类别没有兑现高分辨率收益：`person/motorcycle mAP50-95 = 0.232 / 0.219`，低于上一轮 true P2-P5 `640 + Adam` 的 `0.264 / 0.233`。
- 当前判断：
  - true P2-P5 BiFPN 在当前 `AdamW + lr0=0.001 + cos_lr` 配方下不晋级。
  - 这轮不能单独证明 true P2-P5 结构无效，因为同时改变了 `imgsz`、优化器、学习率策略和训练轮数；但它已经证明 `AdamW + cos_lr` 不是当前最稳的高分辨率配方。
  - 当前主线应回到 `P2 head` 或 `Light NIR + BiFPN-only`，而不是继续扩大 true P2-P5 BiFPN。

## 7. 当前可直接引用的结论

- 当前 RGB-only 基线中，`YOLO11s RGB-only` 已成为更强单模基线：
  - `YOLO11s RGB-only`（80 epoch）：`mAP50 = 0.53782`，`mAP50-95 = 0.36051`
  - `YOLO11n RGB-only`（50 epoch）：`mAP50 = 0.43404`，`mAP50-95 = 0.27339`
- 在 6 类 `person+rider` 合并口径下，`YOLO11s RGB-only` 高配方基线已经补齐：
  - `YOLO11s RGB-only`（100 epoch, `800 + Adam + batch=20`）：`mAP50 = 0.61056`
  - `YOLO11s RGB-only`（100 epoch, `800 + Adam + batch=20`）：`mAP50-95 = 0.42368`
- 在当前已完成的 `YOLO11s RGB-NIR` 6 类对照里，`BiFPN-only + Light NIR branch` 是目前整体最强的一条已完成路线：
  - `BiFPN-only + Light NIR branch`（100 epoch, `800 + Adam + batch=20`）：`mAP50 = 0.66055`，`mAP50-95 = 0.47087`
  - `BiFPN-only + Light NIR branch + P2 head`（100 epoch, `800 + Adam + batch=20, device=0,1`）：`mAP50 = 0.67259`，`mAP50-95 = 0.47049`
  - `BiFPN-only + Light NIR branch + true P2-P5 BiFPN`（100 epoch, `640 + Adam + batch=20`）：`mAP50 = 0.66597`，`mAP50-95 = 0.45832`
  - `BiFPN-only + Light NIR branch + true P2-P5 BiFPN`（150 epoch, `800 + AdamW + lr0=0.001 + cos_lr + batch=20, device=0,1`）：`mAP50 = 0.61474`，`mAP50-95 = 0.42033`
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
  - P2 系列对 `person / motorcycle` 有明确改善信号，但目前更可靠的是“P2 检测头”而不是真正把 P2 纳入 BiFPN 的 P2-P5 结构。
  - true P2-P5 BiFPN 在 `640 + Adam` 下有潜力信号，但在 `800 + AdamW + cos_lr` 下明显退化，当前不作为主线晋级。
  - 当前瓶颈已经从“是否要复杂跨模态注意力”转向“小目标高 IoU 定位质量”和“稳定高分辨率优化配方”；现阶段不应继续扩大 BiFPN 结构复杂度。
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
  - 同配方 `YOLO11s RGB-only 6cls` 已补齐：`mAP50 = 0.61056`，`mAP50-95 = 0.42368`
  - `BiFPN-only + Light NIR branch` 相对同配方 RGB-only：`mAP50 +0.04999`，`mAP50-95 +0.04719`
- 外部 RGB 单模基线 `RT-DETR-R18 RGB-only` 已成功接入并完成 `50 epoch`：
  - `mAP50 = 0.32081`
  - `mAP50-95 = 0.18771`
- 当前 70 epoch 补训练趋势表明：
  - `rgbnir plain` 继续上升
  - `bifpn_only` 继续上升
  - `RT-DETR-R18` 补到 `70 epoch` 后也有明显提升，但仍未超过 `YOLO11 RGB-only`

## 8. 下一步执行方案

- 第一优先级：停止把 true P2-P5 BiFPN 作为当前主线继续扩展。其 `800 + AdamW + cos_lr`、`800 + Adam + close_mosaic=15`、以及 `P2/P3 OA gate` 三组结果均未超过 `P2 head`、`Light NIR` 或原始对称 `BiFPN-only` 高配方。
- 下一步若只允许再跑一组，建议做“P2 head 同配方复核”：
  - mode：`bifpn_only_light_nir_p2_yolo11s_6cls_personmerge`
  - 远端：`ssh lyh`
  - 工程目录：`/data1/lvyanhu/code/yolov11-rgbnir-formal`
  - 数据根：`/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge`
  - 配方：`imgsz=800`、`optimizer=Adam`、`lr0=0.001`、`batch=20`、`epochs=100`、`close_mosaic=15`、`device=0,1`、`WANDB_ENABLED=1`
  - 目的：与当前 `P2-P5 + OA gate` 同配方对齐，验证 `P2 head` 的小目标收益是否稳定来自检测头本身。
  - 晋级标准：若总体 `mAP50-95` 继续接近或超过 `0.470`，且 `person/motorcycle` 明显优于 OA gate 的 `0.251 / 0.231`，则正式停止 true P2-P5 路线。
- 当前论文主线建议收敛为：
  - 总体主结果：`bifpn_only_light_nir_yolo11s_6cls_personmerge` 或原始 `bifpn_only_yolo11s_6cls_personmerge`，二者 `mAP50-95` 都约 `0.471`
  - 小目标增强结果：`bifpn_only_light_nir_p2_yolo11s_6cls_personmerge`，其 `mAP50 = 0.67259` 更高，适合支撑 P2 小目标改进
  - 不再把 `AdamW + cos_lr` 作为当前默认高配方；当前更稳的是 `800 + Adam + batch=20`，`close_mosaic` 重点在 `15/20` 内做少量复核。
- 第二优先级：若需要继续使用 OA 思想，下一步应迁移到 `P2 head` 版本的 `P2/P3` 融合处，而不是继续强化 true P2-P5 BiFPN。
- 已完成补齐：同配方 `YOLO11s RGB-only 6cls` 高配方基线已完成，run 为 `iddaw-yolo11s-rgb-6cls-personmerge8`，指标为 `mAP50 = 0.61056`、`mAP50-95 = 0.42368`。
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
- 第三优先级：若本轮 `P2` 对 `person / motorcycle` 没有明确提升，回退到 `BiFPN-only + Light NIR branch` 作为最终结构，不继续扩大四尺度 head。
- 第四优先级：如果显存和时间允许，再补 `bifpn_only_light_nir_yolo11s_6cls_personmerge` 的 `imgsz=800 + SGD` 消融，用来区分收益来自结构、分辨率还是优化器。
- 暂不建议继续推进 `ResidualQualityAwareFusionV2` 或 `Proposed-Lite + light NIR`，因为当前结果已经说明更轻的 NIR 深层分支比更复杂的残差质量感知更稳。
- 后续论文主线建议写成：`YOLO11s + BiFPN + Light NIR branch`，贡献点聚焦于多尺度融合与 NIR 深层语义轻量化，而不是残差质量感知注意力。


## 9. 2026-04-25 P2 双卡训练结果与下一步计划

### 9.1 `BiFPN-only + Light NIR branch + P2 head` 双卡正式结果

- mode：`bifpn_only_light_nir_p2_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge4`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge4`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2_yolo11s_6cls_personmerge_e100_20260425_212552.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2_yolo11s_6cls_personmerge_e100_20260425_212552.meta`
- W&B run：`pv63f0rw`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/pv63f0rw`
- 运行配置：
  - `imgsz=800`
  - `optimizer=Adam`
  - `batch=20`
  - `device=0,1`
  - `epochs=100`
  - `close_mosaic=20`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 模型规模：
  - unfused summary：`14,399,516` parameters / `60.17 GFLOPs`
  - fused summary：`14,380,236` parameters / `93.16 GFLOPs`（Ultralytics DDP 复验日志口径）
- 训练完成状态：
  - `100 epochs completed in 1.156 hours`
  - `results.csv` 共 `100` 个 epoch
  - `best.pt` 与 `last.pt` 均已导出并完成 optimizer strip
  - 双卡后每卡训练显存约 `11-14GB`，未复现单卡 `batch=20` 的 OOM

- `results.csv` 最优 epoch：
  - epoch `100`
  - `Precision = 0.75923`
  - `Recall = 0.60991`
  - `mAP50 = 0.67259`
  - `mAP50-95 = 0.47049`
- W&B summary：
  - `Precision = 0.75879`
  - `Recall = 0.60988`
  - `mAP50 = 0.67279`
  - `mAP50-95 = 0.47033`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.554`, `mAP50-95 = 0.290`
  - `motorcycle`: `mAP50 = 0.581`, `mAP50-95 = 0.280`
  - `car`: `mAP50 = 0.887`, `mAP50-95 = 0.692`
  - `truck`: `mAP50 = 0.592`, `mAP50-95 = 0.447`
  - `bus`: `mAP50 = 0.639`, `mAP50-95 = 0.496`
  - `autorickshaw`: `mAP50 = 0.784`, `mAP50-95 = 0.617`

### 9.2 与当前主线基线对比

- 与同配方 `BiFPN-only + Light NIR branch`（无 P2，`mAP50 = 0.66055`，`mAP50-95 = 0.47087`）对比：
  - 总体 `mAP50 +0.01204`
  - 总体 `mAP50-95 -0.00038`
  - `person`: `0.502 / 0.255 -> 0.554 / 0.290`
  - `motorcycle`: `0.522 / 0.243 -> 0.581 / 0.280`
- 与同配方 6 类 `YOLO11s RGB-only` 高配方基线（`mAP50 = 0.61056`，`mAP50-95 = 0.42368`）对比：
  - 总体 `mAP50 +0.06203`
  - 总体 `mAP50-95 +0.04681`
- 与 `RGB-NIR plain + Adam + 800`（`mAP50 = 0.612`，`mAP50-95 = 0.415`）对比：
  - 总体 `mAP50 +0.06059`
  - 总体 `mAP50-95 +0.05549`

### 9.3 结果解读

- P2 分支有效改善了小目标类别，`person` 与 `motorcycle` 的 `mAP50` 和 `mAP50-95` 都明显高于无 P2 的 Light NIR 主线。
- P2 没有带来总体 `mAP50-95` 的进一步提升；总体高 IoU 指标与无 P2 主线基本持平，且略低于无 P2 的 `0.47087`。
- P2 的收益主要来自小目标召回/检测能力，而不是全类别定位质量提升；`truck`、`bus` 等类别相对无 P2 主线下降，抵消了小目标增益。
- 单卡 `batch=20` 的 P2 run 在 epoch `88/100` 附近 OOM，双卡 `device=0,1` 能稳定完成，说明后续 P2 类四尺度 head 默认应使用双卡或降低 batch。

### 9.4 下一步执行方案

- 当前论文主线仍建议保持为 `YOLO11s + BiFPN-only + Light NIR branch`，因为它在接近最优总体精度的同时结构更轻。
- P2 作为“小目标增强分支”保留为消融/扩展实验，不建议直接替代主线结构，除非论文重点转向 `person / motorcycle` 小目标。
- 下一轮优先补齐 `bifpn_only_yolo11s_6cls_personmerge` 在 `imgsz=800 + Adam + batch=20 + close_mosaic=20` 下的正式结果，用来回答 close_mosaic=20 是否会让原始 BiFPN-only 接近 Light NIR。
- 如果继续优化小目标，优先方向不是再加更深 head，而是做更轻的 P2 或检测损失/分配策略消融，例如：减小 P2 通道、只对 P2 使用更轻 `C3k2`、或尝试更适合小目标的 label assignment / NMS 参数；这些需要单独阶段验证。
- 最终主表建议暂定比较顺序：
  - `YOLO11s RGB-only 6cls`
  - `RGB-NIR plain`
  - `BiFPN-only`
  - `BiFPN-only + Light NIR branch`
  - `BiFPN-only + Light NIR branch + P2 head`

## 10. 2026-04-26 原始 BiFPN-only close_mosaic=20 正式结果

### 10.1 `YOLO11s BiFPN-only` 高配方结果

- mode：`bifpn_only_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge4`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-6cls-personmerge4`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_yolo11s_6cls_personmerge_e100_20260425_231536.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_yolo11s_6cls_personmerge_e100_20260425_231536.meta`
- W&B run：`l88vy639`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/l88vy639`
- 运行配置：
  - `imgsz=800`
  - `optimizer=Adam`
  - `batch=20`
  - `device=0`
  - `epochs=100`
  - `close_mosaic=20`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 模型规模：
  - build summary：`16,475,766` parameters / `54.82 GFLOPs`
  - fused summary：`16,455,798` parameters / `84.98 GFLOPs`
- 训练完成状态：
  - `results.csv` 共 `100` 个 epoch
  - `best.pt` 与 `last.pt` 均已导出并完成 optimizer strip
  - 单卡 `batch=20` 稳定完成，训练峰值显存约 `16GB`

- `results.csv` 最优 epoch：
  - epoch `99`
  - `Precision = 0.76313`
  - `Recall = 0.58431`
  - `mAP50 = 0.65825`
  - `mAP50-95 = 0.47101`
- epoch `100` 指标：
  - `Precision = 0.75689`
  - `Recall = 0.59181`
  - `mAP50 = 0.65544`
  - `mAP50-95 = 0.46930`
- W&B summary：
  - `mAP50 = 0.65846`
  - `mAP50-95 = 0.47092`
  - `model/GFLOPs = 0`，该 run 启动时仍使用旧版 GFLOPs logger；代码已在后续提交中修复，后续 run 应以日志 summary 和修复后的 W&B 记录为准。
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.512`, `mAP50-95 = 0.267`
  - `motorcycle`: `mAP50 = 0.525`, `mAP50-95 = 0.246`
  - `car`: `mAP50 = 0.877`, `mAP50-95 = 0.684`
  - `truck`: `mAP50 = 0.609`, `mAP50-95 = 0.456`
  - `bus`: `mAP50 = 0.726`, `mAP50-95 = 0.616`
  - `autorickshaw`: `mAP50 = 0.701`, `mAP50-95 = 0.557`

### 10.2 与 Light NIR / P2 对比

- 与 `BiFPN-only + Light NIR branch` 高配方结果（`0.66055 / 0.47087`）对比：
  - 总体 `mAP50 -0.00230`
  - 总体 `mAP50-95 +0.00014`
  - 参数量更高：`16.48M` vs `14.17M`
  - 小目标略高于 Light NIR：`person 0.512 / 0.267` vs `0.502 / 0.255`，`motorcycle 0.525 / 0.246` vs `0.522 / 0.243`
- 与 `BiFPN-only + Light NIR branch + P2 head` 结果（`0.67259 / 0.47049`）对比：
  - 总体 `mAP50 -0.01434`
  - 总体 `mAP50-95 +0.00052`
  - 小目标明显低于 P2：`person 0.512 / 0.267` vs `0.554 / 0.290`，`motorcycle 0.525 / 0.246` vs `0.581 / 0.280`
- 与同配方 6 类 `YOLO11s RGB-only` 高配方基线（`0.61056 / 0.42368`）对比：
  - 总体 `mAP50 +0.04769`
  - 总体 `mAP50-95 +0.04733`

### 10.3 结果解读与下一步

- `close_mosaic=20` 对原始 BiFPN-only 的收益明显，使其总体 `mAP50-95` 追平当前 Light NIR 主线。
- 但原始 BiFPN-only 仍是完整对称 NIR 分支，参数量高于 Light NIR；在效率和论文结构简洁性上，Light NIR 仍更适合作为主线。
- P2 的价值仍集中在 `person / motorcycle` 小目标增强；如果论文主表以总体 `mAP50-95` 排序，P2 不应替代主线；如果强调小目标分析，P2 可以作为专项消融。
- 当前推荐主表结论更新为：`RGB-NIR + BiFPN` 是主要增益来源，`Light NIR branch` 在几乎不损失总体精度的前提下降低冗余，`P2 head` 进一步提升小目标但不提升总体 `mAP50-95`。
- 下一步如果继续训练，应优先补同配方 `RGB-only close_mosaic=20`，否则当前 `RGB-only` 基线仍是 `close_mosaic=10`，和新一批 `close_mosaic=20` 结果存在一个训练策略变量差异。

## 11. 2026-04-26 AdamW/lr0=0.001 与 close_mosaic 消融结果

### 11.1 已完成 run 汇总

三条训练均在 `ssh lyh`、`/data1/lvyanhu/code/yolov11-rgbnir-formal` 下完成，数据根为 `/data1/lvyanhu/code/datasets/iddaw_all_weather_full_yolov11_rgbnir_6cls_personmerge`。

| 模式 | run | 配方 | best epoch | P | R | mAP50 | mAP50-95 | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge6` | `800, AdamW, lr0=0.001, batch=20, close_mosaic=20` | `99` | `0.75772` | `0.57149` | `0.64335` | `0.46029` | `nb3mdkmj` |
| `bifpn_only_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-bifpn-only-light-nir-6cls-personmerge7` | `800, AdamW, lr0=0.001, batch=20, close_mosaic=10` | `100` | `0.72128` | `0.58693` | `0.63484` | `0.44939` | `3ka6dtkb` |
| `rgbnir_light_nir_yolo11s_6cls_personmerge` | `iddaw-yolo11s-rgbnir-light-nir-6cls-personmerge` | `800, AdamW, lr0=0.001, batch=20, close_mosaic=10` | `99` | `0.73076` | `0.55552` | `0.61771` | `0.42634` | `tm1bfkzu` |

对应日志：

- `bifpn_only_light_nir_yolo11s_6cls_personmerge_e100_20260426_031128.stdout.log`
- `bifpn_only_light_nir_yolo11s_6cls_personmerge_e100_20260426_043645.stdout.log`
- `rgbnir_light_nir_yolo11s_6cls_personmerge_e100_20260426_062145.stdout.log`

### 11.2 best.pt 类别表现

`BiFPN + Light NIR, AdamW, close_mosaic=20`：

- `person`: `mAP50 = 0.478`, `mAP50-95 = 0.247`
- `motorcycle`: `mAP50 = 0.501`, `mAP50-95 = 0.233`
- `car`: `mAP50 = 0.866`, `mAP50-95 = 0.668`
- `truck`: `mAP50 = 0.586`, `mAP50-95 = 0.458`
- `bus`: `mAP50 = 0.709`, `mAP50-95 = 0.594`
- `autorickshaw`: `mAP50 = 0.720`, `mAP50-95 = 0.560`

`BiFPN + Light NIR, AdamW, close_mosaic=10`：

- `person`: `mAP50 = 0.483`, `mAP50-95 = 0.243`
- `motorcycle`: `mAP50 = 0.487`, `mAP50-95 = 0.225`
- `car`: `mAP50 = 0.870`, `mAP50-95 = 0.659`
- `truck`: `mAP50 = 0.563`, `mAP50-95 = 0.436`
- `bus`: `mAP50 = 0.690`, `mAP50-95 = 0.567`
- `autorickshaw`: `mAP50 = 0.717`, `mAP50-95 = 0.564`

`RGB-NIR + Light NIR plain, AdamW, close_mosaic=10`：

- `person`: `mAP50 = 0.467`, `mAP50-95 = 0.228`
- `motorcycle`: `mAP50 = 0.452`, `mAP50-95 = 0.197`
- `car`: `mAP50 = 0.859`, `mAP50-95 = 0.635`
- `truck`: `mAP50 = 0.588`, `mAP50-95 = 0.435`
- `bus`: `mAP50 = 0.643`, `mAP50-95 = 0.532`
- `autorickshaw`: `mAP50 = 0.700`, `mAP50-95 = 0.532`

### 11.3 结果分析

- `AdamW + lr0=0.001` 没有超过此前 `Adam + close_mosaic=20` 的 Light NIR 主线：`0.64335 / 0.46029` 低于 `0.66055 / 0.47087`。
- 在同为 `AdamW + lr0=0.001 + batch=20` 下，`close_mosaic=20` 明显优于 `close_mosaic=10`：`mAP50 +0.00851`，`mAP50-95 +0.01090`。
- 在同为 `AdamW + lr0=0.001 + close_mosaic=10` 下，`BiFPN` 相对 Light NIR plain 仍有明确收益：`mAP50 +0.01713`，`mAP50-95 +0.02305`。
- Light NIR plain 的模型更轻：`11.91M parameters / 47.07 GFLOPs`；BiFPN + Light NIR 为 `14.15M parameters / 80.15 GFLOPs`。但精度差距说明 BiFPN 仍是当前结构主增益来源。
- 小目标上，`BiFPN + Light NIR + AdamW close_mosaic=10` 的 `person/motorcycle` 仍明显低于 P2 head：`person 0.483/0.243`、`motorcycle 0.487/0.225`，而 P2 为 `0.554/0.290`、`0.581/0.280`。

### 11.4 下一步计划

- 当前不建议把主优化器切到 `AdamW + lr0=0.001`；论文主线结果继续以 `Adam + close_mosaic=20` 为主。
- 当前不建议回退到 `close_mosaic=10`；后续高配方统一保持 `close_mosaic=20`，除非专门做训练策略消融。
- 当前主线仍定为 `YOLO11s + BiFPN + Light NIR branch`：它在总体精度、参数效率和结构解释性之间最稳。
- 若继续追小目标，下一步应围绕 `P2 head` 做轻量化消融，而不是更换优化器：优先试 `P2-lite`，例如降低 P2 通道、减少 P2 分支 C3k2 深度，目标是保留 `person/motorcycle` 增益，同时减少对中大目标的副作用。
- 最终主表建议使用 `Adam + close_mosaic=20` 口径；`AdamW/lr0=0.001` 作为“优化器未带来增益”的负向消融记录即可。

## 12. 2026-04-26 P2-P5 BiFPN 结构改造

### 12.1 新增模式

- 新增 mode：`bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge`
- 新增配置：`configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_6cls_personmerge.yaml`
- 结构基线：基于 `bifpn_only_light_nir_p2_yolo11s_6cls_personmerge` 修改，而不是基于残差质量感知路线。

### 12.2 结构变化

- 旧 `P2 head` 版本：`P3/P4/P5` 先进三尺度 `BiFPN`，`P2` 只在检测头通过上采样后额外拼接进入 Detect。
- 新 `P2-P5 BiFPN` 版本：`P2/P3/P4/P5` 同时进入 `BiFPNP2P5`，执行 top-down 与 bottom-up 双向融合，再对四个输出尺度分别做轻量 `C3k2` refine 后送入 `Detect`。
- `BiFPN` 内部卷积同步改为 depthwise separable conv，并修复旧三尺度 `BiFPN` 中 `P5` bottom-up 节点重复融合原始 `P5` 的问题。
- 本轮仍保持 `RGB full branch + Light NIR branch`，不引入 residual quality fusion、reflectance branch 或额外 attention。

### 12.3 待验证命令

冒烟：

```bash
WANDB_ENABLED=0 BATCH=20 OPTIMIZER=Adam IMGSZ=800 CLOSE_MOSAIC=20 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge 1 0
```

长训：

```bash
WANDB_ENABLED=1 BATCH=20 OPTIMIZER=Adam IMGSZ=800 CLOSE_MOSAIC=20 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge 100 0
```

### 12.4 评价重点

- 第一优先：总体 `mAP50-95` 是否不低于当前 `P2 head` 与 `Light NIR` 主线。
- 第二优先：`person` 与 `motorcycle` 的 `mAP50-95` 是否保留 P2 小目标增益。
- 第三优先：参数量与 GFLOPs 是否可接受；若显著增大但小目标收益不明显，则不进入主线。

### 12.5 当前构建验证

- 本地 `py_compile` 已通过：`ultralytics/nn/modules/conv.py`、`ultralytics/nn/modules/__init__.py`、`ultralytics/nn/tasks.py`、`formal_rgbnir/iddaw.py`、`scripts/iddaw/run_experiment.py`。
- 本地 YOLO 构建未执行完成，原因是当前本地 Python 缺少 `cv2`；远端 `visnir-exp` 环境已完成实际构建验证。
- 远端构建结果：
  - Detect stride：`[4.0, 8.0, 16.0, 32.0]`
  - Detect 输入尺度数：`4`
  - 模型规模：`536 layers / 9,268,004 parameters / 40.06 GFLOPs @ 640`
- 该规模低于此前 `P2 head` 版本，主要原因是新配置不再使用 YOLO head 的多次上采样/下采样重融合，而是在 `BiFPNP2P5` 后直接对四个尺度做轻量 refine。

### 12.6 Object-aware NIR gate 备选结构

- 新增 mode：`bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge`
- 新增配置：`configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oagate_6cls_personmerge.yaml`
- 基线来源：基于 `bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge` 修改。
- 修改动机：借鉴 `Object-Aware NIR-to-Visible Translation` 中“object-specific reflection + segmentation prior”思想，但不引入完整 NIR-to-VIS 翻译网络；在检测任务中只保留轻量对象感知 NIR 门控。
- 结构变化：
  - `P2/P3`：`Concat(RGB, NIR)` 替换为 `ObjectAwareNIRGateConcat(RGB, NIR)`，先用 NIR reflection cue 与 RGB context 预测 gate，再调制 NIR 后 concat。
  - `P4/P5`：继续使用 plain concat，避免高语义层过度融合。
  - `BiFPNP2P5`、四尺度 refine block 与 `Detect` 保持不变。
- 验证重点：
  - `person` 与 `motorcycle` 的 `mAP50-95` 是否超过当前 P2 head 结果。
  - 总体 `mAP50-95` 是否不明显低于 `0.470`。
  - 若 OA gate 不能改善小目标，则说明瓶颈更可能在样本/标注/assignment，而不是 P2/P3 的 NIR 融合选择。

### 12.7 `P2-P5 BiFPN + Adam/lr0=0.001/close_mosaic=15` 复核结果

- mode：`bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge6`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-6cls-personmerge6`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge_e150_20260426_182521.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge_e150_20260426_182521.meta`
- W&B run：`f4fyeqbr`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/f4fyeqbr`
- 运行配置：
  - `imgsz=800`
  - `optimizer=Adam`
  - `lr0=0.001`
  - `batch=20`
  - `device=0,1`
  - `epochs=150`
  - `close_mosaic=15`
  - `cos_lr=False`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 模型规模：
  - `9,252,260` parameters
  - W&B 记录 `model/GFLOPs = 62.6`
  - best.pt 复验日志 summary：`414 layers / 9,252,260 parameters / 61.81 GFLOPs`
- 训练完成状态：
  - `150 epochs completed in 1.924 hours`
  - `best.pt` 与 `last.pt` 均已导出并完成 optimizer strip
  - `results.csv` 共 `150` 个 epoch

- `results.csv` 最优 epoch：
  - epoch `150`
  - `Precision = 0.73999`
  - `Recall = 0.55773`
  - `mAP50 = 0.63204`
  - `mAP50-95 = 0.44113`
- W&B summary：
  - `Precision = 0.74018`
  - `Recall = 0.55678`
  - `mAP50 = 0.63206`
  - `mAP50-95 = 0.44135`
  - `model/GFLOPs = 62.6`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.514`, `mAP50-95 = 0.259`
  - `motorcycle`: `mAP50 = 0.509`, `mAP50-95 = 0.233`
  - `car`: `mAP50 = 0.873`, `mAP50-95 = 0.657`
  - `truck`: `mAP50 = 0.563`, `mAP50-95 = 0.422`
  - `bus`: `mAP50 = 0.637`, `mAP50-95 = 0.545`
  - `autorickshaw`: `mAP50 = 0.697`, `mAP50-95 = 0.533`

#### 12.7.1 对比分析

- 相比上一轮 `P2-P5 + AdamW + lr0=0.001 + cos_lr=True`（`mAP50 = 0.61474`，`mAP50-95 = 0.42033`）：
  - `mAP50 +0.01730`
  - `mAP50-95 +0.02080`
  - 说明退化不完全来自 `P2-P5` 结构，`AdamW + cos_lr` 在该结构上也不稳。
- 相比单卡 `P2-P5 + 640 + Adam + close_mosaic=20`（`mAP50 = 0.66597`，`mAP50-95 = 0.45832`）：
  - `mAP50 -0.03393`
  - `mAP50-95 -0.01719`
  - 说明升到 `800/150` 并没有自然修复 true P2-P5 的问题。
- 相比 `P2 head` 版本（`mAP50 = 0.67259`，`mAP50-95 = 0.47049`）：
  - 总体 `mAP50 -0.04055`
  - 总体 `mAP50-95 -0.02936`
  - `person`: `0.554 / 0.290 -> 0.514 / 0.259`
  - `motorcycle`: `0.581 / 0.280 -> 0.509 / 0.233`
- 相比无 P2 的 `BiFPN + Light NIR branch` 主线（`mAP50 = 0.66055`，`mAP50-95 = 0.47087`）：
  - 总体 `mAP50 -0.02851`
  - 总体 `mAP50-95 -0.02974`

#### 12.7.2 结论与下一步

- true `P2-P5 BiFPN` 已经过三种配方验证，当前没有超过 `P2 head` 或无 P2 的 Light NIR 主线，不建议作为论文主结构。
- `Adam + lr0=0.001 + close_mosaic=15` 比 `AdamW + cos_lr` 明显更好，但仍不足以让 true P2-P5 晋级。
- 当前最稳主线仍是：
  - 总体主线：`bifpn_only_light_nir_yolo11s_6cls_personmerge`
  - 小目标增强分支：`bifpn_only_light_nir_p2_yolo11s_6cls_personmerge`
- 已实现的 `P2-P5 + OA gate` 只作为诊断实验保留，用来回答“P2/P3 的 NIR 选择性融合是否能修复 true P2-P5 的小目标退化”。
- 下一步执行建议：
  - 先在远端 `git pull` 到包含 OA gate 的 `b731166` 或更新提交。
  - 冒烟验证 `bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge`，确认 `parse_model`、四尺度 Detect、DDP 与 W&B/GFLOPs 记录正常。
  - 若冒烟通过，仅跑一轮同配方诊断训练：`imgsz=800`、`Adam`、`lr0=0.001`、`batch=20`、`close_mosaic=15`、`device=0,1`，与本 run 严格隔离 OA gate 变量；实际执行已按后续策略缩短为 `epochs=100`，见 12.8。
  - OA gate 晋级标准：至少超过当前 true P2-P5 的 `mAP50-95 = 0.44113`，并且 `person/motorcycle` 的 `mAP50-95` 明确高于 `0.259 / 0.233`；若不能接近 `P2 head` 的 `0.290 / 0.280`，则不继续在 P2-P5 上加模块。
  - 如果 OA gate 在 P2-P5 上不达标，下一阶段应把 OA gate 迁移到 `P2 head` 版本，而不是继续强化 true P2-P5 BiFPN。

### 12.8 `P2-P5 BiFPN + P2/P3 OA gate` 诊断训练结果

- mode：`bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-6cls-personmerge2`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-6cls-personmerge2`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge_e100_20260426_210506.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge_e100_20260426_210506.meta`
- W&B run：`369d7rzk`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/369d7rzk`
- 运行配置：
  - `imgsz=800`
  - `optimizer=Adam`
  - `lr0=0.001`
  - `batch=20`
  - `device=0,1`
  - `epochs=100`
  - `close_mosaic=15`
  - `cos_lr=False`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 模型规模：
  - `9,990,568` parameters
  - W&B 记录 `model/GFLOPs = 83.346`
  - best.pt 复验日志 summary：`450 layers / 9,974,056 parameters / 82.43 GFLOPs`
- 训练完成状态：
  - `100 epochs completed in 1.526 hours`
  - `best.pt` 与 `last.pt` 均已导出并完成 optimizer strip
  - `results.csv` 共 `100` 个 epoch

- `results.csv` 最优 epoch：
  - epoch `100`
  - `Precision = 0.70032`
  - `Recall = 0.57634`
  - `mAP50 = 0.63013`
  - `mAP50-95 = 0.43517`
- W&B summary：
  - `Precision = 0.70161`
  - `Recall = 0.57437`
  - `mAP50 = 0.62996`
  - `mAP50-95 = 0.43456`
  - `model/GFLOPs = 83.346`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.512`, `mAP50-95 = 0.251`
  - `motorcycle`: `mAP50 = 0.510`, `mAP50-95 = 0.231`
  - `car`: `mAP50 = 0.877`, `mAP50-95 = 0.655`
  - `truck`: `mAP50 = 0.533`, `mAP50-95 = 0.401`
  - `bus`: `mAP50 = 0.611`, `mAP50-95 = 0.515`
  - `autorickshaw`: `mAP50 = 0.736`, `mAP50-95 = 0.553`

#### 12.8.1 对比分析

- 相比同配方 plain `P2-P5 BiFPN`（`mAP50 = 0.63204`，`mAP50-95 = 0.44113`）：
  - `mAP50 -0.00191`
  - `mAP50-95 -0.00596`
  - `person`: `0.514 / 0.259 -> 0.512 / 0.251`
  - `motorcycle`: `0.509 / 0.233 -> 0.510 / 0.231`
  - OA gate 对小目标没有形成有效增益，反而略降整体定位质量。
- 相比 `P2 head` 版本（`mAP50 = 0.67259`，`mAP50-95 = 0.47049`）：
  - 总体 `mAP50 -0.04246`
  - 总体 `mAP50-95 -0.03532`
  - `person`: `0.554 / 0.290 -> 0.512 / 0.251`
  - `motorcycle`: `0.581 / 0.280 -> 0.510 / 0.231`
  - 说明当前最有效的小目标增强仍是“保留 P2 检测头”，而不是把 P2 纳入 full BiFPN 后再加门控。
- 相比无 P2 的 `BiFPN + Light NIR branch` 主线（`mAP50 = 0.66055`，`mAP50-95 = 0.47087`）：
  - 总体 `mAP50 -0.03042`
  - 总体 `mAP50-95 -0.03570`
  - OA gate 增加了约 `0.72M` 参数和明显 GFLOPs，但没有超过更简单的 Light NIR 主线。

#### 12.8.2 结论与下一步

- `P2-P5 + P2/P3 OA gate` 不晋级为论文主结构。
- 当前结果说明 true `P2-P5 BiFPN` 的问题不主要是 `P2/P3` 融合前缺少 NIR 质量筛选；更可能是 P2 进入 BiFPN 后改变了多尺度特征分配，使小目标定位收益被融合链路稀释。
- 不建议继续在 true `P2-P5 BiFPN` 上增加通道数、增加 gate 或叠加更重的 attention，因为该路径已经连续低于 `P2 head` 与无 P2 的 Light NIR 主线。
- 下一阶段建议回到表现更稳定的两条线：
  - 论文主结构候选：`bifpn_only_light_nir_yolo11s_6cls_personmerge`
  - 小目标增强候选：`bifpn_only_light_nir_p2_yolo11s_6cls_personmerge`
- 下一步优先实验：
  - 做 `P2 head + close_mosaic=15 + Adam + lr0=0.001 + imgsz=800 + batch=20 + device=0,1 + epochs=100` 复核，与当前 OA gate 同配方对齐。
  - 若该结果仍接近或超过 `mAP50-95 = 0.470`，则正式停止 true P2-P5 路线，把小目标章节聚焦到“P2 检测头增强”。
  - 若还要引入 OA 思想，优先迁移到 `P2 head` 的 `P2/P3` 融合处，而不是继续放在 true P2-P5 BiFPN 中。

### 12.9 `true P2-P5 BiFPN c256` 待执行实验

- 新增 mode：
  - `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`
  - `bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge`
- 新增配置：
  - `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_c256_6cls_personmerge.yaml`
  - `configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oagate_c256_6cls_personmerge.yaml`
- 基线来源：
  - plain c256 基于 `bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge`
  - OA gate c256 基于 `bifpn_only_light_nir_p2p5_oagate_yolo11s_6cls_personmerge`
- 结构变化：
  - `BiFPNP2P5` 参数从 `[192, 2]` 改为 `[256, 2]`
  - 四个 Detect 前 refine head 的通道字面量统一改为 `256`
  - backbone、Light NIR branch、P2/P3/P4/P5 输入层、Detect 尺度数保持不变
  - OA gate 版本仍只在 `P2/P3` 使用 `ObjectAwareNIRGateConcat`，`P4/P5` 保持 plain concat
- 执行计划：
  - 当前 `bifpn_only_light_nir_p2_yolo11s_6cls_personmerge` 训练完成后，按顺序启动：
  - 第一轮：`bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`
  - 第二轮：`bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge`
  - 统一配方：`imgsz=800`、`optimizer=Adam`、`lr0=0.001`、`batch=20`、`epochs=100`、`close_mosaic=15`、`device=0,1`
- 验证目的：
  - 判断 true P2-P5 路线之前的退化是否来自 `192` 宽度不足。
  - 如果 c256 仍无法超过 `P2 head` 或无 P2 的 Light NIR 主线，则停止 true P2-P5 方向。
  - 如果 plain c256 有效而 OA gate c256 无效，说明瓶颈在 BiFPN 宽度而不是 NIR gate。

### 12.10 `P2 head + close_mosaic=15` 复核结果

- mode：`bifpn_only_light_nir_p2_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge5`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2-6cls-personmerge5`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2_yolo11s_6cls_personmerge_e100_20260426_225144.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2_yolo11s_6cls_personmerge_e100_20260426_225144.meta`
- W&B run：`8lavg7mj`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/8lavg7mj`
- 运行配置：
  - `imgsz=800`
  - `optimizer=Adam`
  - `lr0=0.001`
  - `batch=20`
  - `device=0,1`
  - `epochs=100`
  - `close_mosaic=15`
  - `cos_lr=False`
  - `WANDB_ENABLED=1`
  - `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- 模型规模：
  - `10,223,642` parameters
  - W&B 记录 `model/GFLOPs = 61.399`
  - best.pt 复验日志 summary：`427 layers / 10,206,410 parameters / 60.60 GFLOPs`
- 训练完成状态：
  - `100` epoch 完成
  - `best.pt` 与 `last.pt` 均已导出
  - `results.csv` 共 `100` 个 epoch

- `results.csv` 最优 epoch：
  - epoch `99`
  - `Precision = 0.72686`
  - `Recall = 0.58771`
  - `mAP50 = 0.64462`
  - `mAP50-95 = 0.44502`
- `results.csv` 最后 epoch：
  - epoch `100`
  - `Precision = 0.70301`
  - `Recall = 0.59609`
  - `mAP50 = 0.64083`
  - `mAP50-95 = 0.44485`
- W&B summary：
  - `Precision = 0.72470`
  - `Recall = 0.58781`
  - `mAP50 = 0.64426`
  - `model/GFLOPs = 61.399`
  - `model/parameters = 10223642`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.504`, `mAP50-95 = 0.250`
  - `motorcycle`: `mAP50 = 0.519`, `mAP50-95 = 0.232`
  - `car`: `mAP50 = 0.879`, `mAP50-95 = 0.663`
  - `truck`: `mAP50 = 0.582`, `mAP50-95 = 0.433`
  - `bus`: `mAP50 = 0.648`, `mAP50-95 = 0.534`
  - `autorickshaw`: `mAP50 = 0.734`, `mAP50-95 = 0.555`

#### 12.10.1 对比分析

- 相比此前 `P2 head + close_mosaic=20`（`mAP50 = 0.67259`，`mAP50-95 = 0.47049`）：
  - `mAP50 -0.02797`
  - `mAP50-95 -0.02547`
  - `person`: `0.554 / 0.290 -> 0.504 / 0.250`
  - `motorcycle`: `0.581 / 0.280 -> 0.519 / 0.232`
  - 说明 `close_mosaic=15` 对当前 P2 head 不如 `close_mosaic=20` 稳定。
- 相比 `P2-P5 + OA gate`（`mAP50 = 0.63013`，`mAP50-95 = 0.43517`）：
  - `mAP50 +0.01449`
  - `mAP50-95 +0.00985`
  - 但 `person/motorcycle` 的 `mAP50-95 = 0.250 / 0.232` 仍没有接近此前 P2 head 的 `0.290 / 0.280`。
- 相比无 P2 的 `BiFPN + Light NIR branch` 主线（`mAP50 = 0.66055`，`mAP50-95 = 0.47087`）：
  - `mAP50 -0.01593`
  - `mAP50-95 -0.02585`

#### 12.10.2 结论

- `P2 head` 的有效版本仍是此前 `close_mosaic=20` 结果，不是本轮 `close_mosaic=15` 复核。
- 当前结果强化了一个结论：小目标收益不仅来自 P2 检测头结构，也明显依赖更长的 no-mosaic 阶段。
- 后续若继续训练 P2 系列，应回到 `close_mosaic=20`；如果为了和 `P2-P5/OA gate` 的 `close_mosaic=15` 对齐，只能作为诊断结果，不适合作为最终主表。
- c256 队列异常原因：
  - `formal_rgbnir/iddaw.py` 已注册新增 c256 mode，但 `scripts/iddaw/run_experiment.py` 的 argparse choices 仍是旧的静态列表。
  - 第一次队列启动 `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge` 后被 argparse 拒绝，日志中报 `invalid choice`。
  - 修复方案：`run_experiment.py` 改为直接使用 `TRAINABLE_MODES | {"decision_fusion"}` 生成 choices，避免以后新增 mode 时再次漏改入口。

### 12.11 `true P2-P5 BiFPN c256` 与 `OA gate c256` 续训前结果

本节只记录当前续训前的两个原始 run。后续 `oagate-c256` 的 `personmerge2` 续训结果单独记录，不提前并入主结论。

#### 12.11.1 plain c256 完整训练结果

- mode：`bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge_e100_20260427_001438.stdout.log`
- W&B run：`61ke0ekm`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/61ke0ekm`
- 运行配置：`100 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1`
- 模型规模：`536 layers / 10,121,924 parameters / 54.57 GFLOPs`
- 训练状态：`100` epoch 正常完成，`best.pt` 与 `last.pt` 已导出，`results.csv` 共 `100` 个 epoch

- `results.csv` 最优 epoch：
  - epoch `94`
  - `Precision = 0.68729`
  - `Recall = 0.59268`
  - `mAP50 = 0.63579`
  - `mAP50-95 = 0.43961`
- `results.csv` 最后 epoch：
  - epoch `100`
  - `Precision = 0.67522`
  - `Recall = 0.58808`
  - `mAP50 = 0.62960`
  - `mAP50-95 = 0.43765`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.514`, `mAP50-95 = 0.266`
  - `motorcycle`: `mAP50 = 0.526`, `mAP50-95 = 0.242`
  - `car`: `mAP50 = 0.872`, `mAP50-95 = 0.661`
  - `truck`: `mAP50 = 0.561`, `mAP50-95 = 0.419`
  - `bus`: `mAP50 = 0.611`, `mAP50-95 = 0.499`
  - `autorickshaw`: `mAP50 = 0.729`, `mAP50-95 = 0.549`

#### 12.11.2 OA gate c256 续训前中断结果

- mode：`bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge_e100_20260427_015015.stdout.log`
- W&B run：`d79prr6p`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/d79prr6p`
- 运行配置：`100 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1`
- 模型规模：`576 layers / 10,844,488 parameters / 67.85 GFLOPs`
- 训练状态：中断于 epoch `91/100` 中途；`results.csv` 最后一条完整记录为 epoch `90`

- 续训前 `results.csv` 最优/最后 epoch：
  - epoch `90`
  - `Precision = 0.75325`
  - `Recall = 0.54912`
  - `mAP50 = 0.62743`
  - `mAP50-95 = 0.43061`
- 备注：
  - 该 run 未完成最后验证与 `best.pt` 复验，因此暂无可靠的最终 per-class AP 表。
  - 当前已从 `last.pt` 启动 15 epoch 续训用于补充观察；由于它是在旧版“权重续训”逻辑下启动，不作为严格公平主表结果。
  - 之后异常中断恢复入口已改为 Ultralytics 原生 `resume=True`，后续新恢复任务会保留 optimizer、EMA、start epoch 与 mosaic 阶段。

#### 12.11.3 对比分析

- plain c256 相比 c192 的 `true P2-P5 + Adam + close_mosaic=15`（`mAP50 = 0.63204`，`mAP50-95 = 0.44113`）：
  - `mAP50 +0.00375`
  - `mAP50-95 -0.00152`
  - 说明把 `BiFPNP2P5` 与 refine head 字面通道从 `192` 加到 `256` 没有带来高 IoU 指标增益。
- plain c256 相比 `P2 head + close_mosaic=20`（`mAP50 = 0.67259`，`mAP50-95 = 0.47049`）：
  - `mAP50 -0.03680`
  - `mAP50-95 -0.03088`
  - `person/motorcycle` 的 `mAP50-95 = 0.266 / 0.242` 也低于 P2 head close_mosaic=20 的 `0.290 / 0.280`。
- plain c256 相比无 P2 的 `BiFPN + Light NIR branch` 主线（`mAP50 = 0.66055`，`mAP50-95 = 0.47087`）：
  - `mAP50 -0.02476`
  - `mAP50-95 -0.03126`
  - 说明 true P2-P5 加宽后仍没有超过更简单、效率更好的 Light NIR 主线。
- OA gate c256 续训前结果低于 plain c256：
  - `mAP50 -0.00836`
  - `mAP50-95 -0.00900`
  - OA gate c256 增加约 `0.72M` 参数与 `13.28` GFLOPs，但中断前没有显示收益。

#### 12.11.4 下一步训练计划

- 本小节是基于 `lr0=0.001` 及续训前结果形成的历史判断。
- 后续 `12.12` 的 `OA gate c256 + Adam lr0=0.01` 已达到 `mAP50-95 = 0.47814`，因此 true P2-P5 + OA gate 路线重新进入候选。
- 当前有效下一步计划以 `12.13` 为准：先补同配方 `plain c256 + lr0=0.01` 与 `c192 + lr0=0.01`，再决定是否实现下一版 Object-Aware Reflectance Gate。

### 12.12 `OA gate c256` 原生 resume 完成结果

- mode：`bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge`
- 原始中断日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge_e100_20260427_015015.stdout.log`
- 原生 resume 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge_e100_20260427_123753.stdout.log`
- W&B run：`ya7peh45`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/ya7peh45`
- 运行配置：`100 epoch, imgsz=800, Adam, lr0=0.001, batch=20, close_mosaic=15, device=0,1`
- 恢复方式：Ultralytics 原生 `resume=True`
- 恢复状态：从 epoch `91` 恢复到 `100` total epochs，并触发 `Closing dataloader mosaic`
- 模型规模：
  - W&B 记录：`10,844,488` parameters / `106.018` GFLOPs
  - `best.pt` 复验日志：`450 layers / 10,826,312 parameters / 104.91 GFLOPs`
- 训练完成状态：
  - `100` epoch 完成
  - `best.pt` 与 `last.pt` 均已导出
  - `results.csv` 共 `100` 个 epoch

- `results.csv` 最优 epoch：
  - epoch `92`
  - `Precision = 0.72876`
  - `Recall = 0.56793`
  - `mAP50 = 0.63061`
  - `mAP50-95 = 0.43168`
- `results.csv` 最后 epoch：
  - epoch `100`
  - `Precision = 0.74222`
  - `Recall = 0.56409`
  - `mAP50 = 0.62974`
  - `mAP50-95 = 0.42835`
- W&B summary：
  - `Precision = 0.72781`
  - `Recall = 0.56808`
  - `mAP50 = 0.63050`
  - `mAP50-95 = 0.43163`
  - `model/GFLOPs = 106.018`
  - `model/parameters = 10844488`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.503`, `mAP50-95 = 0.252`
  - `motorcycle`: `mAP50 = 0.501`, `mAP50-95 = 0.225`
  - `car`: `mAP50 = 0.865`, `mAP50-95 = 0.657`
  - `truck`: `mAP50 = 0.542`, `mAP50-95 = 0.403`
  - `bus`: `mAP50 = 0.640`, `mAP50-95 = 0.506`
  - `autorickshaw`: `mAP50 = 0.732`, `mAP50-95 = 0.547`

#### 12.12.1 结论

- 原生 resume 完成后，`OA gate c256` 最优 `mAP50-95 = 0.43168`，仍低于 plain c256 的 `0.43961`。
- `person/motorcycle` 的 `mAP50-95 = 0.252 / 0.225`，也低于 plain c256 的 `0.266 / 0.242`，说明 `P2/P3` OA gate 没有改善当前最关心的小目标。
- 相比无 P2 的 `BiFPN + Light NIR branch` 主线（`0.66055 / 0.47087`）和 `P2 head + close_mosaic=20`（`0.67259 / 0.47049`），true P2-P5 + OA gate c256 仍明显落后。
- 当前证据不支持继续沿 true P2-P5 加宽或 OA gate 加重方向推进；后续新增训练只作为学习率敏感性诊断。

#### 12.12.2 `lr0=0.01` 复训完成结果

- mode：`bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge3`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oagate-c256-6cls-personmerge3`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge_e100_20260427_125557.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge_e100_20260427_125557.meta`
- W&B run：`02h09m4g`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/02h09m4g`
- 运行配置：`100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1`
- 训练状态：`100` epoch 正常完成，`best.pt` 与 `last.pt` 已导出，`results.csv` 共 `100` 个 epoch
- 模型规模：
  - 训练构建 summary：`576 layers / 10,844,488 parameters / 67.85 GFLOPs`
  - `best.pt` 复验日志：`450 layers / 10,826,312 parameters / 104.91 GFLOPs`
  - W&B 记录：`10,844,488` parameters / `106.018` GFLOPs

- 启动命令：

```bash
WANDB_ENABLED=1 BATCH=20 OPTIMIZER=Adam LR0=0.01 IMGSZ=800 CLOSE_MOSAIC=15 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_oagate_c256_yolo11s_6cls_personmerge 100 0,1
```

- `results.csv` 最优 epoch：
  - epoch `100`
  - `Precision = 0.74477`
  - `Recall = 0.62225`
  - `mAP50 = 0.68654`
  - `mAP50-95 = 0.47814`
- `results.csv` 最后 epoch：
  - epoch `100`
  - `Precision = 0.74477`
  - `Recall = 0.62225`
  - `mAP50 = 0.68654`
  - `mAP50-95 = 0.47814`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.569`, `mAP50-95 = 0.298`
  - `motorcycle`: `mAP50 = 0.586`, `mAP50-95 = 0.274`
  - `car`: `mAP50 = 0.893`, `mAP50-95 = 0.695`
  - `truck`: `mAP50 = 0.618`, `mAP50-95 = 0.453`
  - `bus`: `mAP50 = 0.684`, `mAP50-95 = 0.545`
  - `autorickshaw`: `mAP50 = 0.769`, `mAP50-95 = 0.600`

#### 12.12.3 对比分析

- 相比同结构 `lr0=0.001` 原生 resume 结果（`mAP50 = 0.63061`，`mAP50-95 = 0.43168`）：
  - `mAP50 +0.05593`
  - `mAP50-95 +0.04646`
  - `person`: `0.503 / 0.252 -> 0.569 / 0.298`
  - `motorcycle`: `0.501 / 0.225 -> 0.586 / 0.274`
  - 说明此前 OA gate c256 表现差，很大一部分是优化器学习率不足，而不是结构本身完全无效。
- 相比 plain c256（`mAP50 = 0.63579`，`mAP50-95 = 0.43961`）：
  - `mAP50 +0.05075`
  - `mAP50-95 +0.03853`
  - `person/motorcycle` 的 `mAP50-95` 分别提升 `+0.032 / +0.032`
  - 说明在合适学习率下，P2/P3 OA gate 对小目标和整体定位均有明确增益。
- 相比无 P2 的 `BiFPN + Light NIR branch` 主线（`mAP50 = 0.66055`，`mAP50-95 = 0.47087`）：
  - `mAP50 +0.02599`
  - `mAP50-95 +0.00727`
  - 这是目前首次让 true P2-P5 + OA gate c256 在总体 `mAP50-95` 上超过 Light NIR 主线。
- 相比 `P2 head + close_mosaic=20`（`mAP50 = 0.67259`，`mAP50-95 = 0.47049`）：
  - `mAP50 +0.01395`
  - `mAP50-95 +0.00765`
  - `person/motorcycle` 的 `mAP50-95 = 0.298 / 0.274`，接近或超过此前 P2 head 的 `0.290 / 0.280`。

#### 12.12.4 结论

- `OA gate c256 + Adam lr0=0.01` 暂时成为当前结果最强结构，主指标 `mAP50-95 = 0.47814`。
- 这一结果同时支持两件事：
  - BiFPN 侧：true P2-P5 在足够宽度和合适优化器设置下可以超过外接 P2 head 与无 P2 Light NIR 主线。
  - Object-Aware 侧：P2/P3 的 NIR gate 在小目标上有实际价值，但高度依赖学习率，不能再用 `lr0=0.001` 结果否定 OA gate。
- 但该结论还不能直接写成最终主贡献，因为缺少同配方 `plain c256 + lr0=0.01` 和 `no-OA true P2-P5 + lr0=0.01` 对照。当前提升可能来自 `lr0=0.01`、c256 宽度、OA gate 三者共同作用。

### 12.13 下一轮实验计划：BiFPN 与 Object-Aware 两条线拆分验证

下一轮不再继续随意新增结构，目标是拆清楚当前最强结果中到底是哪一部分在贡献：`BiFPN/宽度/学习率`，还是 `Object-Aware NIR gate`。

#### 12.13.1 第一优先级：BiFPN 侧同配方对照

- 实验 A：`plain true P2-P5 c256 + lr0=0.01`
  - mode：`bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`
  - 配方：`100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1`
  - 目的：排除“只是 lr0=0.01 让 c256 BiFPN 变强”的可能。
  - 判定：如果 plain c256 也接近 `0.478`，则主贡献应写 BiFPN + 优化配方；如果明显低于 OA gate c256，则 OA gate 可作为结构贡献。
- 实验 B：`true P2-P5 c192 + lr0=0.01`
  - mode：`bifpn_only_light_nir_p2p5_yolo11s_6cls_personmerge`
  - 配方同上，只保留原 `[192, 2]` BiFPNP2P5。
  - 目的：判断 `256` 宽度是否必要。
  - 判定：如果 c192 接近 c256，则优先选更轻的 c192；如果 c256 明显更高，再保留 c256。

#### 12.13.2 第二优先级：Object-Aware NIR-to-Visible 思想的轻量改造

- 不建议把 Sherry 的完整 `NIR -> VIS` 翻译网络搬进 YOLO；当前方向应继续采用“检测任务内的对象感知 NIR 选择”。
- 下一版 OA 模块应从单纯 gate 升级为 `Object-Aware Reflectance Gate`，但仍只放在 `P2/P3`：
  - 从 NIR 特征提取局部边缘/反射响应，近似对应论文中的 object-specific reflection。
  - 从 RGB/NIR 融合特征生成目标先验 gate，近似对应论文中的 object prior。
  - 输出仍保持检测友好的 `RGB + gate * NIR_reflectance`，不生成伪 RGB 图像。
- 建议先做一个最小实现版本：
  - `NIR -> depthwise 3x3 -> high-frequency/edge feature`
  - `concat(RGB, NIR, edge) -> 1x1/3x3 conv -> sigmoid gate`
  - `fused = concat(RGB, gate * NIR_projected)` 或 `RGB + gate * NIR_projected`
  - 只替换当前 P2/P3 的 `ObjectAwareNIRGateConcat`
- 不引入分割网络，不新增 NIR-to-RGB 重建 loss；如果需要对象先验监督，优先用检测框生成轻量 foreground mask 作为可选辅助约束。

#### 12.13.3 推荐执行顺序

1. 先跑 `plain true P2-P5 c256 + lr0=0.01`。
2. 如果它低于 `OA gate c256 lr0=0.01` 超过 `0.01 mAP50-95`，再实现 `Object-Aware Reflectance Gate`。
3. 如果它与 OA gate c256 接近，则先跑 `true P2-P5 c192 + lr0=0.01`，判断能否降复杂度。
4. 在上述对照完成前，不建议继续加新 head、加 P6、加完整翻译分支或引入额外损失。

#### 12.13.4 plain true P2-P5 c256 + lr0=0.01 完成结果

- mode：`bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge2`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge2`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge_e100_20260427_151357.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge_e100_20260427_151357.meta`
- W&B run：`j8iwyb9q`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/j8iwyb9q`
- 运行配置：`100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1`
- 训练状态：`100` epoch 正常完成，`best.pt` 与 `last.pt` 已导出，`results.csv` 共 `100` 个 epoch
- 模型规模：
  - `resume_mode=none`
  - `BiFPNP2P5` 参数为 `[[256, 512, 512, 512], 256, 2]`
  - 模型不包含 `ObjectAwareNIRGateConcat`，P2/P3/P4/P5 融合前均为 plain `Concat`
  - 训练构建 summary：`536 layers / 10,121,924 parameters / 54.57 GFLOPs`
  - `best.pt` 复验日志：`414 layers / 10,104,516 parameters / 84.28 GFLOPs`
  - W&B 记录：`10,121,924` parameters / `85.271` GFLOPs

- `results.csv` 最优 epoch：
  - epoch `92`
  - `Precision = 0.72568`
  - `Recall = 0.62905`
  - `mAP50 = 0.69135`
  - `mAP50-95 = 0.47976`
- `results.csv` 最后 epoch：
  - epoch `100`
  - `Precision = 0.73373`
  - `Recall = 0.61665`
  - `mAP50 = 0.68457`
  - `mAP50-95 = 0.47822`
- `best.pt` 复验主要类别表现：
  - `person`: `mAP50 = 0.561`, `mAP50-95 = 0.296`
  - `motorcycle`: `mAP50 = 0.580`, `mAP50-95 = 0.275`
  - `car`: `mAP50 = 0.888`, `mAP50-95 = 0.686`
  - `truck`: `mAP50 = 0.636`, `mAP50-95 = 0.476`
  - `bus`: `mAP50 = 0.669`, `mAP50-95 = 0.538`
  - `autorickshaw`: `mAP50 = 0.815`, `mAP50-95 = 0.604`

##### 12.13.4.1 对比结论

- 相比 `OA gate c256 + lr0=0.01`（`mAP50 = 0.68654`，`mAP50-95 = 0.47814`）：
  - `mAP50 +0.00481`
  - `mAP50-95 +0.00162`
  - `person`: `0.569 / 0.298 -> 0.561 / 0.296`
  - `motorcycle`: `0.586 / 0.274 -> 0.580 / 0.275`
  - 小目标基本持平，总体指标略高，且 plain c256 少约 `0.72M` 参数、约 `20.75` W&B GFLOPs。
- 当前证据说明：`lr0=0.01` 与 true P2-P5 c256 是主要收益来源；现有 `ObjectAwareNIRGateConcat` 并没有在同配方下提供稳定额外增益。
- Object-Aware 方向不应继续沿当前 gate 形式加重；如果继续，应改成更贴近 NIR-to-Visible 论文反射建模思想的 `Object-Aware Reflectance Gate`，而不是复用当前 gate。

#### 12.13.5 当前基线与下一步方向修正

- 当前基线固定为 `plain true P2-P5 c256 BiFPN`，即 `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`。
- 暂不继续做 `c192` 宽度消融，后续所有结构改动都应在 `c256` 基线上做，避免主线再次发散。
- plain c256 已略高于 OA gate c256，说明现有 `ObjectAwareNIRGateConcat` 的贡献不稳定，甚至可能因额外 gate 干扰强 BiFPN 的尺度融合。
- 下一步优先补充两个内容：
  - `BiFPN` 动态尺度边权重分析与可视化：读取 checkpoint 中每个 `WeightedFeatureFusion` 的归一化权重，观察 P2/P3/P4/P5 各融合节点实际依赖哪条路径，用于解释 true P2-P5 BiFPN 的有效性。
  - `Object-Aware` 方向重做为 `OA-Reflect gate`：不再复用当前单一 reflection gate，而是在 P2/P3 引入 luminance cue、reflection cue 和 object-prior gate，保持 P4/P5 plain concat 与 c256 BiFPN 主体不变。

#### 12.13.6 已新增：BiFPN 权重分析脚本

- 新增脚本：`scripts/iddaw/analyze_bifpn_weights.py`
- 功能：
  - 加载 `best.pt` 或 `last.pt`
  - 遍历模型中的 `WeightedFeatureFusion`
  - 导出 `raw_weight`、`relu_weight`、`normalized_weight`
  - 输出 `bifpn_weights.json`、`bifpn_weights.csv`
  - 若环境安装 `matplotlib`，额外输出 `bifpn_weights.png`
- 推荐对当前 c256 基线执行：

```bash
python scripts/iddaw/analyze_bifpn_weights.py \
  --weights runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge2/weights/best.pt \
  --out runs/analysis/bifpn_weights/c256_baseline_best \
  --title "c256 true P2-P5 BiFPN baseline"
```

- 论文分析重点：
  - `p2_out_fuse` 是否更依赖 `P2_in` 还是 `P3_td_up`
  - `p3_out_fuse` 是否吸收了 `P2_out_down`
  - `p4_out_fuse` 是否仍主要依赖原始 `P4_in`
  - `p5_out_fuse` 是否从 `P4_out_down` 获益
  - 两个 repeat block 的权重是否从浅层局部细节逐步转向深层语义路径

#### 12.13.7 已新增：OA-Reflect c256 候选结构

- 新增 mode：`bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge`
- 新增配置：`configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_reflect_c256_6cls_personmerge.yaml`
- 新增模块：`ObjectAwareReflectanceGateConcat`
- 结构保持：
  - RGB full branch 不变
  - Light NIR branch 不变
  - `BiFPNP2P5([P2, P3, P4, P5], out_channels=256, repeats=2)` 不变
  - 四尺度 refine head 通道仍为 `256`
  - Detect head 不变
- 唯一结构变化：
  - `P2/P3`: `Concat(RGB, NIR)` 替换为 `ObjectAwareReflectanceGateConcat(RGB, NIR)`
  - `P4/P5`: 继续使用 plain `Concat`
- OA-Reflect 的实现意图：
  - `luminance cue`: 对 NIR 特征做低频平滑后提取亮度/低照信息
  - `reflection cue`: 使用 `NIR - smooth(NIR)` 提取高频材质/边缘响应
  - `object prior gate`: 由 RGB context、luminance、reflection 及其差异预测空间目标先验
  - `channel gate`: 估计每个 NIR 通道的有效性
  - 输出采用残差式调制，使初始行为接近 plain concat，降低当前 OA gate 对强基线产生副作用的风险

#### 12.13.8 下一轮执行计划

1. 先在远端对 `scripts/iddaw/analyze_bifpn_weights.py` 跑当前 c256 基线 `best.pt`，把 `json/csv/png` 保存到 `runs/analysis/bifpn_weights/c256_baseline_best`。
2. 做 `OA-Reflect c256` 冒烟测试，确认 `parse_model` 能按输入层通道正确构造 `ObjectAwareReflectanceGateConcat`，四尺度 Detect 正常。
3. 若冒烟通过，再用同基线配方训练：`100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1`。
4. 判定标准仍以 plain c256 为唯一主基线：`OA-Reflect c256` 必须超过 `mAP50-95 = 0.47976`，并且 `person/motorcycle` 至少不退化，才进入论文主线。
5. 如果 `OA-Reflect c256` 仍无收益，下一步不继续堆 gate，而是实现 box foreground mask 弱监督，让 object-aware 从隐式自学习变成检测框引导。

### 12.14 OA-Reflect c256 训练结果与下一步计划

#### 12.14.1 训练完成状态

- mode：`bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-reflect-c256-6cls-personmerge3`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-reflect-c256-6cls-personmerge3`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge_e100_20260427_183250.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge_e100_20260427_183250.meta`
- W&B run：`lbtbfnh9`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/lbtbfnh9`
- 运行配置：`100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1`
- 训练状态：`100` epoch 正常完成，`best.pt` 与 `last.pt` 已导出
- 模型规模：
  - 训练构建 summary：`596 layers / 11,511,754 parameters / 80.85 GFLOPs`
  - `best.pt` 复验 summary：`464 layers / 11,492,426 parameters / 125.03 GFLOPs`
  - W&B 记录：`11,511,754` parameters / `126.33` GFLOPs

#### 12.14.2 主指标结果

- `results.csv` 最优 epoch：
  - epoch `95`
  - `Precision = 0.76680`
  - `Recall = 0.61125`
  - `mAP50 = 0.68145`
  - `mAP50-95 = 0.47710`
- `results.csv` 最后 epoch：
  - epoch `100`
  - `Precision = 0.75851`
  - `Recall = 0.60514`
  - `mAP50 = 0.67385`
  - `mAP50-95 = 0.47284`
- `best.pt` 复验：
  - `Precision = 0.767`
  - `Recall = 0.611`
  - `mAP50 = 0.682`
  - `mAP50-95 = 0.477`

#### 12.14.3 类别指标

- `person`: `mAP50 = 0.564`, `mAP50-95 = 0.299`
- `motorcycle`: `mAP50 = 0.576`, `mAP50-95 = 0.265`
- `car`: `mAP50 = 0.890`, `mAP50-95 = 0.686`
- `truck`: `mAP50 = 0.626`, `mAP50-95 = 0.473`
- `bus`: `mAP50 = 0.667`, `mAP50-95 = 0.542`
- `autorickshaw`: `mAP50 = 0.765`, `mAP50-95 = 0.598`

#### 12.14.4 与当前主基线对比

- 相比 plain c256 baseline（`mAP50 = 0.69135`，`mAP50-95 = 0.47976`）：
  - `mAP50 -0.00976`
  - `mAP50-95 -0.00266`
  - 参数量增加约 `+1.39M`
  - W&B GFLOPs 从 `85.271` 增加到 `126.33`
- 类别对比：
  - `person`: `0.561 / 0.296 -> 0.564 / 0.299`，略有改善
  - `motorcycle`: `0.580 / 0.275 -> 0.576 / 0.265`，小目标关键类退化
  - `car`: `0.888 / 0.686 -> 0.890 / 0.686`，基本持平
  - `truck`: `0.636 / 0.476 -> 0.626 / 0.473`，略降
  - `bus`: `0.669 / 0.538 -> 0.667 / 0.542`，高 IoU 略升
  - `autorickshaw`: `0.815 / 0.604 -> 0.765 / 0.598`，mAP50 明显下降
- 结论：`OA-Reflect c256` 没有超过 plain c256 主基线，不能作为下一阶段主结构。它只对 `person` 和 `bus` 的 `mAP50-95` 有轻微收益，但代价是整体 `mAP50`、`motorcycle` 和 `autorickshaw` 下降，且复杂度明显增加。

#### 12.14.5 与旧 OA gate c256 对比

- 相比旧 OA gate c256（`mAP50 = 0.68654`，`mAP50-95 = 0.47814`）：
  - `mAP50 -0.00509`
  - `mAP50-95 -0.00104`
- 类别层面：
  - `person`: `0.569 / 0.298 -> 0.564 / 0.299`，高 IoU 基本持平
  - `motorcycle`: `0.586 / 0.274 -> 0.576 / 0.265`，退化
  - `car`: `0.893 / 0.695 -> 0.890 / 0.686`，退化
  - `truck`: `0.618 / 0.453 -> 0.626 / 0.473`，改善
  - `bus`: `0.684 / 0.545 -> 0.667 / 0.542`，略降
  - `autorickshaw`: `0.769 / 0.600 -> 0.765 / 0.598`，略降
- 结论：`OA-Reflect` 比旧 OA gate 更贴近论文动机，但实测并没有带来稳定指标提升。当前 object-aware 的问题不应继续靠增加无监督 gate 复杂度解决。

#### 12.14.6 BiFPN 权重分析

- `plain c256` 权重目录：`runs/analysis/bifpn_weights/c256_baseline_best`
- `OA gate c256` 权重目录：`runs/analysis/bifpn_weights/oagate_c256_best`
- `OA-Reflect c256` 权重目录：`runs/analysis/bifpn_weights/oa_reflect_c256_best`
- 后续 BiFPN 权重分析仅保存本地 `json/csv/png`，不再上传 W&B。

关键现象：

- plain c256 中，`p5_out_fuse` 两个 block 都几乎完全依赖 `P4_out_down`：
  - block0：`P5_in = 0.0%`，`P4_out_down = 99.8%`
  - block1：`P5_in = 0.0%`，`P4_out_down = 99.9%`
- 旧 OA gate c256 中，第二个 block 的 `p5_out_fuse` 回退到原始 `P5_in`：
  - block0：`P5_in = 0.0%`，`P4_out_down = 99.8%`
  - block1：`P5_in = 80.9%`，`P4_out_down = 17.9%`
- OA-Reflect c256 中，这个问题被修复：
  - block0：`P5_in = 0.0%`，`P4_out_down = 99.9%`
  - block1：`P5_in = 8.2%`，`P4_out_down = 91.8%`

解释：

- OA-Reflect 确实比旧 OA gate 更好地保持了 `P2/P3 -> P4 -> P5` 的 bottom-up 细节传播路径。
- 但指标仍未超过 plain c256，说明当前性能瓶颈已经不是 BiFPN 尺度路径被破坏，而是 OA-Reflect 对 P2/P3 特征的调制没有转化为更好的检测收益。
- 这也说明“仅靠更复杂的无监督 object-aware gate”不足以形成稳定贡献；下一步必须引入更明确的 object prior 或更轻量的约束方式。

#### 12.14.7 当前结论

- 当前正式主基线继续固定为：`bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`
- 当前最强主指标仍是 plain c256：`mAP50 = 0.69135`，`mAP50-95 = 0.47976`
- 旧 OA gate c256 与 OA-Reflect c256 均未超过 plain c256：
  - 旧 OA gate：`0.68654 / 0.47814`
  - OA-Reflect：`0.68145 / 0.47710`
- `OA-Reflect` 不进入主结果表的最优结构，但可作为 object-aware 改造的失败/消融证据。
- 论文主贡献应继续围绕 `true P2-P5 BiFPN c256` 和动态尺度权重可解释性展开；Object-Aware 只能作为附加探索，除非后续弱监督版本证明有效。

#### 12.14.8 下一步计划

1. 暂停继续堆叠更复杂的无监督 OA gate。
2. 保留 plain c256 作为唯一主基线，下一轮如果要继续 Object-Aware，优先做 `foreground-aware weak supervision`：
   - 用 GT box 生成 P2/P3 前景 mask
   - 对 OA spatial/object gate 加轻量 BCE 或 soft target 约束
   - 初始 `lambda` 建议 `0.01`
   - 只训练一个 `P2/P3 foreground-supervised OA` 对照，不改 BiFPN 和 head
3. 如果暂时不改训练 loss，则下一步更稳的是做解释性补强：
   - 导出 plain c256、旧 OA gate、OA-Reflect 三者的 `bifpn_weights.png`
   - 形成论文图：`P2/P3 bottom-up path dominates P4/P5`
   - 重点说明 true P2-P5 BiFPN 如何服务 IDD-AW 的小目标和非结构化道路场景
4. 下一轮训练优先级：
   - 第一优先：`foreground-supervised OA gate c256`
   - 第二优先：plain c256 的可视化与错误案例分析
   - 暂不建议继续做 `OA-Reflect + 更大通道`、`P4/P5 gate` 或 `P6`

### 12.15 Foreground-Supervised OA-Reflect c256 实现记录

#### 12.15.1 新增内容

- 新增 mode：`bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge`
- 新增配置：`configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_fg_c256_6cls_personmerge.yaml`
- 新增模块：`ObjectAwareForegroundReflectanceGateConcat`
- 基线来源：基于 `bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge` 修改。

结构保持不变：

- RGB full branch 不变
- Light NIR branch 不变
- `BiFPNP2P5([P2, P3, P4, P5], out_channels=256, repeats=2)` 不变
- 四尺度 refine head 通道仍为 `256`
- Detect head 不变
- `P4/P5` 继续使用 plain `Concat`

唯一新增约束：

- `P2/P3` 使用 `ObjectAwareForegroundReflectanceGateConcat`
- forward 中保留 OA-Reflect 的 `luminance / reflectance / object-prior` 计算方式
- 训练时记录 `object_prior gate`
- loss 阶段把 batch 中归一化 `xywh` GT box 栅格化到 gate 尺度，生成二值 foreground mask
- 对 gate 与 foreground mask 计算 balanced BCE，默认 `lambda=0.01`
- 辅助损失加到训练总 loss，同时作为第四个日志项 `fg_gate_loss` 输出；普通模式仍保持 `box_loss / cls_loss / dfl_loss` 三项
- 第一个有效训练 batch 会打印一次量级检查：`aux_loss`、`scaled_aux_loss=aux_loss*batch_size`、`det_loss`、`batch_size`

#### 12.15.2 设计目的

- 上一版 `OA-Reflect c256` 的 object prior 完全靠自学习，虽然更符合 NIR 论文的反射/目标先验叙事，但没有稳定超过 plain c256。
- 这一版把检测标注中的框信息作为弱监督先验，让 object-aware gate 至少学会在目标区域提高响应，避免 gate 被背景纹理、道路边缘或车灯眩光主导。
- 当前版本仍只约束 `P2/P3`，因为小目标和边界细节主要依赖浅层高分辨率特征；不对 `P4/P5` 加 gate，避免再次破坏高层语义路径。

#### 12.15.3 建议训练配方

先做远端冒烟：

```bash
WANDB_ENABLED=0 IMGSZ=800 OPTIMIZER=Adam LR0=0.01 BATCH=20 CLOSE_MOSAIC=15 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge 1 0,1
```

冒烟通过后跑正式对照：

```bash
WANDB_ENABLED=1 IMGSZ=800 OPTIMIZER=Adam LR0=0.01 BATCH=20 CLOSE_MOSAIC=15 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge 100 0,1
```

判定标准：

- 第一基线仍是 plain c256：`mAP50 = 0.69135`，`mAP50-95 = 0.47976`
- OA-FG c256 必须超过 plain c256 的 `mAP50-95`
- `person` 和 `motorcycle` 不应低于 plain c256，否则不进入主线
- 若只提升 `person` 但整体不升，可作为 object-aware 消融保留，但不能作为主结构

### 12.16 Foreground-Supervised OA-Reflect c256 训练结果与下一步计划

#### 12.16.1 训练完成状态

- mode：`bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge`
- run：`iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-fg-c256-6cls-personmerge3`
- 结果目录：`runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-fg-c256-6cls-personmerge3`
- 日志：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge_e100_20260427_222112.stdout.log`
- meta：`/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/iddaw/bifpn_only_light_nir_p2p5_oa_fg_c256_yolo11s_6cls_personmerge_e100_20260427_222112.meta`
- W&B run：`ovdxoser`
- W&B 链接：`https://wandb.ai/hilbertschopenhauer-no/iddaw-rgbnir-formal/runs/ovdxoser`
- 运行配置：`100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1`
- 训练状态：`100` epoch 正常完成，`best.pt` 与 `last.pt` 已导出
- 模型规模：
  - 训练构建 summary：`596 layers / 11,511,754 parameters / 80.85 GFLOPs`
  - `best.pt` 复验 summary：`464 layers / 11,492,426 parameters / 125.03 GFLOPs`
  - W&B 记录：`11,511,754` parameters / `126.33` GFLOPs

#### 12.16.2 主指标结果

- `results.csv` 最优 epoch：
  - epoch `96`
  - `Precision = 0.71942`
  - `Recall = 0.61375`
  - `mAP50 = 0.67895`
  - `mAP50-95 = 0.47658`
- `results.csv` 最后 epoch：
  - epoch `100`
  - `Precision = 0.75766`
  - `Recall = 0.59968`
  - `mAP50 = 0.68188`
  - `mAP50-95 = 0.47539`
- `best.pt` 复验：
  - `Precision = 0.718`
  - `Recall = 0.615`
  - `mAP50 = 0.679`
  - `mAP50-95 = 0.476`
- `fg_gate_loss` 变化：
  - epoch `1`: `0.02192`
  - epoch `99` 最低：`0.01145`
  - epoch `100`: `0.01149`
  - 说明前景 mask 约束确实被优化，但没有转化为总体检测指标提升。

#### 12.16.3 类别指标

- `person`: `mAP50 = 0.567`, `mAP50-95 = 0.298`
- `motorcycle`: `mAP50 = 0.575`, `mAP50-95 = 0.262`
- `car`: `mAP50 = 0.884`, `mAP50-95 = 0.689`
- `truck`: `mAP50 = 0.631`, `mAP50-95 = 0.492`
- `bus`: `mAP50 = 0.622`, `mAP50-95 = 0.504`
- `autorickshaw`: `mAP50 = 0.794`, `mAP50-95 = 0.612`

#### 12.16.4 与当前主基线对比

- 相比 plain c256 baseline（`mAP50 = 0.69135`，`mAP50-95 = 0.47976`）：
  - `mAP50 -0.01240`
  - `mAP50-95 -0.00318`
  - `Precision -0.00626`
  - `Recall -0.01530`
- 类别对比：
  - `person`: `0.561 / 0.296 -> 0.567 / 0.298`，略有改善
  - `motorcycle`: `0.580 / 0.275 -> 0.575 / 0.262`，关键小目标类退化
  - `car`: `0.888 / 0.686 -> 0.884 / 0.689`，高 IoU 略升
  - `truck`: `0.636 / 0.476 -> 0.631 / 0.492`，高 IoU 改善
  - `bus`: `0.669 / 0.538 -> 0.622 / 0.504`，明显退化
  - `autorickshaw`: `0.815 / 0.604 -> 0.794 / 0.612`，`mAP50-95` 略升但 `mAP50` 下降
- 结论：`foreground-supervised OA-Reflect c256` 未超过 plain c256 主基线，不能晋级为主结构。它对 `person`、`truck` 和 `autorickshaw` 的高 IoU 指标有局部正收益，但 `motorcycle` 与 `bus` 退化，整体 `mAP50-95` 仍低于主基线。

#### 12.16.5 与 OA-Reflect 对比

- 相比 OA-Reflect c256（`mAP50 = 0.68145`，`mAP50-95 = 0.47710`）：
  - `mAP50 -0.00250`
  - `mAP50-95 -0.00052`
- 现象：
  - `fg_gate_loss` 明确下降，说明辅助监督生效。
  - 总体指标没有提升，说明当前二值 box foreground mask 可能把 gate 约束得过硬，削弱了上下文和非框内反射/边缘信息。
  - `motorcycle` 退化尤其关键，说明对小目标的粗框 mask 监督可能引入噪声，不能简单认为“前景越强越好”。

#### 12.16.6 当前结论

- 当前正式主基线仍固定为 `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge`。
- 当前最强主指标仍是 plain c256：`mAP50 = 0.69135`，`mAP50-95 = 0.47976`。
- 三个 object-aware 变体均未超过 plain c256：
  - 旧 OA gate c256：`0.68654 / 0.47814`
  - OA-Reflect c256：`0.68145 / 0.47710`
  - OA-FG c256：`0.67895 / 0.47658`
- `Object-Aware` 目前适合作为消融探索和论文动机补充，不应覆盖主贡献。主贡献仍应写成 `true P2-P5 BiFPN c256 + Light NIR branch`，并用 BiFPN 动态尺度权重分析解释其有效性。

#### 12.16.7 下一轮训练计划

第一优先级：补 plain c256 的 `close_mosaic=20` 对照，验证当前最强结构是否还能吃到增强策略收益。

```bash
WANDB_ENABLED=1 IMGSZ=800 OPTIMIZER=Adam LR0=0.01 BATCH=20 CLOSE_MOSAIC=20 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge 100 0,1
```

目的：

- 只改变 `close_mosaic: 15 -> 20`，其余保持当前最强 plain c256 配方不变。
- 如果超过 `mAP50-95 = 0.47976`，后续所有 OA 对照都应以该结果作为新主基线。
- 如果不提升，则固定 `close_mosaic=15`，避免继续在增强策略上发散。

第二优先级：如果仍需要继续 Object-Aware 方向，建议不要再直接加重 gate，而是做一个更保守的弱监督版本。

- 建议模式：`OA-FG P2-only, lambda=0.003`
- 结构：只对 `P2` 的 object-prior gate 加 foreground loss，`P3` 回退为普通 OA-Reflect，`P4/P5` 仍 plain concat。
- 配方：`100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1`
- 判定：必须同时超过 plain c256 的 `mAP50-95`，并且 `motorcycle` 不低于 `0.275`，否则停止继续 object-aware 训练。

执行顺序建议：

1. 先跑 `plain c256 + close_mosaic=20`，因为这是当前主线最少变量、最高价值的确认实验。
2. 再根据结果决定是否实现并训练 `OA-FG P2-only, lambda=0.003`。
3. 在下一轮启动前，优先修复 W&B 中 `fg_gate_loss` 的显示别名问题，保证后续辅助损失可以直接在面板中观察。

### 12.17 Multi-Scale OA Soft-Prior c256 实现记录

#### 12.17.1 新增内容

- 新增 mode：`bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge`
- 新增配置：`configs/models/yolo11s_rgbnir_bifpn_p2p5_light_nir_oa_ms_softprior_c256_6cls_personmerge.yaml`
- 新增模块：
  - `ObjectAwareMultiScaleReflectanceGateConcat`
  - `ObjectAwareMultiScaleSoftPriorGateConcat`
- 基线来源：基于 `bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge` 修改。

结构保持不变：

- RGB full branch 不变
- Light NIR branch 不变
- `BiFPNP2P5([P2, P3, P4, P5], out_channels=256, repeats=2)` 不变
- 四尺度 refine head 通道仍为 `256`
- Detect head 不变
- `P4/P5` 继续使用 plain `Concat`

核心变化：

- `P2/P3` 的 NIR luminance/reflectance 分解从单尺度 `3x3 avg_pool` 改成多尺度平滑：

```python
smooth3 = avg_pool3(nir_feat)
smooth5 = avg_pool5(nir_feat)
smooth7 = avg_pool7(nir_feat)
nir_smooth = smooth_fuse(cat([smooth3, smooth5, smooth7]))
```

- `P2` 使用 `ObjectAwareMultiScaleSoftPriorGateConcat`，soft-prior loss 权重为 `0.003`。
- `P3` 使用 `ObjectAwareMultiScaleSoftPriorGateConcat`，soft-prior loss 权重为 `0.001`。
- soft prior 不再是二值 box mask，而是框中心高、边缘低的 Gaussian-like target；目的是保留 object-aware 约束，同时减少对小目标上下文的硬压制。
- `fg_gate_loss` 继续加入总 loss，并额外在 W&B 中写入 `fg_gate_loss` 和 `loss/fg_gate` 两个别名，避免只显示 `train/fg_gate_loss` 时面板不可见。

#### 12.17.2 设计目的

- 旧 `OA-Reflect c256` 的单尺度 `3x3` luminance/reflectance 分解对 IDD-AW 的复杂光照、雨雾低照和不同大小目标不够稳，`motorcycle` 明显退化。
- `OA-FG c256` 的二值 box foreground 监督确实让 `fg_gate_loss` 下降，但对 `motorcycle` 和 `bus` 有副作用，说明硬前景 mask 约束过强。
- 本版把两个问题同时收敛到更温和的实现：
  - 多尺度 smooth 提升 NIR 低频/高频分解稳定性。
  - soft prior 用中心高、边缘低的目标替代硬 box mask。
  - P3 权重低于 P2，避免过度干扰语义层。

#### 12.17.3 建议训练配方

冒烟：

```bash
WANDB_ENABLED=0 IMGSZ=800 OPTIMIZER=Adam LR0=0.01 BATCH=20 CLOSE_MOSAIC=15 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge 1 0,1
```

正式训练：

```bash
WANDB_ENABLED=1 IMGSZ=800 OPTIMIZER=Adam LR0=0.01 BATCH=20 CLOSE_MOSAIC=15 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge 100 0,1
```

判定标准：

- 第一基线仍是 plain c256：`mAP50 = 0.69135`，`mAP50-95 = 0.47976`
- 新版必须超过 plain c256 的 `mAP50-95` 才能作为主结构候选。
- `motorcycle mAP50-95` 不应低于 `0.275`；否则说明 P3 soft prior 仍然伤害关键小目标。
- 如果只提升 `person`，但总体和 `motorcycle` 不升，则只作为 Object-Aware 消融保留。

### 12.18 YOLO11s 统一基线补训计划

为避免后续主表混入历史 YOLO11n 结果，新增两个 YOLO11s 基线 mode：

- `nir_yolo11s_6cls_personmerge`
- `rgbnir_yolo11s_6cls_personmerge`

下一轮排队训练统一使用 YOLO11s 规模、6 类 personmerge、同一训练配方：

- `rgb_yolo11s_6cls_personmerge`
- `nir_yolo11s_6cls_personmerge`
- `rgbnir_yolo11s_6cls_personmerge`
- `rgbnir_light_nir_yolo11s_6cls_personmerge`

统一配方：

- `100 epoch`
- `imgsz=800`
- `optimizer=Adam`
- `lr0=0.01`
- `batch=20`
- `close_mosaic=15`
- `device=0,1`
- `IDDAW_CLASS_SCHEMA=6cls_personmerge`
- `WANDB_ENABLED=1`

执行方式：等待当前 `bifpn_only_light_nir_p2p5_oa_ms_softprior_c256_yolo11s_6cls_personmerge` 完成后，按上述顺序依次启动，每两个训练之间间隔 `8` 分钟。
