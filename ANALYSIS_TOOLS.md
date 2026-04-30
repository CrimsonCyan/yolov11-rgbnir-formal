# 实验与分析工具使用说明

本文档整理当前仓库中已经可用的训练、监控与分析工具，重点覆盖 `BiFPN` 权重分析和 `Object-Aware` 小目标诊断。默认远端环境为：

```bash
ssh lyh
cd /data1/lvyanhu/code/yolov11-rgbnir-formal
```

远端推荐 Python：

```bash
/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python
```

## 1. 统一训练入口

### 工具

```bash
scripts/iddaw/run_experiment.py
scripts/iddaw/launch_nohup_train.sh
```

### 作用

- `run_experiment.py` 是统一的训练、验证、预测入口。
- `launch_nohup_train.sh` 是远端后台训练封装，会生成日志、pid、meta，并自动维护 `latest_<mode>.*` 软链接。
- 冒烟测试默认可关闭 W&B，正式长训默认可开启 W&B。

### 输入

核心参数：

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `mode` | 实验模式名 | `bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge` |
| `epochs` | 训练轮数 | `1`, `100` |
| `device` | 训练设备 | `0`, `0,1` |
| `IMGSZ` | 输入尺寸 | `800` |
| `OPTIMIZER` | 优化器 | `Adam`, `SGD`, `AdamW` |
| `LR0` | 初始学习率 | `0.01` |
| `BATCH` | batch size | `20` |
| `CLOSE_MOSAIC` | 关闭 mosaic 的 epoch 数 | `15`, `20` |
| `WANDB_ENABLED` | 是否启用 W&B | `0`, `1` |

### 输出

| 文件/目录 | 说明 |
| --- | --- |
| `runs/IDD_AW/<run_name>/results.csv` | 每个 epoch 的训练和验证指标 |
| `runs/IDD_AW/<run_name>/weights/best.pt` | 最优权重 |
| `runs/IDD_AW/<run_name>/weights/last.pt` | 最后一轮权重 |
| `remote_logs/iddaw/<mode>_e<epochs>_<time>.stdout.log` | 完整训练日志 |
| `remote_logs/iddaw/<mode>_e<epochs>_<time>.meta` | 本次训练元信息 |
| `remote_logs/iddaw/latest_<mode>.stdout.log` | 最新日志软链接 |
| `remote_logs/iddaw/latest_<mode>.pid` | 最新 pid 软链接 |
| `remote_logs/iddaw/latest_<mode>.meta` | 最新 meta 软链接 |

### 使用示例

1 epoch 冒烟：

```bash
WANDB_ENABLED=0 IMGSZ=800 OPTIMIZER=Adam LR0=0.01 BATCH=20 CLOSE_MOSAIC=15 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge 1 0,1
```

100 epoch 正式训练：

```bash
WANDB_ENABLED=1 IMGSZ=800 OPTIMIZER=Adam LR0=0.01 BATCH=20 CLOSE_MOSAIC=15 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh bifpn_only_light_nir_p2p5_oa_smallprior_p2only_p3plain_c256_yolo11s_6cls_personmerge 100 0,1
```

Ultralytics 原生严格续训：

```bash
WANDB_ENABLED=1 IMGSZ=800 BATCH=20 IDDAW_CLASS_SCHEMA=6cls_personmerge \
bash scripts/iddaw/launch_nohup_train.sh <mode> 100 0,1 runs/IDD_AW/<run_name>/weights/last.pt
```

## 2. BiFPN 权重分析

### 工具

```bash
scripts/iddaw/analyze_bifpn_weights.py
```

### 作用

导出已训练模型中 `WeightedFeatureFusion` 的可学习边权重，用于分析 BiFPN 在 `P2/P3/P4/P5` 之间实际更依赖哪些尺度路径。

该工具默认只保存本地分析结果，不上传 W&B。只有显式加 `--wandb` 才会上传。

### 输入

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--weights` | 是 | `best.pt` 或 `last.pt` 路径 |
| `--out` | 否 | 输出目录，默认 `runs/analysis/bifpn_weights` |
| `--title` | 否 | 图标题 |
| `--no-plots` | 否 | 不生成 png 图 |
| `--wandb` | 否 | 将表格、图片和 artifact 上传 W&B |
| `--wandb-project` | 否 | W&B project，默认 `iddaw-rgbnir-formal` |
| `--wandb-group` | 否 | W&B group，默认 `analysis` |
| `--wandb-run-name` | 否 | W&B run 名称 |
| `--wandb-tags` | 否 | W&B 标签 |

### 输出

| 文件 | 内容 |
| --- | --- |
| `bifpn_weights.json` | 按 fusion node 分组的边权重 |
| `bifpn_weights.csv` | 扁平表格，适合后续统计或写论文表格 |
| `bifpn_weights.png` | 横向条形图，可直接用于可视化分析 |

CSV 关键列：

| 列 | 含义 |
| --- | --- |
| `module` | 模块完整名称 |
| `block` | BiFPN repeat 编号 |
| `node` | 融合节点，如 `p2_out_fuse`, `p5_out_fuse` |
| `edge_label` | 输入边语义，如 `P2_in`, `P3_td_up`, `P4_out_down` |
| `raw_weight` | 原始可学习权重 |
| `relu_weight` | ReLU 后权重 |
| `normalized_weight` | 归一化后实际融合权重 |

### 使用示例

分析当前 BiFPN c256 baseline：

```bash
/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python scripts/iddaw/analyze_bifpn_weights.py \
  --weights runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge3/weights/best.pt \
  --out runs/analysis/bifpn_weights/c256_repeat_best \
  --title "BiFPN c256 repeat"
```

只导出表格，不生成图片：

```bash
/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python scripts/iddaw/analyze_bifpn_weights.py \
  --weights runs/IDD_AW/<run_name>/weights/best.pt \
  --out runs/analysis/bifpn_weights/<analysis_name> \
  --no-plots
```

如确实需要上传 W&B：

```bash
/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python scripts/iddaw/analyze_bifpn_weights.py \
  --weights runs/IDD_AW/<run_name>/weights/best.pt \
  --out runs/analysis/bifpn_weights/<analysis_name> \
  --wandb \
  --wandb-run-name "<analysis_name>"
```

### 解释口径

- `normalized_weight` 越高，说明该融合节点越依赖对应输入边。
- 如果 `P5_in` 接近 `0`，说明模型主动压低原始 P5 语义输入，更依赖 `P4_out_down` 生成 P5 输出。
- 如果 `P2_in` 和 `P3_td_up` 同时较高，说明小目标路径主要依赖高分辨率细节与 P3 语义补充。
- floor 类实验应重点看最小边权重是否被强制抬高，以及这种抬高是否破坏原本的自适应选择。

## 3. OA 小目标与 Gate 诊断

### 工具

```bash
scripts/iddaw/analyze_oa_small_targets.py
```

### 作用

对一个或多个模型进行统一验证集诊断，输出：

- `all/small/medium/large` 分尺度 AP。
- 各类别 AP，尤其 `person` 与 `motorcycle`。
- 指定置信度下的 precision / recall。
- OA gate 的整体均值、GT 框内均值、GT 框外均值。
- `person/motorcycle` 框内 gate 均值。
- residual-reflect 分支的修正量强度。

该工具用于回答：OA 是否真的关注小目标区域，以及 residual correction 是否在小目标框内强于背景。

### 输入

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--case` | 否 | 多模型输入，格式为 `name|mode|weights`，可重复 |
| `--name` | 否 | 单模型名称 |
| `--mode` | 单模型时必填 | mode 名称，用于确定数据集和输入模式 |
| `--weights` | 单模型时必填 | 权重路径 |
| `--out` | 否 | 输出目录，默认 `runs/analysis/oa_small_targets` |
| `--imgsz` | 否 | 推理尺寸，默认 `800` |
| `--conf` | 否 | AP 收集用低阈值，默认 `0.001` |
| `--pr-conf` | 否 | precision/recall 统计阈值，默认 `0.25` |
| `--iou` | 否 | NMS IoU，默认 `0.7` |
| `--device` | 否 | 推理设备，默认 `0`；CPU 诊断可设为 `cpu` |
| `--half` | 否 | CUDA 设备上启用 FP16 推理，降低显存占用 |
| `--per-image` | 否 | 逐图调用 `predict()`，速度较慢但可避免长序列推理显存缓存增长 |
| `--max-images` | 否 | 调试时限制图片数量，默认全量验证集 |

### 输出

每个 case 会生成一个子目录：

```text
runs/analysis/oa_small_targets/<case_name>/
```

子目录内容：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 完整分析结果 |
| `metrics_by_class_area.csv` | 各类别、各尺度 AP 与 PR |
| `gate_summary.csv` | OA gate 和 residual correction 汇总 |

总输出：

| 文件 | 内容 |
| --- | --- |
| `all_cases_summary.json` | 所有 case 的汇总 JSON |

`metrics_by_class_area.csv` 关键列：

| 列 | 含义 |
| --- | --- |
| `class` | 类别名，含 `mean` |
| `area` | `all/small/medium/large` |
| `gt_count` | 该类别和尺度下的 GT 数量 |
| `AP50` | IoU=0.5 的 AP |
| `mAP50_95` | IoU=0.5:0.95 的平均 AP |
| `precision_at_conf` | `--pr-conf` 下 precision |
| `recall_at_conf` | `--pr-conf` 下 recall |

`gate_summary.csv` 关键项：

| 项 | 含义 |
| --- | --- |
| `gate_all` | 全图 object gate 均值 |
| `gate_inside` | 所有 GT 框内 gate 均值 |
| `gate_outside` | GT 框外 gate 均值 |
| `gate_person_inside` | person 框内 gate 均值 |
| `gate_motorcycle_inside` | motorcycle 框内 gate 均值 |
| `residual_all` | 全图 residual correction 平均绝对响应 |
| `residual_inside` | GT 框内 residual correction 响应 |
| `residual_outside` | GT 框外 residual correction 响应 |
| `num_gate_samples` | 有效采样图片数 |

### 使用示例

单模型分析：

```bash
/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python scripts/iddaw/analyze_oa_small_targets.py \
  --name p2_resreflect \
  --mode bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge \
  --weights runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-ms-softprior-resreflect-p2only-c256-6cls-personmerge3/weights/best.pt \
  --out runs/analysis/oa_small_targets/p2_resreflect
```

多模型对比：

```bash
/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python scripts/iddaw/analyze_oa_small_targets.py \
  --case "bifpn_only|bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge|runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge3/weights/best.pt" \
  --case "p2_resreflect|bifpn_only_light_nir_p2p5_oa_ms_softprior_resreflect_p2only_c256_yolo11s_6cls_personmerge|runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-ms-softprior-resreflect-p2only-c256-6cls-personmerge3/weights/best.pt" \
  --case "p3_plain|bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge|runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-resreflect-p2only-p3plain-c256-6cls-personmerge2/weights/best.pt" \
  --out runs/analysis/oa_small_targets/core_compare \
  --device 0 \
  --half \
  --per-image
```

快速调试 20 张图：

```bash
/data1/lvyanhu/miniconda3/envs/visnir-exp/bin/python scripts/iddaw/analyze_oa_small_targets.py \
  --name debug \
  --mode bifpn_only_light_nir_p2p5_oa_resreflect_p2only_p3plain_c256_yolo11s_6cls_personmerge \
  --weights runs/IDD_AW/<run_name>/weights/best.pt \
  --device 0 \
  --half \
  --per-image \
  --max-images 20
```

显存紧张时建议优先使用 `--half --per-image`。如果 GPU 被占用，也可以使用 `--device cpu` 做小样本诊断，但全量验证会明显变慢。

### 解释口径

- `gate_inside > gate_outside`：说明对象先验确实更关注 GT 区域。
- `gate_person_inside` 或 `gate_motorcycle_inside` 不高：说明 OA 对小目标没有形成有效选择性。
- `residual_inside > residual_outside`：说明 residual correction 主要作用于目标区域。
- `residual_inside` 很低：说明 residual 分支接近关闭，即使 gate 有响应也未明显改变特征。
- 当前面积划分由 `formal_rgbnir/box_ops.py` 控制：`small < 32^2`，`medium < 96^2`，其余为 `large`；面积基于验证图像原始像素坐标计算。

## 4. 日志与进程辅助工具

### 查看最新训练元信息

```bash
bash scripts/iddaw/show_latest_run.sh <mode>
```

输入：

- `<mode>`：实验模式名。

输出：

- `remote_logs/iddaw/latest_<mode>.meta` 内容，包括 pid、日志路径、数据集路径、W&B 状态、训练命令等。

### 跟踪最新训练日志

```bash
bash scripts/iddaw/tail_latest_log.sh <mode>
```

输出：

- 实时 `tail -f remote_logs/iddaw/latest_<mode>.stdout.log`。

### 停止最新训练

```bash
bash scripts/iddaw/stop_latest_train.sh <mode>
```

输出：

- 读取 `remote_logs/iddaw/latest_<mode>.pid` 并发送 `kill`。

注意：

- 该脚本只停止 latest pid 指向的进程。
- 若训练是 DDP 多进程，优先确认 `pgrep -af run_experiment.py` 和 GPU 进程是否已完全退出。

## 5. Decision Fusion 工具

### 工具

```bash
scripts/iddaw/run_decision_fusion.py
formal_rgbnir/decision_fusion.py
```

### 作用

离线决策级融合 RGB 与 NIR 的检测输出，用作非端到端融合 baseline。

### 输入

默认脚本内部调用：

```python
run_decision_fusion(split="val", device="0")
```

如需要自定义权重或 split，建议通过 `scripts/iddaw/run_experiment.py --mode decision_fusion` 入口传参。

### 输出

输出目录：

```text
runs/IDD_AW/iddaw-yolo11n-decision-fusion/
```

常见产物包括：

- 融合后预测结果。
- 指标 JSON / CSV。
- 控制台打印的 metrics。

## 6. 数据导出与子集生成工具

### IDD-AW YOLO 导出

```bash
tools/export_iddaw_all_weather_to_yolo.py
tools/export_iddaw_fog_to_yolo.py
```

作用：

- 将 IDD-AW 语义/实例标注转换为 YOLO 检测格式。
- 当前正式实验默认使用已经导出的 `6` 类数据，不建议在结果对比阶段重新导出或改映射。

### bbox 子集生成

```bash
tools/generate_iddaw_bbox_subset.py
```

作用：

- 从已导出的 IDD-AW YOLO 数据中生成 bbox 子集或调试子集。
- 适合快速验证数据读取、标注转换和可视化。

## 7. 常用远端检查命令

确认没有训练队列或训练进程：

```bash
pgrep -af 'queue_.*iddaw|queue_oa|run_experiment.py|launch_nohup_train' || true
```

查看 GPU：

```bash
nvidia-smi
```

查看最新 run：

```bash
ls -td runs/IDD_AW/* | head -n 20
```

查看某个 run 的最后指标：

```bash
tail -n 5 runs/IDD_AW/<run_name>/results.csv
```

查看某个训练日志尾部：

```bash
tail -n 100 remote_logs/iddaw/<log_file>.stdout.log
```

## 8. 当前建议的分析顺序

1. 训练结束后先看 `results.csv` 与 `best.pt` 复验表，确认总体 `mAP50-95` 和 `person/motorcycle`。
2. 对所有 BiFPN 系列模型运行 `analyze_bifpn_weights.py`，确认尺度路径是否仍以 `P2/P3/P4` 为主。
3. 对所有 OA 系列模型运行 `analyze_oa_small_targets.py`，确认 gate 是否真的关注小目标框内区域。
4. 若 OA 的 `gate_inside > gate_outside` 但小目标 AP 不升，说明先验学到了区域但没有改善定位，应在论文中把 OA 定位为可解释辅助模块。
5. 若 OA 的 `gate_person_inside` / `gate_motorcycle_inside` 不高，说明 OA 不适合作为小目标增强主因，论文主线应转向 `Light NIR branch + BiFPN`。
