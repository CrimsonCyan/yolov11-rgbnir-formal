# YOLOv11 RGB+NIR Formal

这是当前正式使用的 `RGB+NIR` 双流检测工程。

它不是把上游 `YOLOv11-RGBT` 整仓继续堆改下去，而是只抽取当前真正需要的部分，形成一个更干净的开发目录。当前活动主线的规范语义已经收口为 `RGBNIR`，也就是 `RGB backbone + NIR backbone / branch`。后续所有正式代码改动，默认都应写在这个目录下，而不是再回到 `yolov11-rgbt-formal`。

当前默认数据根解析顺序为：

1. 优先读取环境变量 `IDDAW_YOLO_ROOT`
2. 若未设置，则优先使用全量 all-weather 根目录 `E:\毕设\code\datasets\iddaw_all_weather_full_yolov11_rgbnir`
3. 若全量根不存在，再回退到旧的 `iddaw_fog_full_yolov11_rgbnir` 与 `iddaw_fog_yolov11_rgbnir`

## 目录职责

- `ultralytics/`
  - 保留上游检测框架和多模态读取能力
- `formal_rgbnir/`
  - 当前项目自己的公共逻辑
- `configs/models/`
  - 当前项目自己的模型配置
- `scripts/iddaw/`
  - 当前 `IDD-AW FOG` 子集实验入口
- `tools/export_iddaw_fog_to_yolo.py`
  - 把 `visnir-det` 的 paired JSON 导出为当前工程可读数据
  - 当前默认导出 `person / rider / motorcycle / car / truck / bus / autorickshaw` 共 `7` 类

## 当前第一阶段入口

导出数据：

```powershell
python tools/export_iddaw_fog_to_yolo.py --clean
```

跑第一阶段三组训练型基线：

```powershell
python scripts/iddaw/run_experiment.py --mode rgb --task train --epochs 1
python scripts/iddaw/run_experiment.py --mode nir --task train --epochs 1
python scripts/iddaw/run_experiment.py --mode rgbnir --task train --epochs 1
```

第二阶段入口：

```powershell
python scripts/iddaw/run_experiment.py --mode input_fusion --task train --epochs 1
python scripts/iddaw/run_experiment.py --mode light_gate --task train --epochs 1
python scripts/iddaw/run_experiment.py --mode proposed_lite_yolo11s_6cls_personmerge --task train --epochs 1
python scripts/iddaw/run_experiment.py --mode bifpn_only_light_nir_yolo11s_6cls_personmerge --task train --epochs 1
python scripts/iddaw/run_experiment.py --mode decision_fusion --task val
```

统一入口现在支持可选覆盖：

```powershell
python scripts/iddaw/run_experiment.py --mode proposed_lite_yolo11s_6cls_personmerge --task train --epochs 70 --imgsz 640 --optimizer SGD --batch 24
python scripts/iddaw/run_experiment.py --mode bifpn_only_light_nir_yolo11s_6cls_personmerge --task train --epochs 70 --imgsz 640 --optimizer SGD --batch 24
python scripts/iddaw/run_experiment.py --mode rgb_yolo11s_6cls_personmerge --task train --epochs 100 --imgsz 800 --optimizer Adam --batch 16
```

## 远端 nohup 训练

远端推荐不用前台直接跑，而是用 `nohup` 后台启动，并把控制台输出写到日志文件。

启动：

```bash
bash scripts/iddaw/launch_nohup_train.sh rgbnir 1 0
```

查看最新日志：

```bash
bash scripts/iddaw/tail_latest_log.sh rgbnir
```

查看最近一次任务元信息：

```bash
bash scripts/iddaw/show_latest_run.sh rgbnir
```

停止最近一次任务：

```bash
bash scripts/iddaw/stop_latest_train.sh rgbnir
```

## 当前原则

- `yolov11-rgbt-formal` 保留为上游参考副本
- `yolov11-rgbnir-formal` 作为新的活动工程
- 后续 `light gate / BiFPN / 质量感知跨模态注意力` 都在本目录内继续实现
