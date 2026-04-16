# YOLOv11 RGB+NIR Formal Overview

## 定位

`yolov11-rgbnir-formal` 现在采用“上游内核 + formal 自定义层”的结构。

- 上游内核：`ultralytics/`
  - 保留 YOLOv11-RGBT 已有的检测框架、多模态读取逻辑、训练验证推理闭环。
- formal 自定义层：只放当前论文主线真正需要的内容
  - `formal_rgbnir/`
  - `configs/models/`
  - `scripts/iddaw_fog/`
  - `tools/export_iddaw_fog_to_yolo.py`

这样处理的目的不是重写上游，而是把“能复用的公共能力”和“论文相关改造”明确分层，避免后续继续把自定义代码散落到上游根目录。

## 当前保留的上游能力

当前真正需要继续依赖的上游部分只有四类：

1. `ultralytics/`
   - 检测训练、验证、推理主流程
   - 模型构建与 head/assigner/NMS
   - 多模态 loader 中 `use_simotm`
2. `ultralytics/cfg/models/11/`
   - 单模态 RGB 与 Gray 版 YOLO11 配置
3. 上游已经实现的 `RGBT` 读取逻辑
   - 通过 `pairs_rgb_ir=['visible', 'infrared']`
   - 通过 `use_simotm='RGBT'` 将 RGB 和灰度 NIR 合并为 4 通道输入
4. 可复用的训练接口
   - `YOLO(...).train()`
   - `YOLO(...).val()`
   - `YOLO(...).predict()`

## 当前不再作为主线维护的部分

以下内容保留在仓库里，但不作为当前正式实验主线：

- 根目录大量示例脚本，如 `train_RGBT.py`、`train_MCF_demo.py`
- 其他数据集转换脚本，如 `transform_MCF.py`
- 分类、分割、姿态、导出等与当前主线无直接关系的脚本
- 论文图片、文档展示材料

它们可以保留作参考，但后续正式实验不应再从这些脚本直接起步。

## formal 自定义层

### `formal_rgbnir/`

这里放项目级公共逻辑，而不是训练入口。

- `formal_rgbnir/iddaw_fog.py`
  - 解析数据根目录
  - 动态生成运行时 dataset yaml
  - 统一定义 `rgb / nir / rgbnir` 三种模式
  - 统一定义 train / val / predict 的公共参数
  - 统一指向项目自定义模型配置

### `configs/models/`

- `configs/models/yolo11_rgbnir_midfusion_plain.yaml`
  - 当前正式第一阶段 `RGB+NIR dual-stream plain baseline`
  - 放在项目配置目录，而不是塞回 `ultralytics/cfg`
  - 便于后续继续演进到 `light gate / BiFPN / 注意力`

### `scripts/iddaw_fog/`

这里放真正面向实验执行的入口。

- `run_experiment.py`
  - 统一入口，支持 `train / val / predict`
- `train_rgb.py`
- `train_nir.py`
- `train_rgbnir_plain.py`

这些脚本只处理当前 `IDD-AW FOG` 子集，不再和其它示例数据集混在一起。

### `tools/export_iddaw_fog_to_yolo.py`

- 负责把 `visnir-det` 里的 paired JSON 导出成 YOLOv11-RGBT 可读目录
- 当前是真正的数据桥接点
- 导出时使用 `sample_id` 作为文件名前缀，避免不同序列同名帧互相覆盖

## 当前建议的工作流

1. 先导出数据

```powershell
python tools/export_iddaw_fog_to_yolo.py --clean
```

2. 再跑三组第一阶段基线

```powershell
python scripts/iddaw_fog/run_experiment.py --mode rgb --task train --epochs 1
python scripts/iddaw_fog/run_experiment.py --mode nir --task train --epochs 1
python scripts/iddaw_fog/run_experiment.py --mode rgbnir --task train --epochs 1
```

3. 训练完成后，用同一入口做验证或预测

```powershell
python scripts/iddaw_fog/run_experiment.py --mode rgbnir --task val --weights <best.pt>
python scripts/iddaw_fog/run_experiment.py --mode rgbnir --task predict --weights <best.pt>
```

## 后续扩展位置

后面真正继续改模型时，只需要看这两个位置：

1. `configs/models/`
   - 增加 `light gate`
   - 增加 `BiFPN`
   - 增加正式版 `Proposed`
2. `formal_rgbnir/`
   - 增加模式开关
   - 增加统一实验配置与结果命名

这样可以保证：

- 上游核心检测代码尽量少动
- 我们自己的改造路径清晰可控
- 论文方法演进不会继续把仓库结构搞乱
