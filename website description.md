# RGB-NIR 检测网站说明

## 1. 网站功能

本网站是一个基于 Gradio 的 RGB-NIR 配对图像检测演示工具，入口文件为：

```text
apps/rgbnir_detect_web.py
```

启动脚本为：

```text
scripts/iddaw/launch_web_demo.sh
```

网站用于调用当前最优 RGB-NIR 检测模型进行推理演示。输入为一张 RGB 图像和一张 NIR 图像，或两个分别存放 RGB/NIR 图像的文件夹；输出为在 RGB 原图上绘制检测框后的可视化结果。NIR 图像只作为模型输入参与推理，不单独输出标注图。

网站页面包含两个功能页：

- 单对 RGB/NIR 图像检测：上传一张 RGB 图像和一张 NIR 图像，输出 RGB 标注图、检测结果表格和 JSON 结果。
- 批量 RGB/NIR 文件夹检测：输入远端服务器上的 RGB 文件夹路径和 NIR 文件夹路径，按文件名 stem 精确匹配图像对，批量输出检测结果。

## 2. 默认模型与参数

默认模型说明：

```text
RGB + 轻量 NIR 分支 + P2-P5 BiFPN + MGF
```

默认权重路径：

```text
/data1/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-oa-segmask-fg050-p2only-c256-r4-red1-8cls-personmerge-traffic/weights/best.pt
```

该权重已另存一份为：

```text
/data1/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/MGFDet/weights/best.pt
```

默认推理参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `imgsz` | `640` | 模型推理输入尺寸 |
| `device` | `0` | 默认使用 GPU 0 |
| `conf` | `0.25` | 置信度阈值 |
| `iou` | `0.7` | NMS IoU 阈值 |
| `max_det` | `300` | 单张图最多保留检测框数量 |

## 3. 启动方法

在远端服务器执行：

```bash
ssh lyh
cd /data1/lvyanhu/code/yolov11-rgbnir-formal
bash scripts/iddaw/launch_web_demo.sh
```

脚本会在后台启动网站，并将日志写入：

```text
/data1/lvyanhu/code/yolov11-rgbnir-formal/remote_logs/web_demo/rgbnir_detect_web_<timestamp>.log
```

默认绑定地址为：

```text
127.0.0.1:7860
```

因此建议在本地使用 SSH 隧道访问：

```bash
ssh -L 7860:127.0.0.1:7860 lyh
```

随后在本地浏览器打开：

```text
http://127.0.0.1:7860
```

## 4. 启动参数覆盖

启动脚本支持通过环境变量覆盖默认设置：

```bash
PORT=7860 DEVICE=1 IOU=0.7 CONF=0.2 bash scripts/iddaw/launch_web_demo.sh
```

常用环境变量如下：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `127.0.0.1` | 服务绑定地址 |
| `PORT` | `7860` | 服务端口 |
| `DEVICE` | `0` | 推理设备 |
| `WEIGHTS` | 默认 best.pt | 权重路径 |
| `IMGSZ` | `640` | 推理输入尺寸 |
| `CONF` | `0.25` | 置信度阈值 |
| `IOU` | `0.7` | NMS IoU 阈值 |
| `MAX_DET` | `300` | 最大检测数量 |
| `PYTHON_BIN` | `visnir-exp` 环境 Python | Python 可执行文件 |

如果希望直接使用命名后的 MGFDet 权重，可以执行：

```bash
WEIGHTS=/data1/lvyanhu/code/yolov11-rgbnir-formal/runs/IDD_AW/MGFDet/weights/best.pt \
bash scripts/iddaw/launch_web_demo.sh
```

## 5. 关闭网站服务

关闭所有网站进程：

```bash
ssh lyh 'pkill -f apps/rgbnir_detect_web.py'
```

查看当前网站进程：

```bash
ssh lyh 'ps -ef | grep apps/rgbnir_detect_web.py | grep -v grep'
```

如果只想关闭某一个 PID，例如 `2473437`：

```bash
ssh lyh 'kill 2473437'
```

## 6. 单对图像检测

使用步骤：

1. 打开网站后进入“单对 RGB/NIR 图像”标签页。
2. 在 `RGB image` 中上传 RGB 图像。
3. 在 `NIR image` 中上传对应的 NIR 图像。
4. 根据需要调整 `Confidence`、`NMS IoU`、`Device` 等参数。
5. 点击 `Run detection`。

输出内容：

- `Annotated RGB`：在 RGB 原图上绘制检测框后的图像。
- `Detections`：检测结果表格，包含类别、置信度和框坐标。
- `JSON result`：完整检测结果、参数和 warning。
- `Download annotated image`：标注图下载文件。

单图检测结果保存目录：

```text
/data1/lvyanhu/code/yolov11-rgbnir-formal/runs/web_detect/single/<timestamp>/
```

## 7. 批量文件夹检测

批量检测用于处理两个文件夹中的 RGB/NIR 配对图像。输入必须是远端服务器上的路径，例如：

```text
/data1/lvyanhu/code/datasets/.../visible/val
/data1/lvyanhu/code/datasets/.../nir/val
```

匹配规则：

- 递归搜索 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff` 文件。
- RGB 文件和 NIR 文件按相同文件名 stem 匹配。
- 示例：`val_FOG_105_00000026.png` 与 `val_FOG_105_00000026.png` 会被视为一对。
- 未匹配文件不会参与检测，会记录到 `summary.json`。

输出目录：

```text
/data1/lvyanhu/code/yolov11-rgbnir-formal/runs/web_detect/<timestamp>/
```

输出文件：

| 文件或目录 | 内容 |
|---|---|
| `annotated/` | 批量生成的 RGB 标注图 |
| `detections.csv` | 所有检测框表格 |
| `summary.json` | 批量检测摘要、未匹配文件、尺寸 warning 和失败样本 |
| `rgbnir_detections.zip` | 上述结果的压缩包 |

## 8. 输入图像处理逻辑

RGB 图像读取为三通道彩色图像。NIR 图像无论原始为单通道还是三通道，都会统一转换为单通道灰度图。

如果 RGB 和 NIR 尺寸不一致，程序会将 NIR resize 到 RGB 原始尺寸，并在 `summary.json` 或单图 JSON 的 `warnings` 中记录该情况。

模型实际输入为四通道：

```text
BGR + NIR(gray)
```

该格式与当前仓库中 Ultralytics 自定义 RGB-NIR 推理预处理保持一致。推理后，检测框坐标对应 RGB 原图坐标，并绘制在 RGB 图像上。

## 9. 注意事项

- 网站只用于推理演示，不会训练模型，也不会修改权重。
- 默认不自动 fallback 到其他权重。如果默认 `best.pt` 不存在，程序会直接报错，避免误用旧模型。
- 网站默认只绑定 `127.0.0.1`，需要通过 SSH 隧道访问，不建议直接暴露公网端口。
- 批量检测时，文件名 stem 必须一致，否则不会匹配。
- 页面中的图像预览框为固定大小，上传大图时只缩放显示，不会改变原始推理图像。
