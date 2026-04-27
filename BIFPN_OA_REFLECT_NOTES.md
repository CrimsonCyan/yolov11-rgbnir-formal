# BiFPN 权重分析与 OA-Reflect 说明

## 1. BiFPN 权重分析脚本如何使用

脚本位置：

```bash
scripts/iddaw/analyze_bifpn_weights.py
```

它用于读取训练好的 `best.pt` 或 `last.pt`，遍历模型中的 `WeightedFeatureFusion`，导出每个 BiFPN fusion node 的可学习边权重。对当前 `true P2-P5 BiFPN c256` 基线，推荐在远端仓库根目录执行：

```bash
python scripts/iddaw/analyze_bifpn_weights.py \
  --weights runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge2/weights/best.pt \
  --out runs/analysis/bifpn_weights/c256_baseline_best \
  --title "c256 true P2-P5 BiFPN baseline"
```

如果只想导出表格，不生成图片：

```bash
python scripts/iddaw/analyze_bifpn_weights.py \
  --weights runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge2/weights/best.pt \
  --out runs/analysis/bifpn_weights/c256_baseline_best \
  --no-plots
```

## 2. 最终结果如何显现

脚本会在 `--out` 指定目录下生成以下文件：

```text
bifpn_weights.json
bifpn_weights.csv
bifpn_weights.png
```

`bifpn_weights.csv` 是最适合后续统计和画论文图的文件，核心列包括：

```text
module
block
node
edge_index
edge_label
raw_weight
relu_weight
normalized_weight
```

`normalized_weight` 是 BiFPN 实际用于融合的归一化权重。`edge_label` 已对 P2-P5 BiFPN 的典型路径做了语义命名，例如：

```text
p2_out_fuse: P2_in / P3_td_up
p3_out_fuse: P3_in / P3_td / P2_out_down
p4_out_fuse: P4_in / P4_td / P3_out_down
p5_out_fuse: P5_in / P4_out_down
```

`bifpn_weights.png` 是横向柱状图，直接展示每条尺度融合边的归一化权重。它主要用于回答：

```text
P2 是否真的被 P3 top-down 信息增强
P3 是否吸收了 P2 bottom-up 信息
P4/P5 是否仍主要依赖原始高层语义
第二个 BiFPN repeat 是否改变了尺度依赖关系
```

如果环境没有安装 `matplotlib`，脚本会跳过 `png`，但仍会正常输出 `json/csv`。

## 3. 是否能接入 W&B

可以。当前脚本已支持可选 W&B 记录。前提是远端环境已安装 `wandb` 并完成登录。

示例命令：

```bash
python scripts/iddaw/analyze_bifpn_weights.py \
  --weights runs/IDD_AW/iddaw-yolo11s-rgbnir-bifpn-only-light-nir-p2p5-c256-6cls-personmerge2/weights/best.pt \
  --out runs/analysis/bifpn_weights/c256_baseline_best \
  --title "c256 true P2-P5 BiFPN baseline" \
  --wandb \
  --wandb-project iddaw-rgbnir-formal \
  --wandb-group bifpn_weight_analysis \
  --wandb-run-name c256-bifpn-weight-analysis \
  --wandb-tags c256,bifpn,weight-analysis
```

W&B 中会记录：

```text
bifpn_weight_table
bifpn_weights_plot
fusion_nodes
fusion_edges
analysis artifact
```

其中 artifact 会包含本地生成的 `json/csv/png`。这适合后续把不同 checkpoint 的 BiFPN 权重分析放到同一个 W&B group 下对比，例如：

```text
c256 baseline best.pt
OA gate c256 best.pt
OA-Reflect c256 best.pt
```

注意：这个脚本是训练后的离线分析工具，不会自动挂到训练过程里逐 epoch 记录。当前更合理的用法是在每个实验完成后，对 `best.pt` 单独执行一次，这样不会增加训练开销，也避免 W&B 日志过重。

## 4. OA-Reflect 相比上一版 OAGate 的变化

上一版 `ObjectAwareNIRGateConcat` 的逻辑比较简单：

```text
RGB -> rgb_context
NIR -> reflectance
concat(rgb_context, reflectance, abs_diff)
channel_gate * spatial_gate
NIR *= gate
concat(RGB, gated NIR)
```

它的主要问题是 object-aware 仍然是隐式自学习，没有明确区分 NIR 中的低频亮度信息和高频反射/边缘信息。当前同配方实验中，plain c256 已略高于 OAGate c256，说明这个 gate 对强 BiFPN 基线没有稳定增益，甚至可能干扰了 P2/P3 的原始细节。

新版本 `ObjectAwareReflectanceGateConcat` 做了四个变化：

1. 增加 `luminance cue`：对 NIR 做平滑后提取低频亮度/低照信息。
2. 增加 `reflection cue`：使用 `NIR - smooth(NIR)` 提取高频材质、轮廓和边缘响应。
3. 增加 `object-prior gate`：由 `RGB context + luminance + reflection + cross-modal diff` 预测空间目标先验，让 gate 更接近 Object-Aware NIR-to-Visible 论文中的 object prior 思路。
4. 改成残差式调制：初始行为更接近 plain concat，降低 gate 早期训练破坏强基线特征的风险。

结构插入位置保持克制：

```text
P2/P3: ObjectAwareReflectanceGateConcat
P4/P5: plain Concat
BiFPNP2P5: 256 channels, repeat=2
Detect head: unchanged
```

这意味着 OA-Reflect 的实验变量只有 P2/P3 的 object-aware NIR 选择方式，不同时改变 BiFPN 宽度、head 或 NIR backbone。

## 5. 下一步实验建议

当前主基线固定为：

```text
bifpn_only_light_nir_p2p5_c256_yolo11s_6cls_personmerge
100 epoch, imgsz=800, Adam, lr0=0.01, batch=20, close_mosaic=15, device=0,1
mAP50-95 = 0.47976
```

下一步建议顺序：

1. 对当前 c256 baseline 的 `best.pt` 执行 BiFPN 权重分析，并上传 W&B。
2. 对 `bifpn_only_light_nir_p2p5_oa_reflect_c256_yolo11s_6cls_personmerge` 做 1 epoch 冒烟，确认建模、前向和四尺度 Detect 正常。
3. 冒烟通过后，使用同配方训练 OA-Reflect c256。
4. 只有当 OA-Reflect c256 超过 plain c256 的 `mAP50-95 = 0.47976`，并且 `person/motorcycle` 不退化，才把它作为 Object-Aware 主线。
5. 如果 OA-Reflect 仍无收益，下一步应转向 box foreground mask 弱监督，而不是继续增加 gate 复杂度。
