# 开始训练

cd /data1/lvyanhu/code/yolov11-rgbnir-formal

bash scripts/iddaw/launch_nohup_train.sh \
  bifpn_only_light_nir_p2p5_oa_segmask_fg050_p2only_c256_r4_reduction1_yolo11s_8cls_personmerge_traffic \
  100 \
  0,1

# 停止训练
cd /data1/lvyanhu/code/yolov11-rgbnir-formal

MODE=bifpn_only_light_nir_p2p5_oa_segmask_fg050_p2only_c256_r4_reduction1_yolo11s_8cls_personmerge_traffic
PID=$(cat remote_logs/iddaw/latest_${MODE}.pid)

kill -TERM $PID

watch -n 1 nvidia-smi

kill -TERM $PID