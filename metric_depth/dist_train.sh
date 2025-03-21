#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=150
bs=1
gpus=2
lr=0.0000003
encoder=vitl
dataset=booster # hypersim
img_size=518
min_depth=0.001
max_depth=20 # 80 for virtual kitti
pretrained_from=../checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
save_path=output/booster/metric_sigmoid # exp/vkitti

mkdir -p $save_path

python3 -m torch.distributed.run \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22065 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 22065 2>&1 | tee -a $save_path/$now.log
