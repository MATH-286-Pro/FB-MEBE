#!/bin/bash

seeds=(0 42 17 5 24)

for seed in "${seeds[@]}"; do
    echo "[INFO] Submitting job with seed=${seed}"
    
    ./docker/cluster/cluster_interface.sh job \
        --config-name=Isaaclab_pretrain_config_go2 \
        env.video_train=False \
        env.video_eval=False \
        env.seed=${seed} \
        train.machine=cluster \
        train.num_train_steps=300000
done

echo "[INFO] All jobs submitted!"