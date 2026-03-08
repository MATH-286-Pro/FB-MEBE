#!/usr/bin/env bash

# set -euo pipefail
# shopt -s expand_aliases
# alias python='/workspace/isaaclab/_isaac_sim/python.sh'

# Note: if you want to run 2 parallel training on TARS or CASE server in CRL
#       you need to change env.device to "cuda:0" and run this script
#       And then change env.device to "cuda:1" and run again

seeds=(0 42 17 5 24)

for seed in "${seeds[@]}"; do
    python scripts/reinforcement_learning/fb_mod/pretrain.py \
        --config-name=Isaaclab_pretrain_config_go2 \
        env.video_train=False \
        env.video_eval=False \
        env.device="cuda:0" \
        train.num_train_steps=300000 \
        env.num_envs=2048 \
        agent.train.batch_size=4096 \
        env.seed=${seed}
done