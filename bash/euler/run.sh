# run in your local computer, not on euler
./docker/cluster/cluster_interface.sh job \
    --config-name=Isaaclab_pretrain_config_go2 \
    env.video_train=False \
    env.video_eval=False \
    train.machine=cluster

# example code with argument
# ./docker/cluster/cluster_interface.sh job --task Isaac-Velocity-Rough-Anymal-C-v0 --headless --video --enable_cameras