# Run all the following command on your local computer
# 在本地电脑上运行以下命令，不是服务器上运行

# build docker image
python docker/container.py start

# you need to modify the USERNAME in ./docker/cluster/.env.cluster first
./docker/cluster/cluster_interface.sh push base