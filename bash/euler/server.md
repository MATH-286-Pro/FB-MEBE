# Tars And Case

First, clone the repo on tars/case server

Then for Docker, change the following files in cloned repo:
```bash
- change .dockerignore: ignore exp_local directory  
- change docker/.container.cfg: set x11_forwarding_enabled = 0 (if no GUI needed)  
```

Then on tars/case server run:  
```bash
python docker/container.py start  # pack and start container  (the image name should be isaac-lab-base:latest)
docker images                     # check image created
python docker/container.py enter  # enter container
mount | grep /workspace           # check mount points
```

# Euler
```bash
# First run scripts to set image on Euler
#    this will automaticly build docker image, 
#    then build apptainer image, 
#    then upload image to euler
./bash/euler/build_image.sh

# To run single job on euler
#    video on euler is all closed
#    video record on euler is extremely slow
./bash/euler/run.sh

# To run multi jobs on euler
./bash/euler/run_multi.sh
```

<!-- We assume we already have the image (isaac-lab-base:latest)

```bash
# on your PC
docker save isaac-lab-base:latest -o eulertest.tar      # save image to tar file
scp eulertest.tar euler:/cluster/home/jiajhu/images     # note: you also need to have "euler" entry in your ~/.ssh/config

# on euler server
# Go to folder with .tar and run build command
apptainer build ./eulertest.sif docker-archive:./eulertest.tar
``` -->

Commands
```bash

# To view which ashre account you can use to run Euler, and how many "average" GPU that share account has
# it also tells you how to change default share account if you have multi
my_share_info

squeue         # view your jobs
scancel <ID>   # cancel job with ID

# 当你有多个 account 都有使用 Euler 的额度
# 如何指定默认使用哪个 account
# 这里演示的是默认使用 es_coros 这个 account
echo account=es_coros >> $HOME/.slurm/defaults

# 之后直接使用 
sbatch job.sh
# Slurm 会自动当作
sbatch -A es_coros job.sh


# Euler 上有很多工程师在维护各种软件
# 如果你想访问这些软件，需要使用 module 命令访问
module load stack/2024-06 python/3.12.8
module avail # 查看所有可用模块

```