#!/usr/bin/env bash

# Load required modules
module load stack/2024-06
module load eth_proxy

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --job-name=fb_pretrain
#SBATCH --output=fb_pretrain_%j.out
#SBATCH --error=fb_pretrain_%j.err

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh


# If you want to use other types of GPUs on ETH Euler
# Please refer to https://docs.hpc.ethz.ch/batchsystem/slurm/
# - v100
# - pro_6000
# - rtx_4090
# - rtx_3090
# - rtx_2080
# - rtx_6000