#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --job-name losstype=${1}_noisetype=${2}_hyperparam=${3}_epochs=${4}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH --account=m3863_g
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0

srun --export=ALL python3 experiment.py $1 $2 $3 $4

exit 0
EOT
