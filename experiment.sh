#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --job-name modetype=${1}_losstype=${2}_noisetype=${3}_hyperparam=${4}_epochs=${5}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00
#SBATCH --account=m3863_g
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0

module load conda
conda activate robust_DL
module load pytorch/2.0.1
srun --export=ALL python3 experiment.py $1 $2 $3 $4 $5 $6

exit 0
EOT
