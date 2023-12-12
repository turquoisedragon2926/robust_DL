#!/bin/bash

# Default values for arguments
mode_type=""
model_type=""
loss_type=""
train_noise=""
eval_noise=""
epochs=""
valid_size=""
eval_interval=""
model_checkpoint=""
optimizer_checkpoint=""
alpha=""
severity=""
w_noise=""
tau1=""
tau2=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode_type) mode_type="$2"; shift ;;
        --model_type) model_type="$2"; shift ;;
        --loss_type) loss_type="$2"; shift ;;
        --train_noise) train_noise="$2"; shift ;;
        --eval_noise) eval_noise="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --valid_size) valid_size="$2"; shift ;;
        --eval_interval) eval_interval="$2"; shift ;;
        --model_checkpoint) model_checkpoint="$2"; shift ;;
        --optimizer_checkpoint) optimizer_checkpoint="$2"; shift ;;
        --alpha) alpha="$2"; shift ;;
        --severity) severity="$2"; shift ;;
        --w_noise) w_noise="$2"; shift ;;
        --tau1) tau1="$2"; shift ;;
        --tau2) tau2="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Your SBATCH script
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --job-name ${mode_type}_${loss_type}_${eval_noise}_${epochs}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=0:30:00
#SBATCH --account=m3863_g
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0

module load conda
conda activate robust_DL
module load pytorch/2.0.1

python3 main.py \
    --mode_type $mode_type \
    --model_type $model_type \
    --loss_type $loss_type \
    --train_noise $train_noise \
    --eval_noise $eval_noise \
    --epochs $epochs \
    --valid_size $valid_size \
    --eval_interval $eval_interval \
    --model_checkpoint $model_checkpoint \
    --optimizer_checkpoint $optimizer_checkpoint \
    --alpha $alpha \
    --severity $severity \
    --w_noise $w_noise \
    --tau1 $tau1 \
    --tau2 $tau2

exit 0
EOT
