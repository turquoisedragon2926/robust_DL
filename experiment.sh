#!/bin/bash

# Default values for arguments
modetype=""
losstype=""
noisetype=""
model_checkpoint=""
alpha=""
severity=""
w_noise=""
tau1=""
tau2=""
epochs=""

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --modetype) modetype="$2"; shift ;;
        --losstype) losstype="$2"; shift ;;
        --noisetype) noisetype="$2"; shift ;;
        --model_checkpoint) model_checkpoint="$2"; shift ;;
        --alpha) alpha="$2"; shift ;;
        --severity) severity="$2"; shift ;;
        --w_noise) w_noise="$2"; shift ;;
        --tau1) tau1="$2"; shift ;;
        --tau2) tau2="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
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
#SBATCH --job-name ${modetype}_${losstype}_${noisetype}_${epochs}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=1:10:00
#SBATCH --account=m3863_g
#SBATCH --gpu-bind=none

export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0

module load conda
conda activate robust_DL
module load pytorch/2.0.1
python3 experiment.py --modetype $modetype --losstype $losstype --noisetype $noisetype --alpha $alpha --severity $severity --w_noise $w_noise --tau1 $tau1 --tau2 $tau2 --epochs $epochs --model_checkpoint $model_checkpoint

exit 0
EOT
