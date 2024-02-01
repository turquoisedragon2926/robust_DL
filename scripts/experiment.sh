#!/bin/bash

# Function to echo the flag and value only if the value is not empty
add_arg() {
    local arg_name="$1"
    local arg_value="$2"
    if [ -n "$arg_value" ]; then
        echo "--$arg_name $arg_value"
    fi
}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --*) arg_name="${1:2}"; declare "$arg_name"="$2"; shift ;;
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
#SBATCH --job-name ${mode_type}_${model_type}_${train_dataset}_${eval_dataset}_${loss_type}_${eval_noise}_${epochs}
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
#SBATCH --account=m3863_g
#SBATCH --gpu-bind=none

python3 main.py \
    $(add_arg mode_type "$mode_type") \
    $(add_arg attack_type "$attack_type") \
    $(add_arg model_type "$model_type") \
    $(add_arg train_dataset "$train_dataset") \
    $(add_arg eval_dataset "$eval_dataset") \
    $(add_arg loss_type "$loss_type") \
    $(add_arg train_noise "$train_noise") \
    $(add_arg eval_noise "$eval_noise") \
    $(add_arg epochs "$epochs") \
    $(add_arg valid_size "$valid_size") \
    $(add_arg eval_interval "$eval_interval") \
    $(add_arg model_checkpoint "$model_checkpoint") \
    $(add_arg optimizer_checkpoint "$optimizer_checkpoint") \
    $(add_arg lr "$lr") \
    $(add_arg alpha "$alpha") \
    $(add_arg severity "$severity") \
    $(add_arg num_samples "$num_samples") \
    $(add_arg w_noise "$w_noise") \
    $(add_arg tau1 "$tau1") \
    $(add_arg tau2 "$tau2")

exit 0
EOT
