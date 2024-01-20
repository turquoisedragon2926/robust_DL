# severities=(0.05 0.1 0.25 0.5 0.75 1)
# train_noises=("gaussian" "uniform" "shot" "blur" "random")

severities=(0.05 0.1)
train_noises=("dynamicBlur")

# module load conda
# conda activate robust_DL
# module load pytorch/2.0.1

for train_noise in "${train_noises[@]}"; do
    for severity in "${severities[@]}"; do
        python3 main.py \
            --mode_type train \
            --attack_type pgd \
            --model_type alexnet \
            --train_dataset cifar10 \
            --eval_dataset cifar10C \
            --loss_type adaptive \
            --train_noise $train_noise \
            --eval_noise gaussian_noise.npy \
            --epochs 10 \
            --valid_size 0.2 \
            --eval_interval 1 \
            --severity $severity \
            --w_noise 0.1 \
            --tau1 10
    done
done

python3 main.py \
            --mode_type train \
            --attack_type pgd \
            --model_type alexnet \
            --train_dataset cifar10 \
            --eval_dataset cifar10C \
            --loss_type adaptive \
            --eval_noise gaussian_noise.npy \
            --epochs 10 \
            --valid_size 0.2 \
            --eval_interval 1 \
            --w_noise 0.1 \
            --tau1 10
