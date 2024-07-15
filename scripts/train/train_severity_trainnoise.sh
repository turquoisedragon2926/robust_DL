severities=(0.05 0.1 0.25 0.5 0.75 1)
train_noises=("uniform") # ("blur" "gaussian" "uniform" "shot" "dynamicBlur" "random")

for train_noise in "${train_noises[@]}"; do
    for severity in "${severities[@]}"; do
        bash scripts/experiment.sh \
            --mode_type train \
            --attack_type identity \
            --train_dataset cifar10 \
            --eval_dataset cifar10C \
            --model_type resnet18 \
            --loss_type adaptive \
            --train_noise $train_noise \
            --eval_noise gaussian_noise.npy \
            --epochs 100 \
            --valid_size 0.2 \
            --eval_interval 1 \
            --severity $severity \
            --w_noise 0.1 \
            --tau1 10
    done
done
