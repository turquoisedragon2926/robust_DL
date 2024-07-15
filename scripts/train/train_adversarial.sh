lrs=(0.05 0.1)

for lr in "${lrs[@]}"; do
    bash scripts/experiment.sh \
        --mode_type train \
        --attack_type identity \
        --train_dataset cifar10 \
        --eval_dataset cifar10C \
        --model_type resnet18 \
        --loss_type adversarial \
        --eval_noise gaussian_noise.npy \
        --epochs 100 \
        --valid_size 0.2 \
        --eval_interval 1 \
        --lr $lr
done
