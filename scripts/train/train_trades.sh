alphas=(0.1 0.2 0.5 1 1.5 2)

for alpha in "${alphas[@]}"; do
    bash scripts/experiment.sh \
        --mode_type train \
        --attack_type identity \
        --train_dataset cifar10 \
        --eval_dataset cifar10C \
        --model_type alexnet \
        --loss_type trades \
        --eval_noise gaussian_noise.npy \
        --epochs 50 \
        --valid_size 0.2 \
        --eval_interval 1 \
        --alpha $alpha
done