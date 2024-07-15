alphas=(0.1 0.25 0.5 1 2 3)

for alpha in "${alphas[@]}"; do
    bash scripts/experiment.sh \
        --mode_type train \
        --attack_type identity \
        --train_dataset cifar10 \
        --eval_dataset cifar10C \
        --model_type resnet18 \
        --loss_type trades \
        --eval_noise gaussian_noise.npy \
        --epochs 100 \
        --valid_size 0.2 \
        --eval_interval 1 \
        --lr 0.01 \
        --alpha $alpha
done
