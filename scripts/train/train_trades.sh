alphas=(0.1 0.25 1 3)
lrs=(0.005 0.01 0.03 0.05)

for lr in "${lrs[@]}"; do
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
            --lr $lr \
            --alpha $alpha
    done
done

        python3 main.py \
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
            --lr 0.03 \
            --alpha 0.25