python3 plotting/all_losses.py \
    --mode_type train \
    --attack_type identity \
    --train_dataset cifar10 \
    --eval_dataset cifar10C \
    --model_type resnet18 \
    --eval_noise gaussian_noise.npy \
    --epochs 100 \
    --valid_size 0.2 \
    --eval_interval 1 \
