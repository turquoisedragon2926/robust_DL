python3 plotting/adaptive_lr_alpha.py \
    --mode_type train \
    --attack_type identity \
    --train_dataset cifar10 \
    --eval_dataset cifar10C \
    --train_noise uniform \
    --model_type resnet18 \
    --loss_type adaptive \
    --eval_noise gaussian_noise.npy \
    --epochs 100 \
    --valid_size 0.2 \
    --eval_interval 1 \
