python3 main.py \
    --mode_type train \
    --attack_type identity \
    --model_type resnet18 \
    --train_dataset augcifar10 \
    --eval_dataset cifar10C \
    --loss_type augmix \
    --eval_noise gaussian_noise.npy \
    --epochs 1 \
    --valid_size 0.2 \
    --eval_interval 1 \
