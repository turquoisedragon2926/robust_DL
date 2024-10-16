# python3 plotting/adaptive_lr_alpha.py \
#     --mode_type train \
#     --attack_type identity \
#     --train_dataset imagenet \
#     --eval_dataset imagenetC \
#     --model_type resnet18 \
#     --loss_type adaptive \
#     --train_noise uniform \
#     --eval_noise gaussian_noise.npy \
#     --epochs 50 \
#     --valid_size 0.2 \
#     --eval_interval 1 \
#     --severity 0.05 \
#     --w_noise 0.1 \
#     --lr 0.05 \
#     --tau1 10 \
#     --num_samples 1

# python3 plotting/adaptive_lr_alpha.py \
#     --mode_type train \
#     --attack_type identity \
#     --train_dataset imagenet \
#     --eval_dataset imagenetC \
#     --model_type resnet18 \
#     --loss_type adaptive \
#     --train_noise gaussian \
#     --eval_noise gaussian_noise.npy \
#     --epochs 50 \
#     --valid_size 0.2 \
#     --eval_interval 1 \
#     --severity 0.05 \
#     --w_noise 0.1 \
#     --lr 0.05 \
#     --tau1 10 \
#     --num_samples 1

python3 plotting/adaptive_lr_alpha.py \
    --mode_type train \
    --attack_type identity \
    --model_type resnet18 \
    --train_dataset cifar100 \
    --eval_dataset cifar100C \
    --loss_type augmix \
    --eval_noise gaussian_noise.npy \
    --epochs 80 \
    --lr 0.03 \
    --valid_size 0.2 \
    --eval_interval 20 \
