## TODO: Fix generic print in plotting/print_eval.py and rewrite this

python3 plotting/print_adaptive.py \
    --mode_type train \
    --attack_type identity \
    --model_type resnet18 \
    --train_dataset cifar10 \
    --eval_dataset cifar10C \
    --loss_type trades \
    --eval_noise gaussian_noise.npy \
    --epochs 100 \
    --valid_size 0.2 \
    --eval_interval 1 \
    --w_noise 0.1 \
    --lr 0.01 \
    --alpha 1 \
    --tau1 10
