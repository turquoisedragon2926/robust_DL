#!/bin/bash

lrs=(0.005)  
severities=(0.05)  

for lr in "${lrs[@]}"; do
    for severity in "${severities[@]}"; do
        bash scripts/experiment.sh \
            --mode_type train \
            --attack_type identity \
            --train_dataset cifar10 \
            --eval_dataset cifar10C \
            --model_type resnet18 \
            --loss_type adaptive \
            --train_noise uniform \
            --eval_noise gaussian_noise.npy \
            --epochs 1 \
            --valid_size 0.2 \
            --eval_interval 1 \
            --severity $severity \
            --lr $lr
    done
done
