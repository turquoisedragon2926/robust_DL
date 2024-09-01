lrs=(0.005 0.01 0.03 0.05)
severities=(0.05 0.1 0.25 0.5 0.75 1)

# lrs=(0.03)
# severities=(0.05)

for lr in "${lrs[@]}"; do
    for severity in "${severities[@]}"; do
        bash scripts/experiment.sh \
            --mode_type train \
            --attack_type identity \
            --train_dataset imagenet \
            --eval_dataset imagenetC \
            --model_type resnet18 \
            --loss_type adaptive \
            --train_noise uniform \
            --eval_noise fog \
            --epochs 100 \
            --valid_size 0.2 \
            --eval_interval 1 \
            --severity $severity \
            --lr $lr \
            --num_samples 1
    done
done
