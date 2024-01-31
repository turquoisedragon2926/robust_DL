w_noises=(0.05 0.1 0.25 0.5 0.75 1.0)

for w_noise in "${w_noises[@]}"; do
    bash scripts/experiment.sh \
        --mode_type train \
        --attack_type identity \
        --train_dataset cifar10 \
        --eval_dataset cifar10C \
        --model_type resnet18 \
        --loss_type adaptive \
        --train_noise uniform \
        --eval_noise gaussian_noise.npy \
        --epochs 100 \
        --valid_size 0.2 \
        --eval_interval 1 \
        --severity 0.05 \
        --lr 0.03 \
        --w_noise "$w_noise" \
        --tau1 10
done
