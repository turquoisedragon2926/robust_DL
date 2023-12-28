models=('alexnet', 'resnet18')
losses=('adaptive', 'ce', 'trades')

for loss in "${losses[@]}"; do
    for model in "${models[@]}"; do
        bash scripts/experiment.sh \
            --mode_type train \
            --train_dataset imagenet \
            --eval_dataset imagenetC \
            --model_type $model \
            --loss_type $loss \
            --train_noise blur \
            --eval_noise gaussian_noise.npy \
            --epochs 50 \
            --valid_size 0.2 \
            --eval_interval 1 \
            --severity 0.05 \
            --w_noise 0.1 \
            --tau1 10
    done
done
