models=('alexnet', 'resnet18')
losses=('adaptive', 'ce', 'trades')

module load conda
conda activate robust_DL
module load pytorch/2.0.1

for loss in "${losses[@]}"; do
    for model in "${models[@]}"; do
        python3 main.py \
            --mode_type train \
            --model_type alexnet \
            --train_dataset cifar10 \
            --eval_dataset cifar10C \
            --loss_type adaptive \
            --train_noise blur \
            --eval_noise gaussian_noise.npy \
            --epochs 2 \
            --valid_size 0.2 \
            --eval_interval 1 \
            --severity 0.05 \
            --w_noise 0.1 \
            --tau1 10
    done
done
