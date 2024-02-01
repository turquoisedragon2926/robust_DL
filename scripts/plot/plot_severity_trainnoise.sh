# remove train_noise and severity 
# bc we eval for all severity and 
# w_noise we have trained upon

# module load conda
# conda activate robust_DL
# module load pytorch/2.0.1

# python3 plotting/severity_trainnoise.py \
#     --mode_type train \
#     --attack_type identity \
#     --model_type alexnet \
#     --train_dataset cifar10 \
#     --eval_dataset cifar10C \
#     --loss_type adaptive \
#     --eval_noise gaussian_noise.npy \
#     --epochs 50 \
#     --valid_size 0.2 \
#     --eval_interval 1 \
#     --w_noise 0.1 \
#     --tau1 10

# Plot with adversarial test

python3 plotting/test_plot.py \
    --mode_type train \
    --attack_type identity \
    --model_type resnet18 \
    --train_dataset cifar10 \
    --eval_dataset cifar10C \
    --loss_type adaptive \
    --eval_noise gaussian_noise.npy \
    --epochs 100 \
    --valid_size 0.2 \
    --eval_interval 1 \
    --w_noise 0.1 \
    --tau1 10
