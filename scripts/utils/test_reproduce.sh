# models=("resnet18")
# losses=("adaptive")

# # module load conda
# # conda activate robust_DL
# # module load pytorch/2.0.1

# for loss in "${losses[@]}"; do
#     for model in "${models[@]}"; do
#         python3 main.py \
#             --mode_type train \
#             --attack_type identity \
#             --model_type $model \
#             --train_dataset cifar10 \
#             --eval_dataset cifar10C \
#             --loss_type $loss \
#             --train_noise blur \
#             --eval_noise gaussian_noise.npy \
#             --epochs 1 \
#             --valid_size 0.8 \
#             --eval_interval 1 \
#             --severity 0.05 \
#             --w_noise 0.1 \
#             --tau1 10
#     done
# done

severities=(0.05)
train_noises=("uniform" "blur" "gaussian" "uniform" "shot" "dynamicBlur" "random")

for train_noise in "${train_noises[@]}"; do
    for severity in "${severities[@]}"; do
        bash scripts/experiment.sh \
            --mode_type train \
            --attack_type identity \
            --train_dataset cifar10 \
            --eval_dataset cifar10C \
            --model_type resnet18 \
            --loss_type adaptive \
            --train_noise $train_noise \
            --eval_noise gaussian_noise.npy \
            --epochs 1 \
            --valid_size 0.5 \
            --eval_interval 1 \
            --severity $severity \
            --w_noise 0.1 \
            --tau1 10
    done
done
