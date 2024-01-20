# remove model_type and loss_type 
# bc we eval for all model_type and 
# loss_type we have trained upon

# module load conda
# conda activate robust_DL
# module load pytorch/2.0.1

python3 plotting/model_losstype.py \
    --mode_type train \
    --attack_type identity \
    --train_dataset imagenet \
    --eval_dataset imagenetC \
    --train_noise blur \
    --eval_noise gaussian_noise.npy \
    --epochs 50 \
    --valid_size 0.2 \
    --eval_interval 1 \
    --severity 0.05 \
    --w_noise 0.1 \
    --tau1 10
