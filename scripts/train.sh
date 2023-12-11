w_noises=(0.1 0.2 0.5 1.0)

for w_noise in "${w_noises[@]}"
do
    bash scripts/experiment.sh --w_noise $w_noise --mode_type train --loss_type adaptive --noise_type gaussian_blur.npy --epochs 50 
done
