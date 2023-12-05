full_noises=("saturate.npy" "spatter.npy" "gaussian_blue.npy" "speckle_noise.npy" "jpeg_compression.npy" "pixelate.npy" "elastic_transform.npy" "contrast.npy" "brightness.npy" "fog.npy" "frost.npy" "snow.npy" "zoom_blur.npy" "motion_blur.npy" "defocus.npy" "impulse_noise.npy" "shot_noise.npy" "gaussian_noise.npy")
test_noises=("spatter.npy" "brightness.npy")

if [[ "$1" == "test" ]]; then
    noises=("${test_noises[@]}")
elif [[ "$1" == "full" ]]; then
    noises=("${full_noises[@]}")
fi

for noise in "${noises[@]}"
do
    bash experiment.sh --modetype eval --losstype trades --noisetype $noise --alpha 2 --model_checkpoint "CIFARC10_Alexnet_TRADES_LOSS_BETA=0.5_EPOCHS=50.pt"
    bash experiment.sh --modetype eval --losstype ce --noisetype $noise --model_checkpoint "CIFARC10_Alexnet_CE_LOSS_EPOCHS=50.pt"
done
