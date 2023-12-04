full_noises=("saturate.npy" "spatter.npy" "gaussian_blue.npy" "speckle_noise.npy" "jpeg_compression.npy" "pixelate.npy" "elastic_transform.npy" "contrast.npy" "brightness.npy" "fog.npy" "frost.npy" "snow.npy" "zoom_blur.npy" "motion_blur.npy" "defocus.npy" "impulse_noise.npy" "shot_noise.npy" "gaussian_noise.npy")
test_noises=("shot_noise.npy" "gaussian_noise.npy")

if [[ "$1" == "test" ]]; then
    noises=("${test_noises[@]}")
elif [[ "$1" == "full" ]]; then
    noises=("${full_noises[@]}")

for noise in "${noises[@]}"
do
    bash experiment.sh "trades" "$noise" 0.5 50
done
