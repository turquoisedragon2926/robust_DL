`~/sshproxy.sh -u richardr`
`ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov`

1. `ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /global/homes/r/richardr/code/dfno/plots/ && tar cz DFNO_3D" | tar zxv -C /Users/richardr2926/Desktop/Research/Code/dfno/plots/`
2. `ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /global/homes/r/richardr/code/dfno/examples/scaling/ && tar cz results" | tar zxv -C /Users/richardr2926/Desktop/Research/Code/dfno/examples/scaling/`

# Move weight file from local to remote
`scp weights/CIFARC10_Alexnet_TRADES_LOSS_BETA=0.5.pt richardr@perlmutter.nersc.gov:/global/homes/r/richardr/deep/robust_DL/weights/ `

# Move all results plots weights optimizers to local
`ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /global/homes/r/richardr/deep/robust_DL/results && tar cz metrics models optimizers plots logs" | tar zxv -C /Users/richardr2926/Desktop/MSCS_Data/CS_7643/robust_DL/results`

# Allocate to run tests
`salloc --nodes=1 --constraint=gpu --gpus=1 --qos=interactive --time=00:10:00 --account=m3863_g --ntasks=1 --gpus-per-task=1`

# Evaluate trades model on given noise type
`python3 experiment.py --mode_type eval --loss_type trades --noise_type gaussian_noise.npy --alpha 2 --model_checkpoint "CIFARC10_Alexnet_TRADES_LOSS_BETA=0.5_EPOCHS=50.pt"`

# Uhhh

`srun --export=ALL python3 experiment.py --mode_type eval --loss_type trades --noise_type gaussian_noise.npy --alpha 2 --severity $severity --w_noise $w_noise --tau1 $tau1 --tau2 $tau2 --epochs $epochs --model_checkpoint $model_checkpoint`

# Train custom loss

`python3 experiment.py --mode_type train --loss_type custom --noise_type gaussian_blur.npy --w_noise 0.1 --severity 0.05 --tau1 10 --tau2 -10 --epochs 50`
```
mv ~/.conda/envs/sysml/* $SCRATCH/sysml/
rmdir ~/.conda/envs/sysml

ln -s $SCRATCH/sysml ~/.conda/envs/sysml
```

```
export HF_HOME=$SCRATCH
export HF_DATASETS_CACHE=$SCRATCH
export TRANSFORMERS_CACHE=$SCRATCH

cd $SCRATCH/smr/fl-minillm

module load conda
conda activate sysml
srun bash fl/fl_train.sh 4

torchrun evaluate.py --json-data --model-path results/gpt2/train/fl-minillm/0/0
```