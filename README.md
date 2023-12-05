`~/sshproxy.sh -u richardr`
`ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov`

1. `ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /global/homes/r/richardr/code/dfno/plots/ && tar cz DFNO_3D" | tar zxv -C /Users/richardr2926/Desktop/Research/Code/dfno/plots/`
2. `ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /global/homes/r/richardr/code/dfno/examples/scaling/ && tar cz results" | tar zxv -C /Users/richardr2926/Desktop/Research/Code/dfno/examples/scaling/`

`salloc --nodes=1 --constraint=gpu --gpus=4 --qos=interactive --time=00:30:00 --account=m3863_g --ntasks=4 --gpus-per-task=1`

`ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /global/homes/r/richardr/deep/robust_DL/ && tar cz plots" | tar zxv -C /Users/richardr2926/Desktop/MSCS_Data/CS_7643/robust_DL/`

`scp weights/CIFARC10_Alexnet_TRADES_LOSS_BETA=0.5.pt richardr@perlmutter.nersc.gov:/global/homes/r/richardr/deep/robust_DL/weights/ `

`ssh -l richardr -i ~/.ssh/nersc perlmutter.nersc.gov "cd /global/homes/r/richardr/deep/robust_DL/ && tar cz results" | tar zxv -C /Users/richardr2926/Desktop/MSCS_Data/CS_7643/robust_DL/`

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