#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
source activate torch39

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

python $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/train_val_temporal_xarray.py

echo "model trained"
