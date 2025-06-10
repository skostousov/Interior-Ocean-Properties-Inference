#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

python -u $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/begin_training.py --model $1 --num_epochs $2 --data_processors $3 \
&& echo "saved_models/$(ls -td -- $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/saved_models/*/ | head -1)" >> $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/logs/latest_model_dir.txt

echo "train.slurm executed"