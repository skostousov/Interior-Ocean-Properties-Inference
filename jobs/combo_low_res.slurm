#!/bin/bash
#SBATCH --job-name=combo_low_res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-1:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

JOBID=$SLURM_JOB_ID

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/combo_low_res.py" --loss MSE --num_epochs 60 --lr 1e-4 --batch_size 32 --season autumn DA_CNN --first_layer_filters 64 --kernel_size 3