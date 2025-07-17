#!/bin/bash
#SBATCH --job-name=eval_gan
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

JOBID=$SLURM_JOB_ID

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

filepath=$1

python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/combo_gan_eval.py" --filepath $filepath