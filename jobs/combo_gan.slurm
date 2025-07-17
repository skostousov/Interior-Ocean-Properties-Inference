#!/bin/bash
#SBATCH --job-name=combo_gan
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=compute_full_node
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

JOBID=$SLURM_JOB_ID

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

groupby=$1
season=$2
filepath=$3

python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/combo_gan.py" --groupby $groupby --season $season --filepath $filepath