#!/bin/bash
#SBATCH --job-name=hp_downsample_EBAM_CNN
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/hp_tuning_python_scripts/hptuning_EBAM_CNN_downsample.py"

echo "hp_tuning executed"