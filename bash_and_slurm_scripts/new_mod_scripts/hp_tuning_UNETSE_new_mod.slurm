#!/bin/bash
#SBATCH --job-name=hp_new_mod_UNETSE
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
# source activate torch39
source activate raytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/hp_tuning_python_scripts/hptuning_UNETSE_new_mod.py" --season $1 --mld_res $2 --loss $3

echo "hp_tuning executed"