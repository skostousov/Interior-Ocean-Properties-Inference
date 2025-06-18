#!/bin/bash
#SBATCH --job-name=hp_UNETSE_ray
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
# source activate torch39
source activate raytorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "hello"
export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src
echo "hi"
python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/hptuning_UNETSE_ray.py"

echo "hp_tuning executed"