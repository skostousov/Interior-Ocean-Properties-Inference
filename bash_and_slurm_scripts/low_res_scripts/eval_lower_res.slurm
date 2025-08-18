#!/bin/bash
#SBATCH --job-name=eval_low_res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-1:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

python -u $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/eval_lower_res.py --model_relative_path $1 --recalculate True

echo "eval_lower_res.slurm executed"