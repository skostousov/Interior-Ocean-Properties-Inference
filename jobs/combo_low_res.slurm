#!/bin/bash
#SBATCH --job-name=combo_low_res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-7:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

JOBID=$SLURM_JOB_ID

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src


loss=MSE
num_epochs=70
coarsen=2
lr=0.0001
batch_size=32
season=all

loss=$1
num_epochs=$2
coarsen=$3
lr=$4
batch_size=$5
season=$6
base_filters=$7
reduction=$8

python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/combo_low_res.py" --loss $loss --num_epochs $num_epochs --coarsen $coarsen --lr $lr --batch_size $batch_size --season $season UNetRegressionSE --base_filters $base_filters --reduction $reduction