#!/bin/bash
#SBATCH --job-name=combo_new_mod
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=compute_full_node
#SBATCH --cpus-per-task=8
#SBATCH --time=0-24:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

# num_epochs=60
# season=autumn
# lat_lon=True
# groupby=months
# model=UNetRegressionSE

num_epochs=$1
season=$2
lat_lon=$3
groupby=$4
model=$5
filepath=$6
full=$7
batch_size=$8



python -u $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/combo_new_mod.py --num_epochs $num_epochs --season $season --lat_lon $lat_lon --groupby $groupby --filepath $filepath --full $full --batch_size $batch_size $model

echo "combo_new_mod.slurm executed"