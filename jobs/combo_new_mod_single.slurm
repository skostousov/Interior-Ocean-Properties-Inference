#!/bin/bash
#SBATCH --job-name=single_new_mod
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-08:00:00
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

num_epochs=60
season=autumn
lat_lon=True
groupby=months
model=DA_CNN
filepath=data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc
batch_size=16
mld_res=1
flf=8
kernel=3
loss=MSE

echo "--num_epochs $num_epochs --season $season --lat_lon $lat_lon --groupby $groupby --filepath $filepath --batch_size $batch_size --mld_res $mld_res --loss $loss $model --first_layer_filters $flf --kernel_size $kernel"

python -u $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/combo_new_mod.py --num_epochs $num_epochs --season $season --lat_lon $lat_lon --groupby $groupby --filepath $filepath --batch_size $batch_size --mld_res $mld_res --loss $loss $model --first_layer_filters $flf --kernel_size $kernel

echo "combo_new_mod.slurm executed"