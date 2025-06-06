#!/bin/bash
#SBATCH --job-name=netcdf4_install
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

module load anaconda3
source activate torch39

conda install -y netcdf4

echo "netCDF4 done in $CONDA_PREFIX"
