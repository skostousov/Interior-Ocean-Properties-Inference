#!/bin/bash
#SBATCH --job-name=train_low_res
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --output=/scratch/n/ngrisoua/kostouso/logs/%x_%j.out

JOBID=$SLURM_JOB_ID

module load anaconda3
source activate torch39

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$PYTHONPATH:$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/src

model=$1
num_epochs=$2
lr=$3
batch_size=$4
season=$5
loss=$6

shift 6


python -u "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/begin_training_lower_res.py" --loss "$loss" --num_epochs "$num_epochs" --lr "$lr" --batch_size "$batch_size" --season "$season" "$model" "$@" \
&& model="lower_res_models/$(ls -td -- $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/lower_res_models/* | head -1)" \
&& model=${model#*$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/} \
&& model=${model%/} \
&& echo "${model}" > "$SCRATCH/logs/${JOBID}_model_dir.txt"

echo "train.slurm executed"