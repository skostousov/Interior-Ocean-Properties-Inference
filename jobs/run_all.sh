#!/bin/bash
cd ~/$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

prevJob_id=$(sbatch jobs/train.slurm UNetRegressionSE --num_epochs "5" --data_processors netcdf4 | awk '{print $4}')
MODEL=
for i in {1..10}
do
    prevJob_id=$(sbatch --dependency=afterok:$prevJob_id jobs/continue_train.slurm $MODEL 5 | awk '{print $4}')
done
echo "model_trained"
sbatch --dependency=afterok:$prevJob_id jobs/continue_train.slurm $MODEL
echo "inference has been run"