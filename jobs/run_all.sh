#!/bin/bash
cd ~/$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

prevJob_id=$(sbatch jobs/train.slurm UNetRegressionSE --num_epochs "1" --data_processors netcdf4 | awk '{print $4}')

while squeue -h -j "$prevJob_id" | grep -q .; do
  sleep 10
done

MODEL=$(tail -n 1 logs/latest_model_dir.txt)

echo "initial training loop has run, model: $MODEL"

for i in {1..2}
do
    prevJob_id=$(sbatch --dependency=afterok:$prevJob_id jobs/continue_train.slurm $MODEL 1 | awk '{print $4}')
done

echo "model trained"
sbatch --dependency=afterok:$prevJob_id jobs/continue_train.slurm $MODEL
echo "inference has been run"