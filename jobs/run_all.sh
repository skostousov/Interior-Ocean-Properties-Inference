#!/bin/bash
cd $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

prevJob_id=$(sbatch jobs/train.slurm PixelWiseRegressor 0 netcdf4 | awk '{print $4}')

TARGET=$SCRATCH/logs/${prevJob_id}_model_dir.txt

echo "Waiting for $TARGET to appear..."
while [ ! -f "$TARGET" ]; do
  sleep 5
done

MODEL=$(tail -n 1 $TARGET)

echo "using, model: $MODEL"

for i in {1..10}
do
    prevJob_id=$(sbatch --dependency=afterok:$prevJob_id jobs/continue_train.slurm $MODEL 10 | awk '{print $4}')
done

echo "model trained"
sbatch --dependency=afterok:$prevJob_id jobs/eval.slurm $MODEL
echo "inference has been run"