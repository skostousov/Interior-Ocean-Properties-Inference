#!/bin/bash
cd $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

prevJob_id=$(sbatch bash_and_slurm_scripts/reanalysis_original_scripts/train.slurm UNetRegressionSE 0 netcdf4 21 0.0027 50 summer --base_filters 64 --reduction 8 | awk '{print $4}')

TARGET=$SCRATCH/logs/${prevJob_id}_model_dir.txt

echo "Waiting for $TARGET to appear..."
while [ ! -f "$TARGET" ]; do
  sleep 5
done

MODEL=$(tail -n 1 $TARGET)

echo "using, model: $MODEL"

for i in {1..10}
do
    prevJob_id=$(sbatch --dependency=afterok:$prevJob_id bash_and_slurm_scripts/reanalysis_original_scripts/continue_train.slurm $MODEL 7 | awk '{print $4}')
done

echo "model trained"
sbatch --dependency=afterok:$prevJob_id bash_and_slurm_scripts/reanalysis_original_scripts/eval.slurm $MODEL True
echo "inference has been run"