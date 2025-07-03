#!/bin/bash
cd $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

prevJob_id=$(sbatch jobs/train_low_res.slurm UNetRegressionSE 0 1.3301113370368737e-05 50 --base_filters 32 --reduction 8 | awk '{print $4}')

TARGET=$SCRATCH/logs/${prevJob_id}_model_dir.txt

echo "Waiting for $TARGET to appear..."
while [ ! -f "$TARGET" ]; do
  sleep 5
done

MODEL=$(tail -n 1 $TARGET)

echo "using, model: $MODEL"

for i in {1..5}
do
    prevJob_id=$(sbatch --dependency=afterok:$prevJob_id jobs/continue_train_low_res.slurm $MODEL 20 | awk '{print $4}')
done

echo "model trained"
sbatch --dependency=afterok:$prevJob_id jobs/eval_low_res.slurm $MODEL True
echo "inference has been run"