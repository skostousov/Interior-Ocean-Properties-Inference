#!/bin/bash
cd $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

prevJob_id=$(sbatch jobs/train_low_res.slurm DA_CNN 0 1e-4 32 autumn MSE --first_layer_filters 64 --kernel_size 3 | awk '{print $4}')

TARGET=$SCRATCH/logs/${prevJob_id}_model_dir.txt

echo "Waiting for $TARGET to appear..."
while [ ! -f "$TARGET" ]; do
  sleep 5
done

MODEL=$(tail -n 1 $TARGET)

echo "using, model: $MODEL"

prevJob_id=$(sbatch --dependency=afterok:$prevJob_id jobs/continue_train_low_res.slurm $MODEL 60 | awk '{print $4}')


echo "model trained"
sbatch --dependency=afterok:$prevJob_id jobs/eval_lower_res.slurm $MODEL True
echo "inference has been run"