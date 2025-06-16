#!/bin/bash
cd $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

MODEL=$1

echo "using, model: $MODEL"

prevJob_id=$(sbatch jobs/continue_train.slurm $MODEL 10 | awk '{print $4}')

for i in {1..10}
do
    prevJob_id=$(sbatch --dependency=afterok:$prevJob_id jobs/continue_train.slurm $MODEL 10 | awk '{print $4}')
done

echo "model trained"
sbatch --dependency=afterok:$prevJob_id jobs/eval.slurm $MODEL True
echo "inference has been run"