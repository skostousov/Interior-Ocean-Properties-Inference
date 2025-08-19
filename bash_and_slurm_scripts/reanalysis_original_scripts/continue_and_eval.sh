#!/bin/bash
cd $SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg

MODEL=$1

echo "using, model: $MODEL"

prevJob_id=$(sbatch bash_and_slurm_scripts/reanalysis_original_scripts/continue_train.slurm $MODEL 10 | awk '{print $4}')

for i in {1..10}
do
    prevJob_id=$(sbatch --dependency=afterok:$prevJob_id bash_and_slurm_scripts/reanalysis_original_scripts/continue_train.slurm $MODEL 5 | awk '{print $4}')
done

echo "model trained"
sbatch --dependency=afterok:$prevJob_id bash_and_slurm_scripts/reanalysis_original_scripts/eval.slurm $MODEL True
echo "inference has been run"