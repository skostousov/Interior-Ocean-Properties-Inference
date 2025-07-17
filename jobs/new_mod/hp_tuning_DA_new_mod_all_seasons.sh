#!/bin/bash

mld_res=0.25
loss=MSE
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/jobs/new_mod/hp_tuning_DA_new_mod.slurm" "autumn" $mld_res $loss
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/jobs/new_mod/hp_tuning_DA_new_mod.slurm" "winter" $mld_res $loss
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/jobs/new_mod/hp_tuning_DA_new_mod.slurm" "spring" $mld_res $loss
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/jobs/new_mod/hp_tuning_DA_new_mod.slurm" "summer" $mld_res $loss