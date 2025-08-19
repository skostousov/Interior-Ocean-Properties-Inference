#!/bin/bash

mld_res=0.25
loss=MSE
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/bash_and_slurm_scripts/new_mod_scripts/hp_tuning_UNETSE_new_mod.slurm" "autumn" $mld_res $loss
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/bash_and_slurm_scripts/new_mod_scripts/hp_tuning_UNETSE_new_mod.slurm" "winter" $mld_res $loss
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/bash_and_slurm_scripts/new_mod_scripts/hp_tuning_UNETSE_new_mod.slurm" "spring" $mld_res $loss
sbatch "$SCRATCH/OceanPropInfSatImgScratch/OceanPropInfSatImg/bash_and_slurm_scripts/new_mod_scripts/hp_tuning_UNETSE_new_mod.slurm" "summer" $mld_res $loss