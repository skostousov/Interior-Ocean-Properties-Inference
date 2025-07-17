#!/bin/bash

# filepath="data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# filepath="data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"
filepath = "data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc"

# sbatch jobs/combo_gan.slurm years all $filepath
# sbatch jobs/combo_gan.slurm years autumn $filepath
# sbatch jobs/combo_gan.slurm years summer $filepath
# sbatch jobs/combo_gan.slurm years spring $filepath
# sbatch jobs/combo_gan.slurm years winter $filepath

sbatch jobs/combo_gan.slurm months all $filepath
sbatch jobs/combo_gan.slurm months autumn $filepath
sbatch jobs/combo_gan.slurm months summer $filepath
sbatch jobs/combo_gan.slurm months spring $filepath
sbatch jobs/combo_gan.slurm months winter $filepath