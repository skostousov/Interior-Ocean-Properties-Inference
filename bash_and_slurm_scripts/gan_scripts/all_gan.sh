#!/bin/bash

filepath="data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# filepath="data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"
# filepath="data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc"

lat_lon=False

# sbatch jobs/combo_gan.slurm years all $filepath $lat_lon
# sbatch jobs/combo_gan.slurm years autumn $filepath $lat_lon
# sbatch jobs/combo_gan.slurm years summer $filepath $lat_lon
# sbatch jobs/combo_gan.slurm years spring $filepath $lat_lon
# sbatch jobs/combo_gan.slurm years winter $filepath $lat_lon

sbatch jobs/combo_gan.slurm months all $filepath $lat_lon
sbatch jobs/combo_gan.slurm months autumn $filepath $lat_lon
sbatch jobs/combo_gan.slurm months summer $filepath $lat_lon
sbatch jobs/combo_gan.slurm months spring $filepath $lat_lon
sbatch jobs/combo_gan.slurm months winter $filepath $lat_lon