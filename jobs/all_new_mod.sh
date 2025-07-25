#!/bin/bash

#filepath="data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# filepath="data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"


# filepath=(
#     "data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc" 
# #"data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"
# #"data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# )

filepath=data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc
num_epochs=60
lat_lon=(True)
seasons=("all" "autumn" "spring" "winter" "summer")
groupbys=("months")
models=("DA_CNN")
batch_size=(16 32 64)
# models=("UNetRegressionSE" "DA_CNN")
# models=("UNetFull", GANGenerator)
# models=("GANGenerator")
full=False


for season in "${seasons[@]}"; do
  for groupby in "${groupbys[@]}"; do
    for model in "${models[@]}"; do
      for filepath in "${filepath[@]}"; do
        for la_lo in "${lat_lon[@]}"; do
        sbatch jobs/combo_new_mod.slurm $num_epochs "$season" "$la_lo" "$groupby" "$model" "$filepath" $full $batch_size
        done
      done
    done
  done
done