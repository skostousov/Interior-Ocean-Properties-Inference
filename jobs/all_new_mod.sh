#!/bin/bash

#filepath="data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# filepath="data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"


# filepath=(
#     "data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc" 
# #"data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"
# #"data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# )

filepath=data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc
num_epochs=70
lat_lon=(True)
# seasons=("all" "autumn" "spring" "winter" "summer")
seasons=("autumn" "spring" "winter" "summer")

groupbys=("months")
models=("DA_CNN")
batch_size=(16 32 64)
# batch_size=(32)
# models=("UNetRegressionSE" "DA_CNN")
# models=("UNetFull", GANGenerator)
# models=("GANGenerator")
# full=False
mld_res=(1 0.5 0.333333333333333333333333333 0.25)
flf=8
kernel=(1 3)
loss=MSE


for season in "${seasons[@]}"; do
  for groupby in "${groupbys[@]}"; do
    for model in "${models[@]}"; do
      for filepath in "${filepath[@]}"; do
        for la_lo in "${lat_lon[@]}"; do
          for k in "${kernel[@]}"; do
            for m in "${mld_res[@]}"; do
              for b in "${batch_size[@]}"; do
                sbatch jobs/combo_new_mod.slurm "$num_epochs" "$season" "$la_lo" "$groupby" "$model" "$filepath" "$b" "$m" "$flf" "$k" "$loss"
              done
            done
          done
        done
      done
    done
  done
done