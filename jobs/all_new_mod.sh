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
# seasons=("all" "autumn" "spring" "winter" "summer")
seasons=("autumn" "spring" "winter" "summer")
# seasons=("all")

groupbys=("months")
models=("UNetRegressionSE")
# batch_size=(16 32 64)

#batch_size=(16)

batch_size=(32 64)

# batch_size=(64)
# models=("UNetRegressionSE" "DA_CNN")
# models=("UNetFull", GANGenerator)
# models=("GANGenerator")
# full=False
mld_res=(1 0.5 0.333333333333333333333333333)
flf=32
reduction=(8)
rim=(1 2 4)
loss=MSE
# dropout=(0.0)
# dropout2=(0.2 0.4)


for season in "${seasons[@]}"; do
  for groupby in "${groupbys[@]}"; do
    for model in "${models[@]}"; do
      for filepath in "${filepath[@]}"; do
        for la_lo in "${lat_lon[@]}"; do
          for m in "${mld_res[@]}"; do
            for b in "${batch_size[@]}"; do
              for r in "${rim[@]}"; do
                sbatch jobs/combo_new_mod.slurm "$num_epochs" "$season" "$la_lo" "$groupby" "$model" "$filepath" "$b" "$m" "$flf" "$k" "$loss" "$r"
              done
            done
          done
        done
      done
    done
  done
done
