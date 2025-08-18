#!/bin/bash

#filepath="data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# filepath="data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"


# filepath=(
#     "data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc" 
# #"data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"
# #"data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# )

filepath=data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc
num_epochs=20
lat_lon=(True)
# seasons=("all" "autumn" "spring" "winter" "summer")
# seasons=("all" "autumn" "spring" "winter" "summer")
seasons=("autumn")
# seasons=("all")
groupbys=("months")
# models=("UNetRegressionSE" "downscaledUNetSE")
models=("DA_CNN")
# batch_size=(16 32 64)

#batch_size=(16)

batch_size=(128 256)

# batch_size=(64)
# models=("UNetRegressionSE" "DA_CNN")
# models=("UNetFull", GANGenerator)
# models=("GANGenerator")
# full=False
# mld_res=(1 0.5 0.333333333333333333333333333)
mld_res=(0.0833333333333333333333333333333333333333333333333333)
flfs=(8 16 32)
reduction=(3)
rim=(2 3 4)
loss=(MSE L1)
# dropout=(0.0)
dropout2=(0.4 0.6)
custom_features="so thetao uo vo zos mlotst"


for season in "${seasons[@]}"; do
  for groupby in "${groupbys[@]}"; do
    for model in "${models[@]}"; do
      for filepath in "${filepath[@]}"; do
        for la_lo in "${lat_lon[@]}"; do
          for m in "${mld_res[@]}"; do
            for b in "${batch_size[@]}"; do
              for r in "${rim[@]}"; do
                for flf in "${flfs[@]}"; do
                  for red in "${reduction[@]}"; do
                    for d in "${dropout2[@]}"; do
                      for l in "${loss[@]}"; do
                        sbatch jobs/combo_new_mod.slurm "$num_epochs" "$season" "$la_lo" "$groupby" "$model" "$filepath" "$b" "$m" "$flf" "$red" "$l" "$r" "$custom_features" "$d"
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
