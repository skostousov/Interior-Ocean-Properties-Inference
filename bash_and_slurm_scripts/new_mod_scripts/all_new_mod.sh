#!/bin/bash

#filepath="data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# filepath="data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"


# filepath=(
#     "data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc" 
# #"data/WaterOnlyDailySlightlyLarger/WaterOnlyDailyExtendedSeasonalitySlightlyLarger.nc"
# #"data/WaterOnlyDailyLarge/WaterOnlyDailyExtendedSeasonalityLarge.nc"
# )

filepath=data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc
num_epochs=50
lat_lon=(True)
# seasons=("all" "autumn" "spring" "winter" "summer")
# seasons=("all" "autumn" "spring" "winter" "summer")
seasons=("winter" "autumn")
# seasons=("all")
groupbys=("months")
# models=("UNetRegressionSE" "downscaledUNetSE")
models=("ResNetValue")
# batch_size=(16 32 64)

#batch_size=(16)

batch_size=(256)

# batch_size=(64)
# models=("UNetRegressionSE" "DA_CNN")
# models=("UNetFull", GANGenerator)
# models=("GANGenerator")
# full=False
mld_res=(1 0.5 0.333333333333333333333333333333333333)
# mld_res=(0.0833333333333333333333333333333333333333333333333333)
flfs=(16 32)
# final_activations=("relu" "sigmoid")
# norms=("in" "bn")
norms=("in")
rim=(2 3 4)
loss=(MSE)
# dropout=(0.0)
n_blocks=(5 6)
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
                  for n in "${n_blocks[@]}"; do
                    for l in "${loss[@]}"; do
                      for norm in "${norms[@]}"; do
                        sbatch bash_and_slurm_scripts/new_mod_scripts/combo_new_mod.slurm "$num_epochs" "$season" "$la_lo" "$groupby" "$model" "$filepath" "$b" "$m" "$flf" "$l" "$r" "$custom_features" "$n" "$norm"
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
