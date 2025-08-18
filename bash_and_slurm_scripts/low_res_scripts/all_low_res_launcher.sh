#!/bin/bash

losses=("MSE" "L1")
num_epochs=70
coarsens=(1)
lr=0.0001
batch_sizes=("32" "64" "16")
seasons=("all" "autumn" "spring" "winter" "summer")
base_filters=32
reduction=8

echo "hello"

for season in "${seasons[@]}"; do
  for coarsen in "${coarsens[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
      for loss in "${losses[@]}"; do
        sbatch jobs/combo_low_res.slurm $loss $num_epochs $coarsen $lr $batch_size $season $base_filters $reduction
      done
    done
  done
done