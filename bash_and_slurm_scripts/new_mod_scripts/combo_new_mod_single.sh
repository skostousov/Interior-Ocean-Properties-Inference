#!/bin/bash

num_epochs=50
season=Feb
lat_lon=True
groupby=months
model=DA_CNN
filepath=data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc
batch_size=32
mld_res=1
flf=8
kernel=3
loss=MSE
dropout=0.0
dropout2=0.2
rim=0

echo "--num_epochs $num_epochs --season $season --lat_lon $lat_lon --groupby $groupby --filepath $filepath --batch_size $batch_size --mld_res $mld_res --loss $loss --rim $rim $model --first_layer_filters $flf --kernel $kernel --dropout $dropout --dropout2 $dropout2"

python3 -u ~/SynologyDrive/OceanPropInfSatImg/combo_new_mod.py --num_epochs $num_epochs --season $season --lat_lon $lat_lon --groupby $groupby --filepath $filepath --batch_size $batch_size --mld_res $mld_res --loss $loss --rim $rim $model --first_layer_filters $flf --kernel $kernel --dropout $dropout --dropout2 $dropout2

echo "combo_new_mod.sh executed"