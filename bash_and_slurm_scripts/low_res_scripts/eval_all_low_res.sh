#!/bin/bash
ROOT_DIR="/scratch/n/ngrisoua/kostouso/OceanPropInfSatImgScratch/OceanPropInfSatImg"
LOW_RES_MODELS_DIR="$ROOT_DIR/lower_res_models"

for subfolder in "$LOW_RES_MODELS_DIR"/*/; do
    # Remove trailing slash
    subfolder_path="${subfolder%/}"
    relative_path="${subfolder_path#$ROOT_DIR/}"

    # Path to the training_info.txt file
    txt_file="$subfolder_path/training_info.txt"

    # Check if the txt file exists and does NOT contain "rmse"
    if [[ -f "$txt_file" ]] && ! grep -q "rmse" "$txt_file"; then
        echo "Submitting evaluation job for subfolder: $relative_path"
        sbatch jobs/eval_lower_res.slurm "$relative_path"
        sleep 2
    else
        echo "Skipping $relative_path (contains 'rmse' or training_info.txt missing)"
    fi
done