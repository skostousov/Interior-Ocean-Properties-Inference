#!/bin/bash
ROOT_DIR="/scratch/n/ngrisoua/kostouso/OceanPropInfSatImgScratch/OceanPropInfSatImg"
LOW_RES_MODELS_DIR="$ROOT_DIR/lower_res_models"

# for subfolder in "$LOW_RES_MODELS_DIR"/*/; do
#     # Remove trailing slash
#     subfolder_path="${subfolder%/}"
#     relative_path="${subfolder_path#$ROOT_DIR/}"

#     # Path to the training_info.txt file
#     txt_file="$subfolder_path/training_info.txt"

#     # Check if the txt file exists and does NOT contain "rmse"
#     if [[ -f "$txt_file" ]] && ! grep -q "r2" "$txt_file"; then
#         echo "Submitting evaluation job for subfolder: $relative_path"
#         sbatch jobs/eval_lower_res.slurm "$relative_path"
#         sleep 1
#     else
#         echo "Skipping $relative_path (contains 'r2' or training_info.txt missing)"
#     fi
# done

# Find all training_info.txt files up to 2 levels deep
find "$LOW_RES_MODELS_DIR" -mindepth 2 -maxdepth 4 -type f -name "training_info.txt" | while read -r txt_file; do
    # Get the folder containing training_info.txt
    subfolder_path="$(dirname "$txt_file")"
    relative_path="${subfolder_path#$ROOT_DIR/}"

    # Check if the file does NOT contain "rmse" AND test_indices.pt exists in the same folder
    if ! grep -q "r2" "$txt_file"; then
        echo "Submitting evaluation job for subfolder: $relative_path"
        sbatch jobs/eval_lower_res.slurm "$relative_path"
        sleep 1
    else
        echo "Skipping $relative_path (contains 'r2')"
    fi
done
