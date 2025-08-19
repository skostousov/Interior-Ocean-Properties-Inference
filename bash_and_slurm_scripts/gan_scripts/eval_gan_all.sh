#!/bin/bash
ROOT_DIR="/scratch/n/ngrisoua/kostouso/OceanPropInfSatImgScratch/OceanPropInfSatImg"
GAN_MODELS_DIR="/scratch/n/ngrisoua/kostouso/OceanPropInfSatImgScratch/OceanPropInfSatImg/gan_models"
for subfolder in "$GAN_MODELS_DIR"/*/; do
    # Remove trailing slash
    subfolder_path="${subfolder%/}"
    relative_path="${subfolder_path#$ROOT_DIR/}"
    # Example command using the relative path
    # echo "Submitting job for subfolder: $relative_path"
    sbatch bash_and_slurm_scripts/gan_scripts/eval_gan.slurm "$relative_path"
done
