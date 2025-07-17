#!/bin/bash
$GAN_MODELS_DIR="/scratch/n/ngrisoua/kostouso/gan_models"
for subfolder in "$GAN_MODELS_DIR"/*/; do
    # Remove trailing slash
    subfolder_path="${subfolder%/}"
    # Example command using the relative path
    echo "Submitting job for subfolder: $subfolder_path"
    sbatch jobs/eval_gan.slurm "$subfolder_path"
done
