# Interior-Ocean-Properties-Inference
## Utilizing Deep Learning Techniques to Infer Interior Ocean Properties from Satellite Imagery

This repository contains source code and models for the prediction of Mixed Layer Depth (MLD) from reanalysis data. As well, it contains scripts to replicate previous research in MLD prediction from interpolated Argo Data.

### Setup
This repository's package dependencies are being handled using poetry.
To install dependencies and enter poetry environment, run
```  bash
poetry install
poetry shell
```
Create an account to download data from https://data.marine.copernicus.eu/ (from which all data in this repository is sourced from)

### Dataset
After having downloaded the GLORYS Reanalysis data from https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description, the processing for training is handled by the most comprehensive of the pytorch datasets in *src/utils*, namely in *src/utils/datasettemporal_new_mod.py*. For more streamlined functionality and to ensure the data is located in the right directory, it is best to download the dataset via the script in *src/utils/download_ds.py*. The relevant configuration for dataset downloading can be set in the *configs/config.yaml* file.

### Training and Evaluation

The training and evaluation scripts for the above dataset are located in the project root, specifically *combo_new_mod.py* and *eval_new_mod.py*, where the former takes care of both training and evaluation, and the latter of evaluation only. NOTE: model and training parameters are passed to the training loop via the command line, processed via python's argparse module. Note that for *combo_new_mod.py*, the config file is used only to specify the dataset, and even then is not neccessary as this information can be passed via the commandline.
For example, when training the ResNet (note that some arguments were not specified and defaults were used instead):
``` python
python combo_new_mod.py --num_epochs 50 --season autumn --mld_res 0.083333333333333333333333333333333333333333333333333333333333 --filepath "data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc" --rim 2 --batch_size 128 --lr 0.00001 ResNetValue --base_channels 16 --n_blocks 5
```

Bash and slurm scripts to run training on the Niagara-Mist Scinet cluster are located in *bash_and_slurm_scripts*.

To compare various model performance, run src/utils/get_ranked_model_results.py

### Models & Hyperparameter Tuning

All models are located in *src/models*. Their trained products with weights are automatically saved in the relevant folder in the project root (e.g. new_mod_model_results) depending on the task.

Hyperparameter tuning scripts for specific models are located in the project root in the *hp_tuning_python_scripts* folder.

### Report & Other Information

The lower resolution argo interpolated data is managed through the low_res* files, but the datasets need to be downloaded manually. These dataset links are included in the attached MLD report.

A comprehensive report of methodology, procedure, and results can be accessed here:
https://docs.google.com/document/d/1cNGMUg1uLSHYB-cL7Wk8OSOa3vPrmSKIFzGW3OxckbU/edit?usp=sharing
