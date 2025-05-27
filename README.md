# Interior-Ocean-Properties-Inference
## Utilizing Deep Learning Techniques to Infer Interior Ocean Properties from Satellite Imagery

### Setup
This repository's package dependencies are being handled using poetry.
To install dependencies and enter poetry environment, run
```bash
poetry install
poetry shell
```
Please update the config file with your own username in order to download data from https://data.marine.copernicus.eu/ (from which all data in this repository is sourced from)

## TODO List:

- ~~Fix train_no_val so that the model actually trains and converges to some solution~~ (15/05/2025)
- ~~Create test script + visualization of results~~ (16/05/2025)
- **Organize config file**
- ~~Allow the model to train on several days at once, and figure out how to approach this -possible solution: add several days to one region~~ (20/05/2025)
- ~~Make sure grids in regions dont overflow on to other regions~~ (15/05/2025)
- ~~Review padding logic and handling of edge cases~~ (No longer neccessary for present method)
- ~~Make training script functional with transforms~~ (15/05/2025)
- ~~Allow for usage of splitter~~ (16/05/2025)
- ~~Implement model saving and loading~~ (16/-5/2025)
- ~~Allow for cross-validation on training and figure out how to implement that, whether or not to retrain model, etc. (Resolved: Unneccessary and of little use)~~
- ~~Fix loss and accuracy output metrics~~ (15/05/2025)
- ~~Normalize inputs~~ (15/05/2025)
- ~~Implement Early Stopping~~
- ~~Automatically remove all regions with no ocean~~ (15/05/25)
- ~~Figure out difference in 1993 and 2021 dataset~~ (22/05/2025)
- ~~Make Unnormalizer functional~~ (20/05/2025)
- ~~Make test dataset return entire regions instead of mini grids used for training~~ (21/05/2025)
- ~~Normalize all days identically~~ (21/05/2025)
- ~~Create one pixel predictor~~ (22/05/2025)
- ~~Build DACNN~~ (23/05/2025)
- ~~Build SE UNET~~ (23/05/2025)
- ~~Create Evaluation script for one pixel predictor~~ (23/05/2025)
- Create framework for 12 month models and corresponding dataset downloading, etc.
- **For large model allow for a stratified split so each of 12 months makes an appearance in every split**

**BOLD** denotes tasks to be prioritized

## 15/05/2025

1. ~~Implement RobustScaler~~
2. ~~Zoom @ 11~~
3. ~~Continue with 1st, 12th,~~ and 13th task of todo list

#### Log:
- Implemented Robust Scaler
- Removed variables concerning ice from feature map: *vsi*, *usi*, *sithick*, *siconc*
- Modified data loader to remove features with no ocean
- Fixed grid generator to avoid regional overlap when partitioning image
- Fixed transforms
- Implemented model saving and early stopping

## 16/05/2025

1. ~~Implement train-test split within dataset.py~~
2. ~~Create test script + visualization of results~~

#### Log:
- implemented train-test split (including unique test subset class), although model fails to converge even remotely for the 1993 dataset (2021 is fine)!!!!
- MSE Loss is almost zero but mixed layer depth estimation is pretty dismal. MSE might be an unsuitable loss function. As well, the unnormalizer is currently non-functional.
- Upon further investigation, and after trying HuberLoss (penalizes rapid changes), it seems that the network is collapsing to a uniform classifier (upon further review, seems that mse is best option and not sure problem lies in it. might need more training data). Time to read some papers.
    - possible solution: use se attenion blocks to weight channels

## 20/05/2025

1. Deal with problem in 2021 vs 1993 dataset (EDIT: will save this for later)
2. ~~Tackle problem of days to expand training set~~
3. ~~make unnormalizer functional~~

#### Log:
- created new dataset class in dataset2.py which allows model to train on several days, thereby expanding training set.
- tried adding channel dropout to improve generalization but this fails miserably
- currently Unet achieves loss: 0.000045, Simple CNN loss: 0.000025 (almost HALF)
- spending most of the time debugging dataloader. trying to return day as well in order to unnormalize but doing so rather unsuccessfully
- FINALLY implemented (with some help) the capability to unnormalize during inference

## 21/05/2025

1. ~~Normalize all days identically, (perhaps also figure out better normalization method)~~
2. ~~Organize saved models~~ (more specific file names are now generated)
3. ~~Implement regional evaluation~~
4. If time: Organize config file

#### Log:

- All days now use the same scale, although unnormalization still produces unrealistic results.
- UPDATE: normalization is now fixed entirely. lots of my confusion was caused by the fact that the default normalization param was set to True.
- Run diff models again to get proper results with normalization functioning. Val Loss:
    - PixelWiseRegressor: 4.8e-05
    - Unet: 2.4e-05
- Implemented regional evaluation
- Leaving model to train overnight on large dataset (I postulate that small dataset may be main problem)

## 22/05/2025

1. ~~Figure out diff in 1993 2021 datasets~~
2. ~~Try larger regions~~

#### Log:
- Model trained had pretty poor loss (~0.3 when normalized) and regions i was able to test on had MLD predicted way out of scope
- Seems 1993 dataset has more variation in MLD which explains higher loss
- After increasing region size, loss on 1993 dataset: 0.4, although it seems that now the dataset is too small as only two regions in test set
- After reading paper on thermocline depth, implementin new dataloader which functions the same way
- struggling with implmenting normalization so that it is done on the train idx only and then saving mean and std to later apply on test 

## 23/05/2025

1. ~~Fix Normalization for monthly method.~~
2. ~~Create evaluation script.~~
3. *See Log*

#### Log:
- Created and ran evaluation script. It seems that one could only properly train the models on more data as there are seasonal patterns unnacounted for when training on just one year (as I am doing now)
- Ran training on basic CNN as a sanity check and alhtough it performs veryy poorly, seems to be learning
- Added SE blocks to every CONVRELU block in model and trained on 10 epochs. currenty evaluating
- Created DA-CNN from MLD bay of bengal paper and waiting to evaluate it.

## PLAN FOR WEEKEND:
- Train for 50 epochs ordinary UNET
- Evaluate ordinary UNET
- Train for 50 epochs UNET with SE
- Evaluate UNET with SE
- Train for 50 epochs DA-CNN
- Evaluate DA-CNN

## 26/05/2025

1. ~~Create evaluation proccess that minimizes RAM Usage~~
2. ~~Create nice plottings of MAE~~ and other evaluation metrics
3. Create framework for 12 month models and corresponding dataset downloading, etc.
4. ~~Also for normal large model allow for a stratified split so each of 12 months makes an appearance in every split~~

#### Log:
- Implemented Incremental Loading to minimize working memory usage

