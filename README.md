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
- **Create test script + visualization of results**
- Organize config file
- Allow the model to train on several days at once, and figure out how to approach this
- ~~Make sure grids in regions dont overflow on to other regions~~ (15/05/2025)
- Review padding logic and handling of edge cases
- ~~Make training script functional with transforms~~ (15/05/2025)
- Allow for usage of splitter
- Implement model saving and loading
- Allow for cross-validation on training and figure out how to implement that, whether or not to retrain model, etc.
- ~~Fix loss and accuracy output metrics~~ (15/05/2025)
- ~~Normalize inputs~~ (15/05/2025)
- ~~Implement Early Stopping~~
- ~~Automatically remove all regions with no ocean~~ (15/05/25)

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

1. Implement train-test split within dataset.py
2. Create test script + visualization of results

#### Log:
- 