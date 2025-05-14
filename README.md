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

### TODO List:
1. Fix train_no_val so that the model actually trains and converges to some solution
2. Create test script + visualization of results
3. Organize config file
4. Allow the model to train on several days at once, and figure out how to approach this
5. Make sure grids in regions dont overflow on to other regions
6. Review padding logic and hadnling of edge cases
7. Make training script functional with transforms
8. Allow for usage of splitter
9. Implement model saving and loading
10. Allow for cross-validation on training and figure out how to implement that, whether or not to retrain model, etc.
11. Fix loss and accuracy output metrics
