from models.DA_CNN import DA_CNN
from utils.config import PROJECT_ROOT
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import BasicVariantGenerator
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import GroupShuffleSplit
from ray.tune import CLIReporter
import ray
import tempfile
import os
import torch.nn as nn
from utils.datasettemporal_new_mod import TemporalDatasetNewMod, TestSubsetRegressionNewMod

root = Path(PROJECT_ROOT)

filepath = root/"data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc"

def train_model(config):
    # Prepare dataset
    dataset = TemporalDatasetNewMod(season=config["season"], mld_res=config["mld_res"], feature_res=float(config["feature_res"]), filepath=filepath)
    groups = dataset.groups

    all_indices = list(range(len(dataset)))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss_test.split(all_indices, groups=groups))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    model = DA_CNN(dataset[0][0].shape[0], first_layer_filters=int(config["first_layer_filters"]), kernel_size=int(config['kernel_size']))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # criterion = torch.nn.L1Loss()
    criterion = config["criterion"]()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss = float('inf')
    for epoch in range(15):
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets.unsqueeze(1)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        tune.report(val_loss=best_val_loss)

def main(args):
    print(os.getcwd())
    loss_dict = {'L1' : nn.L1Loss, 'MSE' : nn.MSELoss}
    season = str(args.season)
    mld_res = float(args.mld_res)
    loss_fn = loss_dict[args.loss]

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([10, 32, 50, 100, 200]),
        "first_layer_filters": tune.choice([8, 16, 32, 64]),
        "kernel_size": 3,
        "feature_res": tune.choice([1/12, 1/8, 1/6, 1/4]),
        "criterion": loss_fn,
        "season": season,
        "mld_res": mld_res,
    }
    temp_dir = tempfile.mkdtemp(prefix=f"da_dynamic_{season}_mres:{mld_res}")
    ray.init(ignore_reinit_error=True, _temp_dir=temp_dir)

    print("Trial started:")
    import sys
    sys.stdout.flush()

    reporter = CLIReporter(
    parameter_columns=["lr", "batch_size", "first_layer_filters", "feature_res", "kernel_size"],
    metric_columns=["val_loss", "training_iteration", "total_time_s"]
    ) 

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=15,
        grace_period=15,
        reduction_factor=2
    )

    tune.run(
        train_model,
        resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        config=search_space,
        num_samples=30,
        scheduler=scheduler,
        storage_path=str(root / "ray_results" / f"DA_dynamic_{season}_loss_{args.loss}_mres_{mld_res}"),
        verbose=True,
        progress_reporter=reporter,
        search_alg=OptunaSearch(
            metric="val_loss",
            mode="min",
            #dsd
        ),
        log_to_file=True,
        resume="AUTO"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model on xarray data")
    parser.add_argument('--season', default='all', type=str)
    parser.add_argument('--mld_res', default=1, type=float, help='Resolution of MLD data')
    parser.add_argument('--loss', default='MSE', type=str, choices=['MSE', 'L1'], help='Loss function to use for training')
    args = parser.parse_args()
    main(args)