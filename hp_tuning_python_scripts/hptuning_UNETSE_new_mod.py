from utils.config import PROJECT_ROOT
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from models.downscaledUNetSE import UNetRegressionSE
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
    val_subset = TestSubsetRegressionNewMod(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    model = UNetRegressionSE(dataset[0][0].shape[0], base_filters=int(config['base_filters']), reduction=int(config['reduction']))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = config["criterion"]()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss = float('inf')
    for epoch in range(15):  # Use small number for tuning speed
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

    mld_res_space = {1/4:[1/12], 1/3:[1/12, 1/8], 1/2: [1/12, 1/8, 1/6], 1:[1/12, 1/8, 1/6, 1/4]}


    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([10, 32, 50, 100, 200]),
        "base_filters": tune.choice([8, 16, 32, 64]),
        "reduction": tune.choice([2, 4, 6, 8, 16]),
        "feature_res": tune.choice(mld_res_space[mld_res]),
        "criterion": loss_fn,
        "season": season,
        "mld_res": mld_res,
    }


    temp_dir = tempfile.mkdtemp(prefix=f"u_se{season}:{mld_res:.2f}")
    ray.init(ignore_reinit_error=True, _temp_dir=temp_dir)

    print("Trial started:")
    import sys
    sys.stdout.flush()

    reporter = CLIReporter(
    parameter_columns=["lr", "batch_size", "base_filters", "reduction", "feature_res"],
    metric_columns=["val_loss", "training_iteration", "total_time_s"]
    )

    root = Path(PROJECT_ROOT)

    # preset = {
    #     "lr": 5e-5,
    #     "batch_size": 50,
    #     "grid_size": 21,
    #     "base_filters": 32,
    #     "reduction": 8,
    # }

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
        storage_path=str(root / "ray_results" / f"U_SE_dyn_{season}_ls_{args.loss}_mres_{float(mld_res):.2f}"),
        verbose=True,
        progress_reporter=reporter,
        search_alg=OptunaSearch(
            metric="val_loss",
            mode="min",
            # points_to_evaluate=[preset]
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