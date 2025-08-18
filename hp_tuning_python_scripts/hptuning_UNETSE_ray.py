from utils.datasettemporal import TemporalDataset
from utils.config import PROJECT_ROOT
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from models.UNET_regressionSE import UNetRegressionSE
from sklearn.model_selection import GroupShuffleSplit
from ray.tune import CLIReporter
import ray
import tempfile
import os

print(os.getcwd())

season = "winter"

temp_dir = tempfile.mkdtemp(prefix=f"ray_job_unetse_{season}_")
ray.init(ignore_reinit_error=True, _temp_dir=temp_dir)



print("Trial started:")
import sys
sys.stdout.flush()

reporter = CLIReporter(
    parameter_columns=["lr", "batch_size", "grid_size", "base_filters", "reduction"],
    metric_columns=["val_loss", "training_iteration", "total_time_s"]
)


root = Path(PROJECT_ROOT)

# filepath = 'data/daily_alternative_small/small_daily_alternative_sample_1993-1993.nc'
# filepath  = 'data/monthly/ten_sample_1993-2003.nc'
# filepath = 'data/BoBDaily/BoBDaily_1993-1993.nc'
# filepath= 'data/BoBMonthly/BoBMonthly_1993-2003.nc'
# filepath = "data/WaterOnlyDaily/WaterOnlyDaily_1993-1993.nc"
# filepath = "data/WaterOnlyMonthly/WaterOnlyMonthly_1993-2003.nc"
filepath = "data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc"

def train_model(config):
    # Prepare dataset
    dataset = TemporalDataset(filepath=root / filepath, grid_size=int(config["grid_size"]), season=season)
    groups = dataset.groups

    all_indices = list(range(len(dataset)))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss_test.split(all_indices, groups=groups))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=int(config["batch_size"]), shuffle=False)

    model = UNetRegressionSE(6, grid_size=int(config["grid_size"]), base_filters=int(config['base_filters']), reduction=int(config['reduction']))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(7):  # Use small number for tuning speed
        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        tune.report(val_loss=val_loss)

search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([10, 50, 100, 200, 500]),
    "grid_size": tune.choice([17, 21, 25]),
    "base_filters": tune.choice([8, 16, 32, 64]),
    "reduction": tune.choice([2, 4, 6, 8, 16]),
}

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
    max_t=7,
    grace_period=2,
    reduction_factor=2
)

tune.run(
    train_model,
    resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
    config=search_space,
    num_samples=10,
    scheduler=scheduler,
    storage_path=str(root / "ray_results" / f"hptuning_UNET_SE_{filepath.split('/')[-1].split('.')[0]}+{season}"),
    verbose=True,
    progress_reporter=reporter,
    search_alg=OptunaSearch(
        metric="val_loss",
        mode="min",
        # points_to_evaluate=[preset]
    ),
    log_to_file=True,
    resume="AUTO+RESTART_ERRORED”"
)