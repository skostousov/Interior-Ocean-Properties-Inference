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
from data.argo.alternate_dataset import myDataset

print(os.getcwd())

season = "winter" #"summer" (4) 

temp_dir = tempfile.mkdtemp(prefix=f"ray_job_unetse_full_grace_{season}_")
ray.init(ignore_reinit_error=True, _temp_dir=temp_dir)

print("Trial started:")
import sys
sys.stdout.flush()

reporter = CLIReporter(
    parameter_columns=["lr", "batch_size", "base_filters", "reduction"],
    metric_columns=["val_loss", "training_iteration", "total_time_s"]
)
criterion = torch.nn.MSELoss()

root = Path(PROJECT_ROOT)

def train_model(config):
    # Prepare dataset
    dataset = myDataset(season=season)
    groups = dataset.groups

    all_indices = list(range(len(dataset)))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss_test.split(all_indices, groups=groups))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    model = UNetRegressionSE(dataset[0][0].shape[0], base_filters=int(config['base_filters']), reduction=int(config['reduction']))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_loss = float('inf')
    for epoch in range(20):  # Use small number for tuning speed
        model.train()
        for batch in train_loader:
            inputs, targets, _ = batch
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
                inputs, targets, _ = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets.unsqueeze(1)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        tune.report(val_loss=best_val_loss)

search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([10, 50, 100, 200]),
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
    max_t=20,
    grace_period=20,
    reduction_factor=2
)

tune.run(
    train_model,
    resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
    config=search_space,
    num_samples=70,
    scheduler=scheduler,
    storage_path=str(root / "ray_results" / f"full_grace_UNET_SE_low_res_{season}_loss_{criterion.__class__.__name__}"),
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