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


root = Path(PROJECT_ROOT)

filepath = 'data/daily_alternative_small/small_daily_alternative_sample_1993-1993.nc'
# filepath  = 'data/monthly/ten_sample_1993-2003.nc'

def train_model(config):
    # Prepare dataset
    dataset = TemporalDataset(filepath=root / filepath, grid_size=int(config["grid_size"]))
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

    for epoch in range(5):  # Use small number for tuning speed
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
        tune.report({'val_loss': val_loss})

search_space = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([10, 50, 100, 200, 500]),
    "grid_size": tune.choice([17, 21, 25]),
    "base_filters": tune.choice([8, 16, 32, 64]),
    "reduction": tune.choice([2, 4, 6, 8, 16]),
}

scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=5,
    grace_period=1,
    reduction_factor=2
)

tune.run(
    train_model,
    resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
    config=search_space,
    num_samples=20,
    scheduler=scheduler,
    storage_path=str(root / "ray_results" / "hptuning_UNET_SE"),
    search_alg=OptunaSearch(
        metric="val_loss",
        mode="min",
    ),
    name="hptuning_UNET_SE",
)