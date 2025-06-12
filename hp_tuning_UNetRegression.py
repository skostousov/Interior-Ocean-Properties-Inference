from models.UNET_regression import UNetRegression
from utils.datasettemporal import TemporalDataset
from utils.config import PROJECT_ROOT
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit, ParameterSampler
from scipy.stats import loguniform
import pprint  # just for pretty printing at the end

root = Path(PROJECT_ROOT)


def train_model(config):
    # Prepare dataset
    dataset = TemporalDataset(
        filepath=root / 'data/daily_alternative_small/small_daily_alternative_sample_1993-1993.nc',
        grid_size=int(config["grid_size"])
    )
    groups = dataset.groups

    all_indices = list(range(len(dataset)))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss_test.split(all_indices, groups=groups))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=False
    )

    model = UNetRegression(6, grid_size=int(config["grid_size"]), first_out=int(config['first_out']))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    size_train = len(train_loader)
    size_val = len(val_loader)

    for epoch in range(5):  # keep small for speed
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            if i%2000==0:
                print(f"train {i}/{size_train}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += criterion(model(inputs), targets).item()
                if i%2000==0:
                    print(f"val {i}/{size_val}")
        val_loss /= len(val_loader)

    return val_loss


# ---------------------------------------------------------------------------
# scikit-learn random search (ParameterSampler)  ----------------------------
# ---------------------------------------------------------------------------

param_distributions = {
    "lr": loguniform(1e-5, 1e-2),
    "batch_size": [10, 50, 100],
    "grid_size": [17, 21, 25],
    "first_out": [8, 16, 32, 64]
}




num_samples = 10
sampler = ParameterSampler(param_distributions,
                           n_iter=num_samples,
                           random_state=0)

best_loss = float("inf")
best_config = None

for i, config in enumerate(sampler, start=1):
    val_loss = train_model(config)
    print(f"Trial {i:02d}/{num_samples} | val_loss = {val_loss:.6f} | "
          f"config = {config}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_config = config

    print("\nBest configuration found so far: ")
    pprint.pp(best_config)
    print(f"Validation loss: {best_loss:.6f}")

print("\nBest configuration found")
pprint.pp(best_config)
print(f"Validation loss: {best_loss:.6f}")
