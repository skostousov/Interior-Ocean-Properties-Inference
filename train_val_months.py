import torch
from torch import nn
from pathlib import Path
import os
from models.UNET_regression import UNetRegression
from models.UNET_regressionSE import UNetRegressionSE
from models.simple_CNN_regression import PixelWiseRegressor
from torch.utils.data import Subset, DataLoader
from utils.datasetmonths import DatasetOverMonths
# from utils.transforms import RescaledRotationTransform, ToTensor, Compose
from torchvision.transforms import Normalize, Compose, ToTensor
from utils.config import DATA_DIR, RAW_CONFIG, SAVED_MODELS_DIR, EARLY_STOP
from torch.optim import AdamW
import time
from utils.splitter import train_val_test_split_months
from torch.optim.lr_scheduler import ReduceLROnPlateau

cfg = RAW_CONFIG
mode = cfg['monthly']['training']['mode']

def train_loop(model, train_dataloader, optimizer, loss_fn, device):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * batch_size + len(images)
            print(f"loss: {loss:>7f} [{current:>6d}/{size:>5d}]")

def val_loop(val_dataloader, model, loss_fn, device, scheduler):
    model.eval()
    num_batches = len(val_dataloader)
    val_loss = 0
    with torch.no_grad():
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            val_loss += loss_fn(pred, label).item()

    val_loss /= num_batches
    print(f"Val Loss: {val_loss:>8f} \n")
    scheduler.step(val_loss)
    return val_loss

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# data_aug = Compose([ToTensor(), RescaledRotationTransform()])
data = DatasetOverMonths()

batch_size = RAW_CONFIG['monthly']['training']["batch_size"]
epochs = RAW_CONFIG['monthly']['training']["epochs"]

# model = UNetRegression(data[0][0].shape[0], data[0][1].shape[0])
model = UNetRegressionSE(data[0][0].shape[0], data[0][1].shape[0])
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

loss_fn = nn.L1Loss()


train_idx, val_idx, _ = train_val_test_split_months(data, seed=42, test_indices_path=Path(cfg['monthly'][mode]["test_indices"]))

# mu, std = data.generate_mean_and_std_partial(train_idx)
# mu_label, std_label = data.generate_mean_and_std_labels(train_idx)

# print(f"mu_label: {mu_label}, std_label: {std_label}")
# print(f"mu: {mu}, std: {std}")
# stats_path = Path(cfg['monthly'][cfg['monthly']['training']['mode']]["stats"])
# if not os.path.exists(stats_path):
#     os.makedirs(stats_path.parent, exist_ok=True)
# torch.save({"mu": mu, "std": std, "mu_label": mu_label, "std_label": std_label}, stats_path)

# stats = torch.load(stats_path, weights_only=False)
# mu, std = stats["mu"], stats["std"]
# mu_label, std_label = stats["mu_label"], stats["std_label"]

# data.transform = Normalize(mu, std)
# data.target_transform = Normalize(mu_label, std_label)

train_data, val_data, = Subset(data, train_idx), Subset(data, val_idx)

# train_data, val_data = simple_train_val_split(data)

print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")
print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Val dataset shape: img: {val_data[0][0].shape}, lbl: {val_data[0][1].shape}")

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

best_loss = 1000000000000000000

def checkpoint(model, filename):
     torch.save(model.state_dict(), filename)

start_timestamp = time.strftime('%Y%m%d_%H%M%S')
best_epoch=0
for epoch in range(0, epochs):
        print('Epoch {}:'.format(epoch + 1))
        model.train(True)
        train_loop(model, train_dataloader, optimizer, loss_fn, device)
        val_loss = val_loop(val_dataloader, model, loss_fn, device, scheduler)
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            save_dir = SAVED_MODELS_DIR / f'model: {model.__repr__()} training_start_time: {start_timestamp} datafile: {cfg["monthly"][mode]["output_file"]} strat: {cfg["monthly"][mode]["test_indices"].split("s/")[-1]}'
            os.makedirs(save_dir, exist_ok=True)
            model_state_path = save_dir / 'best_model_state'
            model_path = save_dir / 'best_model'
            torch.save(model.state_dict(), model_state_path)
            torch.save(model, model_path)
        if epoch - best_epoch >= EARLY_STOP:
             print(f"Early stopping on epoch {epoch}")
             break
print(f"Best loss: {best_loss}")


