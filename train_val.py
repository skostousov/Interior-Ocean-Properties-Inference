import torch
from torch import nn
from pathlib import Path
import os
from models.Unet import UNet
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from utils.dataset import GLORYSDS
from utils.dataset2 import GLORYSDS2
from utils.transforms import RescaledRotationTransform, ToTensor, Compose
from utils.config import DATA_DIR, RAW_CONFIG, SAVED_MODELS_DIR, EARLY_STOP
from torch.optim import Adam
import time
from utils.splitter import simple_train_val_split, train_val_test_split
from models.simple_CNN import PixelWiseRegressor
from models.CNN_EBAM import CNN_EBAM

cfg = RAW_CONFIG

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

def val_loop(val_dataloader, model, loss_fn, device):
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
    return val_loss

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_aug = Compose([ToTensor(), RescaledRotationTransform()])
# ds_path = DATA_DIR/"lat:30-60_long:-190--120_date:1993-10-11-1993-10-12.nc"
ds_path = DATA_DIR/cfg["datafile"]
# data = GLORYSDS(ds_path, data_aug, days = False, normalize=True)
data = GLORYSDS2(ds_path, data_aug, normalize=True)

batch_size = RAW_CONFIG["batch_size"]
epochs = RAW_CONFIG["epochs"]

model = UNet(data[0][0].shape[0], data[0][1].shape[0])
# model = PixelWiseRegressor(data[0][0].shape[0], data[0][1].shape[0])
model = model.to(device)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


loss_fn = nn.MSELoss()


train_idx, val_idx, _ = train_val_test_split(data, seed=42, test_indices_path=Path(cfg["test_indices"]))
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
        val_loss = val_loop(val_dataloader, model, loss_fn, device)
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            save_dir = SAVED_MODELS_DIR / f'model: {model.__repr__()} training_start_time: {start_timestamp} datafile: {cfg["datafile"]} strat: {cfg["test_indices"].split("/")[-1]}'
            os.makedirs(save_dir, exist_ok=True)
            model_state_path = save_dir / 'best_model_state'
            model_path = save_dir / 'best_model'
            torch.save(model.state_dict(), model_state_path)
            torch.save(model, model_path)
        if epoch - best_epoch >= EARLY_STOP:
             print(f"Early stopping on epoch {epoch}")
             break
print(f"Best loss: {best_loss}")


