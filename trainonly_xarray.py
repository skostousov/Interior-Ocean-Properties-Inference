import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch import nn
from pathlib import Path
import os
from models.UNET_regression import UNetRegression
from models.UNET_regressionSE import UNetRegressionSE
from models.simple_CNN_regression import PixelWiseRegressor
from torch.utils.data import Subset, DataLoader
from utils.datasettemporalxarray import XArrayDataset, TestSubsetRegression
from utils.datasettemporal import TemporalDataset, TestSubsetRegression
from torchvision.transforms import Normalize, Compose
from utils.transforms import RescaledRotationTransform, ToTensor 
from utils.config import PROJECT_ROOT, RELEVANT_CONFIG, RAW_CONFIG
from torch.optim import AdamW
import time
from utils.splitter import train_val_test_split_temp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import sys, importlib

sys.modules.setdefault("numpy._core", importlib.import_module("numpy.core"))
torch.backends.cudnn.benchmark = True

cfg = RELEVANT_CONFIG
root = Path(PROJECT_ROOT)
submode = RELEVANT_CONFIG["submode"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data_aug = RescaledRotationTransform()
data = XArrayDataset(transform=data_aug)

batch_size = cfg['training']["batch_size"]
epochs = cfg['training']["epochs"]

model = UNetRegressionSE(data[0][0].shape[0], data[0][1].shape[0])
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
loss_fn = nn.L1Loss()


train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_indices_path=Path(cfg['data'][submode]["test_indices"]), gen_new=True)
train_data, val_data, = Subset(data, train_idx), Subset(data, val_idx)

print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")
print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Val dataset shape: img: {val_data[0][0].shape}, lbl: {val_data[0][1].shape}")

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

best_loss = 1000000000000000000
train_loss = 1000000000000000000
best_epoch=0

start_timestamp = time.strftime('%Y%m%d_%H%M%S')
model_name = f"MODEL:{model.name()}>TRAINSTART:{start_timestamp}>DATAFILE:{(cfg['data'][submode]['output_file']).replace('/', '_')}>STRAT:{(cfg['data'][submode]['test_indices']).replace('/', '_')}>"
model_dir = cfg["training"]["model_save_dest"]
save_dir = root / model_dir / model_name
os.makedirs(save_dir, exist_ok=True)

writer= SummaryWriter(save_dir / 'tensorboard_logs')
info_path =  save_dir / 'training_info.txt'
info_bef = {
    "start_time": start_timestamp,
    "data_file": f"{cfg['data']['data_dir']}/{cfg['data'][submode]['output_file']}",
    "test_indices": f"{cfg['data'][submode]['test_indices']}",
    "total_epochs": epochs,
    "batch_size": batch_size,
    "model": model.__repr__(),
    "optimizer": optimizer.__repr__(),
    "loss_fn": loss_fn.__repr__(),
    "scheduler": scheduler.__repr__(),
    "train_dataset_size": len(train_data),
    "val_dataset_size": len(val_data),
    "test_dataset_size": len(test_idx),
    "transform": data.transform.__repr__() if data.transform else None,
    "target_transform": data.target_transform.__repr__() if data.target_transform else None,
    "downsample": data.downsample if hasattr(data, 'downsample') else None,
    "grid_size": data.grid_size if hasattr(data, 'grid_size') else None,
    "datatype": data.name,
    "current_epoch": 0,
    "training_completed": False,
    "best_epoch": best_epoch,
    "best_val_loss": best_loss,
    "corresponding_train_loss": train_loss,
    "early_stopping_thresh": cfg["training"]["early_stopping_thresh"],}

with open(info_path, 'w') as f:
    for key, value in info_bef.items():
        f.write(f"{key}: {value}\n")

def update_values(info_path, key_values):
    info = {}
    with open(info_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.rstrip('\n').split(':', 1)
            info[key.strip()] = val.strip()
    for key, value in key_values:
        info[key] = value
    with open(info_path, 'w') as f:
        for key, val in info.items():
            f.write(f"{key}: {val}\n")

def train_loop(model, train_dataloader, optimizer, loss_fn, device):
    size = len(train_dataloader)
    total_size = len(train_dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch % 200 == 0:
            loss, current = loss.item(), batch * batch_size + len(images)
            print(f"loss: {loss:>12f} [{current:>6d}/{total_size:>5d}]")
    total_loss /= size
    return total_loss

def val_loop(val_dataloader, model, loss_fn, device, scheduler):
    model.eval()
    num_batches = len(val_dataloader)
    val_loss = 0
    size = len(val_dataloader)
    with torch.no_grad():
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            val_loss += loss_fn(pred, label).item()
    val_loss /= num_batches
    print(f"Val Loss: {val_loss:>12f} \n")
    scheduler.step(val_loss)
    return val_loss

for epoch in range(0, epochs):
    print('Epoch {}:'.format(epoch + 1))
    model.train(True)
    train_loss = train_loop(model, train_dataloader, optimizer, loss_fn, device)
    update_values(info_path, {'current_epoch': epoch})
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    for name, p in model.named_parameters():
            writer.add_histogram(f"weights/{name}", p, epoch)
            if p.grad is not None:
                writer.add_histogram(f"gradients/{name}", p.grad, epoch)
    val_loss = val_loop(val_dataloader, model, loss_fn, device, scheduler)
    writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss}, epoch)
    if val_loss < best_loss:
        best_epoch = epoch
        best_loss = val_loss
        corresponding_train_loss = train_loss
        checkpoint = {
            'model_state':model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            }
        torch.save(checkpoint, save_dir / 'checkpoint.pt')
        torch.save(model, save_dir / 'best_model')
    update_values(info_path, {"best_epoch": best_epoch, "best_val_loss": best_loss, "corresponding_train_loss": corresponding_train_loss})
    if epoch - best_epoch >= cfg["training"]["early_stopping_thresh"]:
            print(f"Early stopping on epoch {epoch}")
            update_values(info_path, {'training_completed': True})
            break

print(f"Best loss: {best_loss}")
print(f"Training completed. Best model saved at {save_dir / 'best_model'}")