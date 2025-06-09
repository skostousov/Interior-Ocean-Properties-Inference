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
# from utils.transforms import RescaledRotationTransform, ToTensor, Compose
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
        if batch % 50 == 0:
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

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data_aug = RescaledRotationTransform()
# data = TemporalDataset(transform=data_aug)
data = XArrayDataset(transform=data_aug)


batch_size = cfg['training']["batch_size"]
epochs = cfg['training']["epochs"]

model = UNetRegressionSE(data[0][0].shape[0], data[0][1].shape[0])

model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# loss_fn = nn.MSELoss()

# loss_fn = nn.HuberLoss()

loss_fn = nn.L1Loss()


print("hello")
train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_indices_path=Path(cfg['data'][submode]["test_indices"]), gen_new=True)


train_data, val_data, = Subset(data, train_idx), Subset(data, val_idx)

# train_data, val_data = simple_train_val_split(data)

print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")
print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Val dataset shape: img: {val_data[0][0].shape}, lbl: {val_data[0][1].shape}")

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

best_loss = 1000000000000000000

def checkpoint(model, filename):
     torch.save(model.state_dict(), filename)

start_timestamp = time.strftime('%Y%m%d_%H%M%S')
model_name = f"MODEL:{model.name()}>TRAINSTART:{start_timestamp}>DATAFILE:{(cfg['data'][submode]['output_file']).replace('/', '_')}>STRAT:{(cfg['data'][submode]['test_indices']).replace('/', '_')}>"
model_dir = cfg["training"]["model_save_dest"]
save_dir = root / model_dir / model_name
# os.makedirs(save_dir, exist_ok=True)

writer= SummaryWriter(save_dir / 'tensorboard_logs')


best_epoch=0
for epoch in range(0, epochs):
        print('Epoch {}:'.format(epoch + 1))
        model.train(True)
        train_loss = train_loop(model, train_dataloader, optimizer, loss_fn, device)
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
            os.makedirs(save_dir, exist_ok=True)
            model_state_path = save_dir / 'best_model_state'
            model_path = save_dir / 'best_model'
            torch.save(model.state_dict(), model_state_path)
            torch.save(model, model_path)
        if epoch - best_epoch >= cfg["training"]["early_stopping_thresh"]:
             print(f"Early stopping on epoch {epoch}")
             break
print(f"Best loss: {best_loss}")


info = {
    "start_time": start_timestamp,
    "data_file": f"{cfg['data']['data_dir']}/{cfg['data'][submode]['output_file']}",
    "test_indices": f"{cfg['data'][submode]['test_indices']}",
    "epochs": epochs,
    "batch_size": batch_size,
    "model": model.__repr__(),
    "optimizer": optimizer.__repr__(),
    "loss_fn": loss_fn.__repr__(),
    "scheduler": scheduler.__repr__(),
    "best_loss": best_loss,
    "best_epoch": best_epoch,
    "corresponding_train_loss": corresponding_train_loss,
    "train_dataset_size": len(train_data),
    "val_dataset_size": len(val_data),
    "test_dataset_size": len(test_idx),
    "transform": data.transform.__repr__() if data.transform else None,
    "target_transform": data.target_transform.__repr__() if data.target_transform else None,
    "downsample": data.downsample if hasattr(data, 'downsample') else None,
    "grid_size": data.grid_size if hasattr(data, 'grid_size') else None,
    "datatype": data.name,
}


info_path =  save_dir / 'training_info.txt'
with open(info_path, 'w') as f:
    for key, value in info.items():
        f.write(f"{key}: {value}\n")
print(f"Training completed. Best model saved at {model_path}")


if cfg["training"]["immediate_test"]:
    test_data = TestSubsetRegression(data, test_idx)
    test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    num_workers=6,
    pin_memory=True,)

    model.eval()
    loss = 0

    after_model = []

    filepath = save_dir/"results.pkl"

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            images, labels, metadata = batch
            images_gpu = images.to(device)
            preds = model(images_gpu)
            loss += loss_fn(preds, labels.to(device)).item()
            preds, labels = preds.cpu(), labels.cpu()
            batch_dict = {"image": images, "label": labels, "pred": preds, "grid": metadata[0], "centre": metadata[1], "month": metadata[2]}
            after_model.append(batch_dict)

            if i % 100 == 0:
                print(f"Processed {i} batches")

            # Periodically append new results
            if i % 10000 == 0 and i > 0:
                with open(filepath, "ab") as f:
                    pickle.dump(after_model, f)
                print(f"Appended {i} batches")
                after_model.clear()

        # if i > num_samples:
        #     break
    # Save any remaining batches
        if len(after_model) > 0:
            with open(filepath, "ab") as f:
                pickle.dump(after_model, f)
    print(f"Total loss: {loss / len(test_dataloader)}")
    with open(info_path, 'a') as f:
        f.write(f"total_test_loss: {loss / len(test_dataloader)}\n")

    test_data = TestSubsetRegression(data, test_idx)

    def iter_pickled_batches(file_path):
        """Generator to yield batches from a pickled file."""
        with open(file_path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    print("Unpickling error encountered.")
                    break

    def get_t_from_pickled(file_path):
        time_steps = set()
        for batch in iter_pickled_batches(file_path):
            for entry in batch:
                time_steps.add(entry["month"].item())
        return tuple(time_steps)

    time_steps = get_t_from_pickled(filepath)

    mean, std = data.mean_label, data.std_label
    lat_range, lon_range = data.feature_map.shape[-2], data.feature_map.shape[-1]

    pred_maps = []
    label_maps = []
    for t in time_steps:
        pred_maps.append(np.zeros((lat_range, lon_range)))
        label_maps.append(np.zeros((lat_range, lon_range)))

    for batch in iter_pickled_batches(filepath):
        for entry in batch:
            entry["label_unnorm"] = ((entry["label"] * std) + mean).item()
            entry["pred_unnorm"] = ((entry["pred"] * std) + mean).item()
            t = entry["month"].item()
            t_idx = time_steps.index(t)
            lat, lon = entry["centre"][0].item(), entry["centre"][1].item()
            print(f"Processing time step: {t}, lat: {lat}, lon: {lon}")
            pred_maps[t_idx][lat, lon] = entry["pred_unnorm"]
            label_maps[t_idx][lat, lon] = entry["label_unnorm"]

    mae_loss = nn.L1Loss()

    fig, ax = plt.subplots(len(time_steps), 2, figsize=(15, 8* len(time_steps)))
    for i, t in enumerate(time_steps):
        pred_map = pred_maps[i]
        label_map = label_maps[i]    
        im_0 = ax[i, 0].imshow(label_map, origin='lower', vmin=0, vmax=100, cmap='viridis')
        ax[i, 0].set_title('Target Map for Time Step ' + str(t))
        ax[i, 0].set_xlabel('Longitude Index')
        ax[i, 0].set_ylabel('Latitude Index')
        fig.colorbar(im_0, label='Actual MLD (m)', ax=ax[i, 0])
        im_1 = ax[i, 1].imshow(pred_map, origin='lower', vmin=0, vmax=100, cmap='viridis')
        mae = mae_loss(torch.tensor(label_map), torch.tensor(pred_map)).item()
        ax[i, 1].set_title('Prediction Map for Time Step ' + str(t) + " MAE: " + f"{mae:.7f}")
        ax[i, 1].set_xlabel('Longitude Index')
        ax[i, 1].set_ylabel('Latitude Index')
        fig.colorbar(im_1, label="Predicted MLD (m)", ax=ax[i, 1])
    plt.suptitle(f"{model_name} \n Results")
    plt.subplots_adjust(hspace=0.5)
    fig.savefig(save_dir / "results.png", dpi=600)

    fig, ax = plt.subplots(len(time_steps), 1, figsize=(6, 5* len(time_steps)))
    for i, t in enumerate(time_steps):
        if len(time_steps) == 1:
            ax = [ax]  # Ensure ax is iterable if there's only one time step
        pred_map = pred_maps[i]
        label_map = label_maps[i]    
        im_0 = ax[i,].imshow(abs(label_map - pred_map), origin='lower', vmin=0, vmax=80, cmap='viridis')
        ax[i].set_title('Difference Map for Time Step ' + str(t))
        ax[i].set_xlabel('Longitude Index')
        ax[i].set_ylabel('Latitude Index')
        fig.colorbar(im_0, label='MLD MAE (m)', ax=ax[i])
    plt.suptitle(f"{model_name} \n Difference Results")
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    fig.savefig(save_dir / "results_diff.png", dpi=600)