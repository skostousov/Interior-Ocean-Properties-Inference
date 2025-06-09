from trainonly_xarray import train_loop, val_loop
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

project_root = Path(PROJECT_ROOT)
model_relative_path = "saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250603_220037>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_01.pt>"
save_dir = project_root / model_relative_path

info_txt_path = save_dir / "training_info.txt"
with open(info_txt_path, "r") as f:
    info_text = f.read()
info = {}
for line in info_text.strip().split('\n'):
    if ':' in line:
        key, value = line.split(':', 1)
        info[key.strip()] = value.strip()

training_completed = info["training_completed"]
if not training_completed:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = torch.load(save_dir/'best_model', map_location=device, weights_only=False)

    start_epoch = info['t']



    for epoch in range(start_epoch, end_epoch):
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




