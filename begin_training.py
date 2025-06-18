import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch import nn
from pathlib import Path
import os
from models.UNET_regression import UNetRegression
from models.UNET_regressionSE import UNetRegressionSE
from models.simple_CNN_regression import PixelWiseRegressor
from models.DA_CNN import DA_CNN
from models.CNN_EBAM import EBAM_CNN
from torch.utils.data import Subset, DataLoader
from utils.datasettemporal import TestSubsetRegression
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

def update_values(info_path, key_values):
    info = {}
    with open(info_path, 'r') as f:
        for line in f:
            if ':: ' not in line:
                continue
            key, val = line.rstrip('\n').split(':: ', 1)
            info[key.strip()] = val.strip()
    for key, value in key_values.items():
        info[key] = value
    with open(info_path, 'w') as f:
        for key, val in info.items():
            f.write(f"{key}:: {val}\n")

def train_loop(model, train_dataloader, optimizer, loss_fn, device):
    size = len(train_dataloader)
    batch_size = train_dataloader.batch_size
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


def fetch_data_processor(name):
    from utils.datasettemporalxarray import XArrayDataset
    from utils.datasettemporal import TemporalDataset
    from utils.dataset025 import PaperlikeDataset
    data_processors = {
        XArrayDataset.name(): XArrayDataset,
        TemporalDataset.name(): TemporalDataset,
        PaperlikeDataset.name(): PaperlikeDataset,
    }
    return data_processors[name]


def main(args):
    models = {
        "UNetRegression": (UNetRegression, {"first_out": getattr(args, "first_out", 64)}),
        "UNetRegressionSE": (UNetRegressionSE, {"base_filters": getattr(args, "base_filters", 64), "reduction": getattr(args, "reduction", 16)}),
        "PixelWiseRegressor": (PixelWiseRegressor, {}),
        "DA_CNN": (DA_CNN, {"first_layer_filters": getattr(args, "first_layer_filters", 64), "kernel_size": getattr(args, "kernel_size", 3)}),
        "EBAM_CNN": (EBAM_CNN, {"num_heads": getattr(args, "num_heads", 4)}),
    }

    cfg = RELEVANT_CONFIG
    root = Path(PROJECT_ROOT)
    submode = RELEVANT_CONFIG["submode"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if args.grid_size is not None:
        grid_size = args.grid_size
    else:
        grid_size = cfg['data']['grid_size']

    data_aug = RescaledRotationTransform()
    print(args.downsample)
    data = fetch_data_processor(args.data_processors)(transform=data_aug, grid_size=grid_size, downsample=bool(args.downsample))

    batch_size = args.batch_size
    epochs = cfg['training']["epochs"]

    model = models[args.model][0](data[0][0].shape[0], data[0][1].shape[0], grid_size=grid_size, **models[args.model][1])
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
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
        "model": model.__repr__().replace("\n", r"\n"),
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "train_dataset_size": len(train_data),
        "val_dataset_size": len(val_data),
        "test_dataset_size": len(test_idx),
        "transform": data.transform.__class__.__name__ if data.transform else None,
        "target_transform": data.target_transform.__class__.__name__ if data.target_transform else None,
        "downsample": data.downsample if hasattr(data, 'downsample') else None,
        "grid_size": data.grid_size,
        "data_processor": data.name(),
        "current_epoch": 0,
        "training_completed": False,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "corresponding_train_loss": train_loss,
        "early_stopping_thresh": cfg["training"]["early_stopping_thresh"],
        "lr": args.lr,
        "model_specific_args": models[args.model][1],
        }

    with open(info_path, 'w') as f:
        for key, value in info_bef.items():
            f.write(f"{key}:: {value}\n")

    num_epochs = args.num_epochs

    if num_epochs is None:
        num_epochs = epochs

    checkpoint = {
                'model_state':model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                }
    torch.save(checkpoint, save_dir / 'checkpoint.pt')
    torch.save(model, save_dir / 'best_model')

    # for epoch in range(0, epochs):
    for epoch in range(0, num_epochs):
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
        update_values(info_path, {'current_epoch': epoch})
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
        elif epoch == epochs:
                print(f"finished all epochs")
                update_values(info_path, {'training_completed': True})
                break

    print(f"Best loss: {best_loss}")
    print(f"Training completed. Best model saved at {save_dir / 'best_model'}")

if __name__ == "__main__":
    from utils.datasettemporalxarray import XArrayDataset
    from utils.datasettemporal import TemporalDataset
    from utils.dataset025 import PaperlikeDataset
    import argparse
    parser = argparse.ArgumentParser(description="Train a model on xarray data")
    subs = parser.add_subparsers(dest='model', required=True, help='Model to train', )
    m1 = subs.add_parser('UNetRegression', help='Train UNetRegression model')
    m2 = subs.add_parser('UNetRegressionSE', help='Train UNetRegressionSE model')
    m3 = subs.add_parser('PixelWiseRegressor', help='Train PixelWiseRegressor model')
    m4 = subs.add_parser('DA_CNN', help='Train DA_CNN model')
    m5 = subs.add_parser('EBAM_CNN', help='Train EBAM_CNN model')
    m1.add_argument('--first_out', type=int, default=64, help='Number of filters in the first layer of UNetRegression')
    m2.add_argument('--reduction', type=int, default=16, help='Reduction factor for UNetRegression')
    m2.add_argument('--base_filters', type=int, default=64, help='Base filters for UNetRegressionSE')
    m4.add_argument('--first_layer_filters', type=int, default=64, help='Number of filters in the first layer of DA_CNN')
    m4.add_argument('--kernel_size', type=int, default=3, choices=[1, 3], help='Kernel size for DA_CNN')
    m5.add_argument('--num_heads', type=int, default=4, help='Number of heads for EBAM_CNN')
    # parser.add_argument('--model', type=str, default='UNetRegressionSE', choices=['UNetRegression', 'UNetRegressionSE', 'PixelWiseRegressor', 'DA_CNN', 'EBAM_CNN'], help='Model to train')
    parser.add_argument('--num_epochs', type=int, help="number of epochs to train for")
    parser.add_argument('--data_processors', type=str, default = TemporalDataset.name(), choices=[XArrayDataset.name(), TemporalDataset.name(), PaperlikeDataset.name()], help='Data processor to use')
    parser.add_argument('--grid_size', type=int, default=21)
    parser.add_argument('--downsample', default=False, type=bool)
    parser.add_argument('--lr', default = 1e-4, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    args = parser.parse_args()
    main(args)