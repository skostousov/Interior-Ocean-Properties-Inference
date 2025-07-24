import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch import nn
from pathlib import Path
import os
from models.downscaledUNet import UNetRegression
from models.downscaledUNetSE import UNetRegressionSE
from models.simple_CNN_regression import PixelWiseRegressor
from models.DA_CNN import DA_CNN
from torch.utils.data import Subset, DataLoader
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
from data.argo.alternate_dataset import myDataset, TestSubset
import scipy.ndimage as ndimage


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
    for batch, (images, labels, _) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * batch_size + len(images)
            print(f"loss: {loss:>12f} [{current:>6d}/{total_size:>5d}]")
    total_loss /= size
    print(f"Train Loss: {total_loss:>12f}")
    return total_loss

def val_loop(val_dataloader, model, loss_fn, device, scheduler):
    model.eval()
    num_batches = len(val_dataloader)
    val_loss = 0
    size = len(val_dataloader)
    with torch.no_grad():
        for image, label, _ in val_dataloader:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            val_loss += loss_fn(pred, label.unsqueeze(1)).item()
    val_loss /= num_batches
    print(f"Val Loss: {val_loss:>12f} \n")
    scheduler.step(val_loss)
    return val_loss


def main(args):
    models = {
        "UNetRegression": (UNetRegression, {"first_out": getattr(args, "first_out", 64)}),
        "UNetRegressionSE": (UNetRegressionSE, {"base_filters": getattr(args, "base_filters", 64), "reduction": getattr(args, "reduction", 16)}),
        "PixelWiseRegressor": (PixelWiseRegressor, {}),
        "DA_CNN": (DA_CNN, {"first_layer_filters": getattr(args, "first_layer_filters", 64), "kernel_size": getattr(args, "kernel_size", 3)}),
    }

    root = Path(PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    season = args.season

    data_aug = RescaledRotationTransform(scaling_interval=(1, 1.2))
    data = myDataset(transform=data_aug, season=season, coarsen=args.coarsen)

    batch_size = args.batch_size
    epochs = 100
    grid_size=12
    early_stopping_thresh = 50
    model = models[args.model][0](data[0][0].shape[0], out_channels=1, grid_size=grid_size, **models[args.model][1])

    def initialize_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(initialize_weights)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    loss_dict = {'L1' : nn.L1Loss, 'MSE' : nn.MSELoss}

    loss_fn = loss_dict[args.loss]()

    # test_indices_path = Path((cfg['data'][submode]["test_indices"]).replace('.pt', f'{str(season)}'+'.pt'))
    test_indices_path= Path(f"low_res_indices/test_indices_{str(season)}.pt")
    # train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_indices_path=test_indices_path, gen_new=True)
    train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_frac=0.1, val_frac=0.135, gen_new=True)


    train_data, val_data, test_data = Subset(data, train_idx), TestSubset(data, val_idx), TestSubset(data, test_idx)

    print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")
    print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Val dataset shape: img: {val_data[0][0].shape}, lbl: {val_data[0][1].shape}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

    best_loss = float('inf')
    corresponding_train_loss = float('inf')
    best_epoch=0

    start_timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_name = f"SEASON:{args.season}>LOSS:{args.loss}>MODEL:{model.name()}>TRAINSTART:{start_timestamp}>"
    model_dir = 'lower_res_models'
    
    save_dir = root / model_dir / season / model.name() / model_name
    os.makedirs(save_dir, exist_ok=True)

    writer= SummaryWriter(save_dir / 'tensorboard_logs')
    info_path =  save_dir / 'training_info.txt'
    info_bef = {
        "start_time": start_timestamp,
        "total_epochs": epochs,
        "batch_size": batch_size,
        "model": model.__repr__().replace("\n", r"\n"),
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "loss_name": args.loss,
        "scheduler": scheduler.__class__.__name__,
        "train_dataset_size": len(train_data),
        "val_dataset_size": len(val_data),
        "test_dataset_size": len(test_idx),
        "transform": data.transform.__class__.__name__ if data.transform else None,
        "current_epoch": 0,
        "training_completed": False,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "corresponding_train_loss": corresponding_train_loss,
        "early_stopping_thresh": early_stopping_thresh,
        "lr": args.lr,
        "season": season,
        "model_specific_args": models[args.model][1],
        "coarsen":args.coarsen,
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
    best_checkpoint = {
                'best_model_state':model.state_dict(),
                'best_optimizer_state': optimizer.state_dict(),
                'best_scheduler_state': scheduler.state_dict(),
                }
    torch.save(checkpoint, save_dir / 'checkpoint.pt')
    torch.save(best_checkpoint, save_dir / 'best_checkpoint.pt')
    torch.save(model, save_dir / 'best_model')
    torch.save(model, save_dir / 'model')


    # for epoch in range(0, epochs):
    for epoch in range(0, num_epochs):
        print('Epoch {}:'.format(epoch + 1))
        train_loss = train_loop(model, train_dataloader, optimizer, loss_fn, device)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        for name, p in model.named_parameters():
                writer.add_histogram(f"weights/{name}", p, epoch)
                if p.grad is not None:
                    writer.add_histogram(f"gradients/{name}", p.grad, epoch)
        val_loss = val_loop(val_dataloader, model, loss_fn, device, scheduler)
        writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss}, epoch)
        update_values(info_path, {'current_epoch': epoch})
        checkpoint = {
                    'model_state':model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    }
        torch.save(checkpoint, save_dir / 'checkpoint.pt')
        torch.save(model, save_dir / 'model')
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            print(f"New best model saved with validation loss: {best_loss:.6f}")
            corresponding_train_loss = train_loss
            best_checkpoint = {
                'best_model_state':model.state_dict(),
                'best_optimizer_state': optimizer.state_dict(),
                'best_scheduler_state': scheduler.state_dict(),
                }
            torch.save(best_checkpoint, save_dir / 'best_checkpoint.pt')
            torch.save(model, save_dir / 'best_model')
        update_values(info_path, {"best_epoch": best_epoch, "best_val_loss": best_loss, "corresponding_train_loss": corresponding_train_loss})
        if epoch - best_epoch >= early_stopping_thresh:
                print(f"Early stopping on epoch {epoch}")
                update_values(info_path, {'training_completed': True})
                break
        elif epoch == epochs:
                print(f"finished all epochs")
                update_values(info_path, {'training_completed': True})
                break

    print(f"Best loss: {best_loss}")
    print(f"Training completed. Best model saved at {save_dir / 'best_model'}")

    model = torch.load(save_dir / 'best_model', map_location=device, weights_only=False)

    test_months = list(set([data.groups[i] for i in test_idx]))
    loss = 0

    max_dict = {"summer" : 70, "spring" : 70, "winter" : 100, "autumn" : 100}
    vmax = max_dict[season]

    mld_labels = np.zeros((len(test_months), len(data.argo_cut["latitude"]), len(data.argo_cut["longitude"])))
    mld_preds = np.zeros((len(test_months), len(data.argo_cut["latitude"]), len(data.argo_cut["longitude"])))

    for X, y, extra_info in test_dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        month_idx = int(extra_info[-1])
        month_pos = test_months.index(month_idx)
        lat = int((extra_info[0]).item() - data.argo_cut["latitude"].values.min())
        lon = int((extra_info[1]).item() - data.argo_cut["longitude"].values.min())
        temp = extra_info[2]
        print(f"Processing month {month_idx}, lat {lat}, lon {lon}, time {temp}")
        mld_labels[month_pos, lat, lon] = y.item() * (data.stds["mld"]) + data.means["mld"]
        mld_preds[month_pos, lat, lon] = preds.item() * (data.stds["mld"]) + data.means["mld"]
    mask = (mld_preds != 0).astype(float)
    masked_data = mld_preds * mask
    smoothed_data = ndimage.gaussian_filter(masked_data, sigma=(0, 1, 1), order=0)
    weights = ndimage.gaussian_filter(mask, sigma=(0, 1, 1), order=0)
    eps = 1e-8
    normalized = np.zeros_like(mld_preds)
    nonzero = mask.astype(bool)
    normalized[nonzero] = smoothed_data[nonzero] / (weights[nonzero] + eps)
    mld_preds_smoothed = normalized
##-------------##-------------##--------------#########################
    mae_loss = nn.L1Loss()
    fig, axs = plt.subplots(len(test_months), 3, figsize=(16, 5 * len(test_months)), constrained_layout=True)
    total_rmse = 0
    total_mae = 0
    for i, month in enumerate(test_months):
        im_0 = axs[i, 0].imshow(mld_labels[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
        fig.colorbar(im_0, ax=axs[i, 0], orientation='vertical', fraction=0.02, pad=0.04)
        axs[i, 0].set_title(f"Label - Month: {month}")
        im_1 = axs[i, 1].imshow(mld_preds[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
        fig.colorbar(im_1, ax=axs[i, 1], orientation='vertical', fraction=0.02, pad=0.04)
        rmse = np.sqrt(np.mean((mld_labels[i] - mld_preds[i])**2))
        mae= np.mean(np.abs(mld_labels[i] - mld_preds[i]))
        total_rmse += rmse
        total_mae += mae
        axs[i, 1].set_title(f"Prediction - Month: {month}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        axs[i, 2].imshow(mld_preds_smoothed[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
        fig.colorbar(axs[i, 2].imshow(mld_preds_smoothed[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower'), ax=axs[i, 2], orientation='vertical', fraction=0.02, pad=0.04)
        axs[i, 2].set_title(f"Smoothed Prediction - Month: {month}")
    total_rmse = total_rmse / len(test_months)
    total_mae = total_mae / len(test_months)
    plt.suptitle(f" {season} \n {model_name} \n \n total RMSE: {total_rmse:.2f} total MAE: {total_mae:.2f}")
    update_values(info_path, {"rmse": total_rmse, "mae": total_mae})


    fig.savefig(save_dir / f"rmse:{total_rmse:.2f}.png", dpi=300)

    fig, axs = plt.subplots(len(test_months), 2, figsize=(10, 5 * len(test_months)), constrained_layout=True)
    for i, month in enumerate(test_months):
        if len(test_months) == 1:
            ax = [ax]  # Ensure ax is iterable if there's only one time step
        im_2 = axs[i, 0].imshow(abs(mld_labels[i] - mld_preds[i]), cmap='coolwarm', vmin=0, vmax=40, origin='lower')
        fig.colorbar(im_2, ax=axs[i, 0], orientation='vertical', fraction=0.02, pad=0.04)
        axs[i, 0].set_title(f"Absolute Error - Month: {month}, RMSE: {np.sqrt(np.mean((mld_labels[i] - mld_preds[i])**2)):.2f}")
        im_3 = axs[i, 1].imshow(abs(mld_labels[i] - mld_preds_smoothed[i]), cmap='coolwarm', vmin=0, vmax=40, origin='lower')
        fig.colorbar(im_3, ax=axs[i, 1], orientation='vertical', fraction=0.02, pad=0.04)
        axs[i, 1].set_title(f"Absolute Error (smoothed) - Month: {month}, RMSE: {np.sqrt(np.mean((mld_labels[i] - mld_preds_smoothed[i])**2)):.2f}")
    plt.suptitle(f"{model_name} \n Difference Results")
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    fig.savefig(save_dir / "results_diff.png", dpi=600)


if __name__ == "__main__":
    from utils.datasettemporalxarray import XArrayDataset
    from utils.datasettemporal import TemporalDataset
    from utils.dataset025 import PaperlikeDataset
    import argparse
    parser = argparse.ArgumentParser(description="Train a model on 1-degree mld data")
    subs = parser.add_subparsers(dest='model', required=True, help='Model to train', )
    m1 = subs.add_parser('UNetRegression', help='Train UNetRegression model')
    m2 = subs.add_parser('UNetRegressionSE', help='Train UNetRegressionSE model')
    m4 = subs.add_parser('DA_CNN', help='Train DA_CNN model')
    m1.add_argument('--first_out', type=int, default=64, help='Number of filters in the first layer of UNetRegression')
    m2.add_argument('--reduction', type=int, default=16, help='Reduction factor for UNetRegression')
    m2.add_argument('--base_filters', type=int, default=64, help='Base filters for UNetRegressionSE')
    m4.add_argument('--first_layer_filters', type=int, default=64, help='Number of filters in the first layer of DA_CNN')
    m4.add_argument('--kernel_size', type=int, default=3, choices=[1, 3], help='Kernel size for DA_CNN')
    # parser.add_argument('--model', type=str, default='UNetRegressionSE', choices=['UNetRegression', 'UNetRegressionSE', 'PixelWiseRegressor', 'DA_CNN', 'EBAM_CNN'], help='Model to train')
    parser.add_argument('--num_epochs', type=int, help="number of epochs to train for")
    parser.add_argument('--lr', default = 1e-4, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--season', default='all', type=str)
    parser.add_argument('--loss', type=str, default='L1', choices=['L1', 'MSE'])
    parser.add_argument('--coarsen', type=int, default=1)
    args = parser.parse_args()
    main(args)