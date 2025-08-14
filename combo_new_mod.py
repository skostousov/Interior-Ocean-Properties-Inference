import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch import nn
from pathlib import Path
import os
from models.UNET_regression import UNetRegression
from models.UNET_regressionSE import UNetRegressionSE
from models.downscaledUNetSE import UNetRegressionSE as downscaledUNetSE
from models.downscaledUNet import UNetRegression as downscaledUNet
from models.simple_CNN_regression import PixelWiseRegressor
from models.CNN_EBAM import EBAM_CNN
from models.GAN import GeneratorUNetRegressionSEConditional
from models.DA_CNN import DA_CNN
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Normalize, Compose
from utils.transforms import RescaledRotationTransform, ToTensor, GANTransformRotate
from utils.config import PROJECT_ROOT, RELEVANT_CONFIG, RAW_CONFIG
from torch.optim import AdamW
import time
from utils.splitter import train_val_test_split_temp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import sys, importlib
from utils.datasettemporal_new_mod import TestSubsetRegressionNewMod as TestSubset
from utils.datasettemporal_new_mod import TemporalDatasetNewMod as TemporalDataset
import scipy.ndimage as ndimage
from models.UNetfullimageoutput import UNet


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

def plot_grids(test_dataloader, model, device):
    # test_idx = test_dataloader.dataset.dataset.indices
    test_idx = test_dataloader.dataset.indices


    data = test_dataloader.dataset.dataset


    test_temps = list(set([data.grid_and_centre_coords_and_temp_unit[i][-1] for i in test_idx]))
    print(len(test_temps))

    mld_labels = np.zeros((len(test_temps), data.feature_map.shape[-2], data.feature_map.shape[-1]))
    mld_preds = np.zeros((len(test_temps), data.feature_map.shape[-2], data.feature_map.shape[-1]))


    for X, y, extra_info in test_dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        month_idx = int(extra_info[-1])
        month_pos = test_temps.index(month_idx)
        lat = int((extra_info[1][0]).item())
        lon = int((extra_info[1][1]).item())
        # temp = extra_info[2]
        # time_steps.add(temp.item())
        print(f"Processing lat {lat}, lon {lon}, temp {month_idx}")
        mld_labels[month_pos, extra_info[0][0]:extra_info[0][0]+data.grid_size, extra_info[0][1]:extra_info[0][1]+data.grid_size] = y.item() * (data.std_label) + data.mean_label
        mld_preds[month_pos, extra_info[0][0]:extra_info[0][0]+data.grid_size, extra_info[0][1]:extra_info[0][1]+data.grid_size] = preds.item() * (data.std_label) + data.mean_label

    return mld_labels, mld_preds, test_temps

def general_plot(mld_labels, mld_preds, test_temps, season, model_name, save_dir, num_to_plot=None,):
    loss = 0
    print(season)
    max_dict = {"summer":40, "spring":70, "winter":90, "autumn":70}
    # vmax = max_dict.get(season, 70)
    # print(vmax)

    mae_loss = nn.L1Loss()

    if not num_to_plot:
        num_to_plot = len(test_temps)

    fig, ax = plt.subplots(max(1, num_to_plot), 2, figsize=(15, 8* num_to_plot))    
    ax = np.atleast_2d(ax)
    total_rmse = 0
    total_mae = 0

    y_true_flat = mld_labels.flatten()
    y_pred_flat = mld_preds.flatten()

    mass_mae = np.mean(np.abs(y_true_flat - y_pred_flat))

    mass_rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))

    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    mass_r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    print(f"MAE: {mass_mae:.4f}, RMSE: {mass_rmse:.4f}, R^2: {mass_r2:.4f}")


    for i, t in enumerate(test_temps[:num_to_plot]):
        vmax = max_dict.get(season, False)
        pred_map = mld_preds[i]
        label_map = mld_labels[i]    
        if not vmax:
            vmax = np.max([np.max(pred_map), np.max(label_map)])
        im_0 = ax[i, 0].imshow(label_map, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        ax[i, 0].set_title('Target Map for Time Step ' + str(t))
        ax[i, 0].set_xlabel('Longitude Index')
        ax[i, 0].set_ylabel('Latitude Index')
        fig.colorbar(im_0, label='Actual MLD (m)', ax=ax[i, 0])
        im_1 = ax[i, 1].imshow(pred_map, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        mae = mae_loss(torch.tensor(label_map), torch.tensor(pred_map)).item()
        rmse = np.sqrt(np.mean((pred_map - label_map)**2))
        total_rmse += rmse
        total_mae += mae
        ax[i, 1].set_title('Prediction Map for Time Step ' + str(t) + " MAE: " + f"{mae:.7f}" + " RMSE: " + f"{rmse:.2f}")
        ax[i, 1].set_xlabel('Longitude Index')
        ax[i, 1].set_ylabel('Latitude Index')
        fig.colorbar(im_1, label="Predicted MLD (m)", ax=ax[i, 1])
    total_rmse = total_rmse / num_to_plot
    total_mae = total_mae / num_to_plot
    plt.suptitle(f"Season: {season}\n{model_name} \n \n RMSE: {mass_rmse:.2f} | MAE: {mass_mae:.2f} | R2: {mass_r2:.2f}")
    plt.subplots_adjust(hspace=0.5)
    # fig.savefig(save_dir / "results.png", dpi=300)
    plt.savefig(save_dir / f"results_rmse:{mass_rmse:.2f}_mae:{mass_mae:.2f}_r2:{mass_r2:.2f}.png", dpi=200)

    fig, ax = plt.subplots(max(1, num_to_plot), 1, figsize=(6, 5* num_to_plot))
    for i, t in enumerate(test_temps[:num_to_plot]):
        if len(test_temps) == 1:
            ax = [ax]  # Ensure ax is iterable if there's only one time step
        pred_map = mld_preds[i]
        label_map = mld_labels[i]
        im_0 = ax[i].imshow(abs(label_map - pred_map), origin='lower', vmin=0, vmax=70, cmap='viridis')
        ax[i].set_title('Difference Map for Time Step ' + str(t))
        ax[i].set_xlabel('Longitude Index')
        ax[i].set_ylabel('Latitude Index')
        fig.colorbar(im_0, label='MLD MAE (m)', ax=ax[i])
    plt.suptitle(f"{model_name} \n Difference Results")
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    fig.savefig(save_dir / "results_diff.png", dpi=300)

    return mass_mae, mass_rmse, mass_r2

def plot_full(test_dataloader, model, device):
    test_idx = test_dataloader.dataset.dataset.indices


    # test_temps = list(set([data.indices[i] for i in test_idx]))
    test_temps = list(set(test_dataloader.dataset.indices))

    data = test_dataloader.dataset.dataset

    mld_labels = np.zeros((len(test_temps), data.feature_map.shape[-2], data.feature_map.shape[-1]))
    mld_preds = np.zeros((len(test_temps), data.feature_map.shape[-2], data.feature_map.shape[-1]))

    for i, (X, y), in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        preds = model(X)
        # month_idx = int(extra_info[-1])
        # month_pos = test_temps.index(month_idx)
        # lat = int((extra_info[1][0]).item())
        # lon = int((extra_info[1][1]).item())
        # print(f"Processing lat {lat}, lon {lon}, temp {month_idx}")
        y, preds = y.cpu(), preds.cpu()
        preds = preds.detach()
        mld_labels[i] = y * (data.std_label) + data.mean_label
        mld_preds[i] = preds * (data.std_label) + data.mean_label
    
    return mld_labels, mld_preds, test_temps

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
        # print(images.shape, pred.shape, labels.shape)
        # print(torch.isnan(images).sum().item(), torch.isnan(pred).sum().item(), torch.isnan(labels).sum().item())
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * batch_size + len(images)
            print(f"loss: {loss:>12f} [{current:>7d}/{total_size:>7d}]")
    total_loss /= size
    print(f"Train Loss: {total_loss:>12f}")
    return total_loss

def val_loop(val_dataloader, model, loss_fn, device, scheduler, full):
    model.eval()
    num_batches = len(val_dataloader)
    val_loss = 0
    size = len(val_dataloader)
    with torch.no_grad():
        if not full:
            for image, label, _ in val_dataloader:
                image, label = image.to(device), label.to(device)
                pred = model(image)
                val_loss += loss_fn(pred, label).item()
        else:
            for image, label, in val_dataloader:
                image, label = image.to(device), label.to(device)
                pred = model(image)
                val_loss += loss_fn(pred, label).item()
    val_loss /= num_batches
    print(f"Val Loss: {val_loss:>12f} \n")
    scheduler.step(val_loss)
    return val_loss


def main(args):
    models = {
        "UNetRegression": (UNetRegression, {"first_out": getattr(args, "first_out", 64)}),
        "UNetRegressionSE": (UNetRegressionSE, {"base_filters": getattr(args, "base_filters", 64), "reduction": getattr(args, "reduction", 16)}),
        "PixelWiseRegressor": (PixelWiseRegressor, {}),
        "DA_CNN": (DA_CNN, {"first_layer_filters": getattr(args, "first_layer_filters", 64), "kernel_size": getattr(args, "kernel_size", 3), "dropout": getattr(args, "dropout", 0.0), "dropout2": getattr(args, "dropout2", 0.0)}),
        "UNetFull": (UNet, {"base_channels": getattr(args, "base_channels", 64)}),
        "GANGenerator": (GeneratorUNetRegressionSEConditional, {}),
        "downscaledUNetSE": (downscaledUNetSE, {"base_filters": getattr(args, "base_filters", 64), "reduction": getattr(args, "reduction", 16), "dropout": getattr(args, "dropout", 0.0)}),
        "downscaledUNet": (downscaledUNet, {"first_out": getattr(args, "first_out", 64)}),
        "EBAM_CNN": (EBAM_CNN, {"num_heads": getattr(args, "num_heads", 4)}),
    }

    cfg = RELEVANT_CONFIG
    root = Path(PROJECT_ROOT)
    submode = RELEVANT_CONFIG["submode"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    season = args.season
    mld_res = args.mld_res
    feature_res = args.feature_res
    filepath = root / args.filepath
    groupby=args.groupby
    lat_lon=args.lat_lon
    full=args.full
    rim = args.rim
    custom_features = args.custom_features

    if custom_features:
        custom_features = list(custom_features.split(" "))

    # data_aug = RescaledRotationTransform(scaling_interval = (1, 1.4))
    data_aug = RescaledRotationTransform(scaling_interval = (1, 1.6))

    # data_aug = GANTransformRotate()
    data = TemporalDataset(transform=data_aug, filepath=filepath, season=season, mld_res=mld_res, feature_res=feature_res, groupby=groupby, lat_lon=lat_lon, full=full, rim=rim, custom_features=custom_features)

    batch_size = args.batch_size
    epochs = 100
    early_stopping_thresh = 50
    model = models[args.model][0](data[0][0].shape[0], out_channels=1, grid_size=data.grid_size, **models[args.model][1])

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
    
    datafile_name = args.filepath.split("/")[-1].replace(".nc", "")

    start_timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_name = f"SEASON:{season}>MLDRES:{mld_res:.2f}>FTRRES:{feature_res:.2f}>MODEL:{model.name()}>TRAINSTART:{start_timestamp}>DATAFILE:{str(filepath).split('/')[-1].replace('.nc', '')}>"
    model_dir = 'new_mod_model_results'
    save_dir = root / model_dir / datafile_name / season / model_name
    os.makedirs(save_dir, exist_ok=True)
    
    # test_indices_path = Path(root/f"test_indices/{datafile_name}/{str(data.grid_size)}+{mld_res:.2f}+{feature_res:.2f}+{str(season)}"+".pt")
    test_indices_path = save_dir/'test_indices.pt'
    # train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_indices_path=test_indices_path, gen_new=True)
    train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_frac=0.1, val_frac=0.135, test_indices_path=test_indices_path)


    train_data, val_data, test_data = Subset(data, train_idx), TestSubset(data, val_idx), TestSubset(data, test_idx)

    print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")
    print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Val dataset shape: img: {val_data[0][0].shape}, lbl: {val_data[0][1].shape}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    best_loss = float('inf')
    corresponding_train_loss = float('inf')
    best_epoch=0

    writer= SummaryWriter(save_dir / 'tensorboard_logs')
    info_path =  save_dir / 'training_info.txt'
    info_bef = {
        "start_time": start_timestamp,
        "data_file": f"{filepath}",
        "test_indices": f"{test_indices_path}",
        "total_epochs": epochs,
        "batch_size": batch_size,
        "model": model.__repr__().replace("\n", r"\n"),
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "loss_name": args.loss,
        "scheduler": scheduler.__class__.__name__,
        "features": data.features,
        "train_dataset_size": len(train_data),
        "val_dataset_size": len(val_data),
        "test_dataset_size": len(test_idx),
        "transform": data.transform.__class__.__name__ if data.transform else None,
        "target_transform": data.target_transform.__class__.__name__ if data.target_transform else None,
        "current_epoch": 0,
        "training_completed": False,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "corresponding_train_loss": corresponding_train_loss,
        "early_stopping_thresh": early_stopping_thresh,
        "lr": args.lr,
        "season": season,
        "model_specific_args": models[args.model][1],
        "mld_res": mld_res,
        "feature_res": feature_res,
        "groupby":groupby,
        "lat_lon":lat_lon,
        "full": full,
        "rim": rim
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
        val_loss = val_loop(val_dataloader, model, loss_fn, device, scheduler, full)
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
    model.eval()

    num_to_plot = args.num_to_plot

    if full:
        mld_labels, mld_preds, test_temps = plot_full(test_dataloader, model, device)
    else:
        mld_labels, mld_preds, test_temps = plot_grids(test_dataloader, model, device)

    total_mae, total_rmse, r2 = general_plot(mld_labels, mld_preds, test_temps, season, model.name(), save_dir, num_to_plot=num_to_plot)
    update_values(info_path, {'rmse': total_rmse, 'mae': total_mae, 'r2': r2})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model on 1-degree mld data")
    subs = parser.add_subparsers(dest='model', required=True, help='Model to train', )
    m1 = subs.add_parser('UNetRegression', help='Train UNetRegression model')
    m2 = subs.add_parser('UNetRegressionSE', help='Train UNetRegressionSE model')
    m4 = subs.add_parser('DA_CNN', help='Train DA_CNN model')
    m3= subs.add_parser('UNetFull', help='Train UNet model')
    m5 = subs.add_parser('GANGenerator')
    m6 = subs.add_parser('downscaledUNetSE', help='Train downscaled UNetSE model')
    m7 = subs.add_parser('downscaledUNet', help='Train downscaled UNet model')
    m8 = subs.add_parser('EBAM_CNN', help='Train EBAM_CNN model')
    m1.add_argument('--first_out', type=int, default=64, help='Number of filters in the first layer of UNetRegression')
    m2.add_argument('--reduction', type=int, default=16, help='Reduction factor for UNetRegression')
    m2.add_argument('--base_filters', type=int, default=64, help='Base filters for UNetRegressionSE')
    m4.add_argument('--first_layer_filters', type=int, default=64, help='Number of filters in the first layer of DA_CNN')
    m4.add_argument('--kernel_size', type=int, default=3, choices=[1, 3], help='Kernel size for DA_CNN')
    m4.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for DA_CNN')
    m4.add_argument('--dropout2', type=float, default=0.0, help='Second dropout rate for DA_CNN')
    m3.add_argument('--base_channels', type=int, default=64)
    # parser.add_argument('--model', type=str, default='UNetRegressionSE', choices=['UNetRegression', 'UNetRegressionSE', 'PixelWiseRegressor', 'DA_CNN', 'EBAM_CNN'], help='Model to train')
    m6.add_argument('--base_filters', type=int, default=64, help='Base filters for downscaled UNetSE')
    m6.add_argument('--reduction', type=int, default=16, help='Reduction factor for downscaled UNetSE')
    m6.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for downscaled UNetSE')
    m7.add_argument('--first_out', type=int, default=64, help='Number of filters in the first layer of downscaled UNet')
    m8.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for EBAM_CNN')
    parser.add_argument('--num_epochs', default = 1, type=int, help="number of epochs to train for")
    parser.add_argument('--lr', default = 1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--season', default='all', type=str)
    parser.add_argument('--mld_res', default=1, type=float, help='Resolution of MLD data')
    parser.add_argument('--feature_res', default=1/12, type=float, help='Resolution of feature data')
    parser.add_argument('--loss', default='MSE', type=str, choices=['MSE', 'L1'], help='Loss function to use for training')
    parser.add_argument('--filepath', type=str, default = "data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc")
    parser.add_argument('--groupby', default='months', type=str, choices=['days', 'months', 'years'], help="Group by days, months, or years")
    parser.add_argument('--lat_lon', default=True, type=bool)
    parser.add_argument('--full', default=False, type=bool, help="Whether to use full dataset or not")
    parser.add_argument('--rim', default=0, type=int, help="Rim padding size")
    parser.add_argument('--num_to_plot', default=None, type=int, help="Number of time steps to plot. If None, all time steps will be plotted.")
    parser.add_argument('--custom_features', default=False, type=str, help="List of custom features to use. If None, all features will be used.")
    args = parser.parse_args()
    main(args)