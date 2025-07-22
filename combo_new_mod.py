import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch import nn
from pathlib import Path
import os
from models.UNET_regression import UNetRegression
from models.downscaledUNetSE import UNetRegressionSE
from models.simple_CNN_regression import PixelWiseRegressor
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


sys.modules.setdefault("numpy._core", importlib.import_module("numpy.core"))
torch.backends.cudnn.benchmark = True

def plot_grids(test_dataloader, model):
    test_temps = list(set([data.grid_and_centre_coords_and_temp_unit[i][-1] for i in test_idx]))

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

def general_plot(mld_labels, mld_preds, test_temps, num_to_plot):
    loss = 0

    max_dict = {"summer" : 50, "spring" : 70, "winter" : 100, "autumn" : 100}
    vmax = getattr(max_dict, season, 100)

    mae_loss = nn.L1Loss()

    fig, ax = plt.subplots(max(1, num_to_plot), 2, figsize=(15, 8* num_to_plot))    
    ax = np.atleast_2d(ax)
    total_rmse = 0
    total_mae = 0

    for i, t in enumerate(test_temps[:num_to_plot]):
        pred_map = mld_preds[i]
        label_map = mld_labels[i]    
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
    plt.suptitle(f"Season: {season}\n{model_name} \n \n RMSE: {total_rmse:.2f} | MAE: {total_mae:.2f}")
    plt.subplots_adjust(hspace=0.5)
    # fig.savefig(save_dir / "results.png", dpi=300)
    plt.savefig(save_dir / f"results_rmse:{total_rmse:.2f}_mae:{total_mae:.2f}.png", dpi=200)

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

def plot_full(test_dataloader, model):
    test_temps = list(set([data.indices[i] for i in test_idx]))

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
        mld_labels[i] = y.item() * (data.std_label) + data.mean_label
        mld_preds[i] = preds.item() * (data.std_label) + data.mean_label
    
    return mld_labels, mld_preds, test_temps



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
        "DA_CNN": (DA_CNN, {"first_layer_filters": getattr(args, "first_layer_filters", 64), "kernel_size": getattr(args, "kernel_size", 3)}),
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

    data_aug = RescaledRotationTransform()
    # data_aug = GANTransformRotate()
    data = TemporalDataset(transform=data_aug, filepath=filepath, season=season, mld_res=mld_res, feature_res=feature_res, groupby=groupby, lat_lon=lat_lon, full=full)

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
    test_indices_path = Path((cfg['data'][submode]["test_indices"]).replace('.pt', f'{str(data.grid_size)}+{mld_res:.2f}+{feature_res:.2f}+{str(season)}'+'.pt'))
    # train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_indices_path=test_indices_path, gen_new=True)
    train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_frac=0.1, val_frac=0.135, gen_new=True)


    train_data, val_data, test_data = Subset(data, train_idx), TestSubset(data, val_idx), TestSubset(data, test_idx)

    print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")
    print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Val dataset shape: img: {val_data[0][0].shape}, lbl: {val_data[0][1].shape}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)

    best_loss = float('inf')
    corresponding_train_loss = float('inf')
    best_epoch=0

    start_timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_name = f"SEASON:{season}>MLDRES:{mld_res:.2f}>FTRRES:{feature_res:.2f}>MODEL:{model.name()}>TRAINSTART:{start_timestamp}>DATAFILE:{str(filepath).split('/')[-1].replace('.nc', '')}>"
    model_dir = 'dynamic_res_models'
    datafile_name = args.filepath.split("/")[-1].replace(".nc", "")
    save_dir = root / model_dir / datafile_name / season / model_name
    os.makedirs(save_dir, exist_ok=True)

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

    num_to_plot = 20

    if full:
        mld_labels, mld_preds, test_temps = plot_full(test_dataloader, model)
    else:
        mld_labels, mld_preds, test_temps = plot_grids(test_dataloader, model)

    general_plot(mld_labels, mld_preds, test_temps, num_to_plot)



if __name__ == "__main__":
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
    parser.add_argument('--num_epochs', default = 1, type=int, help="number of epochs to train for")
    parser.add_argument('--lr', default = 1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--season', default='all', type=str)
    parser.add_argument('--mld_res', default=1/3, type=float, help='Resolution of MLD data')
    parser.add_argument('--feature_res', default=1/12, type=float, help='Resolution of feature data')
    parser.add_argument('--loss', default='MSE', type=str, choices=['MSE', 'L1'], help='Loss function to use for training')
    parser.add_argument('--filepath', type=str, default = "data/WaterOnlyDailySmall/WaterOnlyDailyExtendedSeasonalitySmall.nc")
    parser.add_argument('--groupby', default='months', type=str, choices=['days', 'months', 'years'], help="Group by days, months, or years")
    parser.add_argument('--lat_lon', default=True, type=bool)
    parser.add_argument('--full', default=False, type=bool, help="Whether to use full dataset or not")
    args = parser.parse_args()
    main(args)