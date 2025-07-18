import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch import nn
from pathlib import Path
import os
from models.GAN import PatchDiscriminatorConditional, GeneratorUNetRegressionSEConditional, GeneratorUNetRegressionSEConditional2, GeneratorUNetRegressionRandom, PatchDiscriminatorRegressionRandom, DCGANGenerator, DCGANDiscriminator
from torch.utils.data import Subset, DataLoader
from utils.transforms import RescaledRotationTransform, ToTensor, GANTransform
from utils.config import PROJECT_ROOT, RELEVANT_CONFIG, RAW_CONFIG
from torch.optim import AdamW
import time
from utils.splitter import train_val_test_split_temp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import sys, importlib
from utils.gandataset import GANDataset, TestSubset
import scipy.ndimage as ndimage
import torch.nn.functional as F


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

def plot_from_test_dataloader(G, test_dataloader, dataset, device, vmax, season, save_dir, model_name, num_plots):
    test_temp_units = len(test_dataloader.dataset.indices)
    fig, ax = plt.subplots(nrows=num_plots, ncols=2, figsize=(12, 4 * num_plots))
    total_rmse = 0
    total_mae = 0
    for i, (X, y) in enumerate(test_dataloader):
        real_idx = test_dataloader.dataset.indices[i]
        X, y = X.float().to(device), y.float()
        fake_y = G(X).detach().cpu().numpy()
        y, fake_y = dataset.std_label*y+dataset.mean_label, dataset.std_label*fake_y+dataset.mean_label
        y, fake_y = y.numpy(), fake_y.numpy()
        rmse = np.sqrt(np.mean((y[0][0] - fake_y[0][0])**2, dtype=np.float32))
        mae = np.mean(np.abs(y[0][0] - fake_y[0][0]))
        total_rmse += rmse
        total_mae += mae
        vmin = 0
        ax[i, 0].imshow(y[0][0], cmap='viridis', vmin=vmin, vmax=vmax)
        # ax[i, 0].set_title("Real MLD | Date: {} | Season: {} | Model: {}".format(dataset.full_dates[real_idx], season, model_name))
        ax[i, 0].set_title("Real MLD | Season: {}".format(season))        
        ax[i, 1].imshow(fake_y[0][0], cmap='viridis', vmin=vmin, vmax=vmax)
        ax[i, 1].set_title("Generated MLD | RMSE: {:.2f}, MAE: {:.2f}".format(rmse, mae))
        if i == num_plots-1:
            break

    # total_rmse /= len(test_dataloader)
    # total_mae /= len(test_dataloader)
    total_rmse /= num_plots
    total_mae /= num_plots
    fig.suptitle(f"Test Results | Average RMSE: {total_rmse:.2f}, Average MAE: {total_mae:.2f}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    plt.savefig(save_dir / "results.png", dpi=200)
    plt.show()
    return total_rmse, total_mae

def fetch_info(info_path):
    with open(info_path, "r") as f:
        info_text = f.read()
        info = {}
        for line in info_text.strip().split('\n'):
            if ':: ' in line:
                key, value = line.split(':: ', 1)
                value = value.strip()
                if value == "None":
                    value = None
                elif value == "True":
                    value = True
                elif value == "False":
                    value = False
                info[key.strip()] = value
        return info

def main(args):

    cfg = RELEVANT_CONFIG
    root = Path(PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    filepath = args.filepath
    save_dir = Path(PROJECT_ROOT)/filepath

    info_path =  save_dir / 'training_info.txt'
    info_dict = fetch_info(info_path)

    test_indices_path = info_dict["test_indices"]
    test_indices_path = Path(PROJECT_ROOT)/test_indices_path

    season = info_dict["season"]
    groupby = info_dict["groupby"]

    dataset_file = root/info_dict["data_file"]


    # data_aug = RescaledRotationTransform(scaling_interval=(1, 1.2), degree_range=0)
    dataset = GANDataset(filepath = dataset_file, normalize=True, season=season, groupby=groupby)

    _, _, test_idx = train_val_test_split_temp(dataset, seed=42, test_frac=0.1, val_frac=0.135, test_indices_path=test_indices_path)
    test_data = TestSubset(dataset, test_idx)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)


    model = torch.load(save_dir / 'best_G_model.pt', map_location=device, weights_only=False)
    model.eval()

    model_name = model.name() if hasattr(model, 'name') else model.__class__.__name__
    print(f"Model name: {model_name}")

    max_dict = {"summer" : 70, "spring" : 70, "winter" : 100, "autumn" : 100}
    vmax = getattr(max_dict, season, 100)

    # if not full_frames:
    #     pred_map, mld_map = get_pred_and_mld_tensors(model, test_dataloader, dataset, device)
    #     plot_pred_and_mld_maps(pred_map, mld_map, save_dir, model_name, vmax, season)
    # else:
    total_rmse, total_mae = plot_from_test_dataloader(model, test_dataloader, dataset, device, vmax, season, save_dir, model_name, 30)
    update_values(info_path, {"rmse":total_rmse, "mae":total_mae})




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model on 1/12-degree mld data")
    parser.add_argument('--filepath', default='data/WaterOnlyMonthlySmall/WaterOnlyMonthlyExtendedSeasonalitySmall.nc', type=str, help="Path to the dataset file")
    args = parser.parse_args()
    main(args)