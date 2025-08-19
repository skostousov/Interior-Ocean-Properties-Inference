from utils.config import RAW_CONFIG, PROJECT_ROOT, RELEVANT_CONFIG
from pathlib import Path
from torch.utils.data import DataLoader
from utils.splitter import test_indices
import torch
import pickle
from pathlib import Path
from utils.config import PROJECT_ROOT
import matplotlib.pyplot as plt
import numpy as np
from continue_training import fetch_info
import torch.nn as nn
import os
from data.argo.alternate_dataset import myDataset
from data.argo.alternate_dataset import TestSubset
import scipy.ndimage as ndimage
from combo_low_res import update_values
from utils.splitter import train_val_test_split_temp



def main(args):
    project_root = Path(PROJECT_ROOT)
    model_relative_path = args.model_relative_path
    model_name = model_relative_path.split("/")[-1]
    model_path = project_root / model_relative_path

    info_path = model_path / "training_info.txt"
    info = fetch_info(info_path)

    loss_dict = {'L1' : nn.L1Loss(), 'MSE' : nn.MSELoss()}
    loss_dict[loss_dict['L1'].__class__.__name__] = loss_dict['L1']
    loss_dict[loss_dict['MSE'].__class__.__name__] = loss_dict['MSE']

    loss_fn = loss_dict[info['loss_fn']]

    assert info['loss_fn'] == loss_fn.__class__.__name__, f"Loss function mismatch: {info['loss_fn']} != {loss_fn.__class__.__name__}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path/'best_model', map_location=device, weights_only=False)
    model.eval()

    season = info['season']

    max_dict = {"summer" : 70, "spring" : 70, "winter" : 100, "autumn" : 100}
    vmax = getattr(max_dict, season, 90)
    # vmax = max_dict[season]

    coarsen = getattr(info, "coarsen", 1)

    dataset = myDataset(season=season, coarsen=coarsen)

    # data = XArrayDataset(filepath=data_file)

    test_indices_file = getattr(info, 'test_indices', None)
    if test_indices_file:
        test_indices_file = project_root / info['test_indices']
        test_idx = test_indices(test_indices_file)
    else:
        _, _, test_idx = train_val_test_split_temp(dataset, seed=42, test_frac=0.1, val_frac=0.135, gen_new=True)


    test_data = TestSubset(dataset, test_idx)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True,)
    test_months = list(set([dataset.groups[i] for i in test_idx]))
    loss = 0

    mld_labels = np.zeros((len(test_months), len(dataset.argo_cut["latitude"]), len(dataset.argo_cut["longitude"])))
    mld_preds = np.zeros((len(test_months), len(dataset.argo_cut["latitude"]), len(dataset.argo_cut["longitude"])))

    for X, y, extra_info in test_dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        month_idx = int(extra_info[-1])
        month_pos = test_months.index(month_idx)
        lat = int((extra_info[0]).item() - dataset.argo_cut["latitude"].values.min())
        lon = int((extra_info[1]).item() - dataset.argo_cut["longitude"].values.min())
        time = extra_info[2]
        print(f"Processing month {month_idx}, lat {lat}, lon {lon}, time {time}")
        mld_labels[month_pos, lat, lon] = y.item() * (dataset.stds["mld"]) + dataset.means["mld"]
        mld_preds[month_pos, lat, lon] = preds.item() * (dataset.stds["mld"]) + dataset.means["mld"]
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
    fig, axs = plt.subplots(len(test_months), 3, figsize=(20, 7 * len(test_months)), constrained_layout=True)
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


    for i, month in enumerate(test_months):
        im_0 = axs[i, 0].imshow(mld_labels[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
        axs[i, 0].set_title(f"LABEL\nMonth: {month}", fontsize=24, pad=20)
        im_1 = axs[i, 1].imshow(mld_preds[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
        rmse = np.sqrt(np.mean((mld_labels[i] - mld_preds[i])**2))
        mae= np.mean(np.abs(mld_labels[i] - mld_preds[i]))
        total_rmse += rmse
        total_mae += mae
        axs[i, 1].set_title(f"PREDICTION\nMonth: {month}, RMSE: {rmse:.2f}, MAE: {mae:.2f}", fontsize=24, pad=20)
        axs[i, 2].imshow(mld_preds_smoothed[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
        axs[i, 2].set_title(f"Smoothed PREDICTION\nMonth: {month}", fontsize=24, pad=20)


        # fig.colorbar(axs[i, 2].imshow(mld_preds_smoothed[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower'), ax=axs[i, 2], orientation='vertical', fraction=0.02, pad=0.04)

    cbar = fig.colorbar(im_0, ax=axs, location='right', fraction=0.04, pad=0.1, label='MLD (m)')
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label='MLD (m)', fontsize=26)
    total_rmse = total_rmse / len(test_months)
    total_mae = total_mae / len(test_months)
    plt.suptitle(f"\n{model_name}\nSeason: {season}\nRMSE: {mass_rmse:.2f} MAE: {mass_mae:.2f} R2: {mass_r2:.2f}\n\n", fontsize=34,)
    update_values(info_path, {"rmse": mass_rmse, "mae": mass_mae, "r2": mass_r2})


    fig.savefig(model_path / f"rmse:{mass_rmse:.2f}_mae:{mass_mae:.2f}_r2:{mass_r2:.2f}.png", dpi=600)

    # for i, month in enumerate(test_months):
    #     im_0 = axs[i, 0].imshow(mld_labels[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
    #     # fig.colorbar(im_0, ax=axs[i, 0], orientation='vertical', fraction=0.02, pad=0.04)
    #     axs[i, 1].imshow(mld_preds_smoothed[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower')
    #     # fig.colorbar(axs[i, 1].imshow(mld_preds_smoothed[i], cmap='inferno', vmin=0, vmax=vmax, origin='lower'), ax=axs[i, 1], orientation='vertical', fraction=0.02, pad=0.04)
    #     axs[i, 0].set_xticks([])
    #     axs[i, 0].set_yticks([])
    #     axs[i, 0].set_xticklabels([])
    #     axs[i, 0].set_yticklabels([])
    #     axs[i, 1].set_xticks([])
    #     axs[i, 1].set_yticks([])
    #     axs[i, 1].set_xticklabels([])
    #     axs[i, 1].set_yticklabels([])
    #     axs[i, 0].axis('off')
    #     axs[i, 1].axis('off') 
    # plt.subplots_adjust(wspace=0.2)
    # plt.subplots_adjust(hspace=0.05)

    fig.savefig(model_path / f"rmse:{total_rmse:.2f}_poster.png", dpi=600)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Continue training a model.")
    parser.add_argument('--model_relative_path', type=str, default="saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250603_220037>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_01.pt>", help="Relative path to the model directory.")
    parser.add_argument('--recalculate', type=bool, default=True, help="recompute inference forward pass")
    args = parser.parse_args()
    main(args)