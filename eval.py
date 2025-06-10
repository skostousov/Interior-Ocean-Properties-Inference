from utils.config import RAW_CONFIG, PROJECT_ROOT, RELEVANT_CONFIG
from pathlib import Path
from utils.datasettemporal import TestSubsetRegression
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
from begin_training import fetch_data_processor


def main(args):
    project_root = Path(PROJECT_ROOT)
    model_relative_path = args.model_relative_path
    model_name = model_relative_path.split("/")[-1]
    model_path = project_root / model_relative_path

    info_path = model_path / "training_info.txt"
    info = fetch_info(info_path)
    data_file = project_root / info['data_file']
    test_indices_file = project_root / info['test_indices']

    loss_fn = nn.L1Loss()
    assert info['loss_fn'] == loss_fn.__class__.__name__, f"Loss function mismatch: {info['loss_fn']} != {loss_fn.__class__.__name__}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path/'best_model', map_location=device, weights_only=False)
    model.eval()

    print(f"Using data processor {info['data_processor']}")
    data = fetch_data_processor(info['data_processor'])(filepath=data_file)

    # data = XArrayDataset(filepath=data_file)
    test_idx = test_indices(test_indices_file)
    test_data = TestSubsetRegression(data, test_idx)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True,)
    
    loss = 0

    after_model = []

    filepath = model_path/"results.pkl"

    if args.recalculate or not os.path.exists(filepath):
        if os.path.exists(filepath):
            os.remove(filepath)
            print("previous file removed")
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

            if len(after_model) > 0:
                with open(filepath, "ab") as f:
                    pickle.dump(after_model, f)
            print(f"Total loss: {loss / len(test_dataloader)}")

    with open(info_path, 'a') as f:
        f.write(f"total_test_loss: {loss / len(test_dataloader)}\n")

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
    ax = np.atleast_2d(ax)
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
    fig.savefig(model_path / "results.png", dpi=600)

    fig, ax = plt.subplots(len(time_steps), 1, figsize=(6, 5* len(time_steps)))
    for i, t in enumerate(time_steps):
        if len(time_steps) == 1:
            ax = [ax]  # Ensure ax is iterable if there's only one time step
        pred_map = pred_maps[i]
        label_map = label_maps[i]    
        im_0 = ax[i].imshow(abs(label_map - pred_map), origin='lower', vmin=0, vmax=80, cmap='viridis')
        ax[i].set_title('Difference Map for Time Step ' + str(t))
        ax[i].set_xlabel('Longitude Index')
        ax[i].set_ylabel('Latitude Index')
        fig.colorbar(im_0, label='MLD MAE (m)', ax=ax[i])
    plt.suptitle(f"{model_name} \n Difference Results")
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    fig.savefig(model_path / "results_diff.png", dpi=600)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Continue training a model.")
    parser.add_argument('--model_relative_path', type=str, default="saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250603_220037>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_01.pt>", help="Relative path to the model directory.")
    parser.add_argument('--recalculate', type=bool, default=True, help="recompute inference forward pass")
    args = parser.parse_args()
    main(args)