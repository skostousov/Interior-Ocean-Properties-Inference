from utils.config import RAW_CONFIG, PROJECT_ROOT, RELEVANT_CONFIG
from pathlib import Path
from utils.datasettemporal_new_mod import TestSubsetRegressionNewMod as TestSubsetRegression
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
from combo_new_mod import update_values, plot_grids, general_plot, plot_full
from utils.datasettemporal_new_mod import TemporalDatasetNewMod as TemporalDataset

def main(args):
    project_root = Path(PROJECT_ROOT)
    model_relative_path = args.model_relative_path
    model_name = model_relative_path.split("/")[-1]
    model_path = project_root / model_relative_path

    info_path = model_path / "training_info.txt"
    info = fetch_info(info_path)
    data_file = project_root / info['data_file']
    test_indices_file = project_root / info['test_indices']

    loss_dict = {'L1' : nn.L1Loss, 'MSE' : nn.MSELoss}
    loss_fn = loss_dict[info['loss_name']]()
    assert info['loss_fn'] == loss_fn.__class__.__name__, f"Loss function mismatch: {info['loss_fn']} != {loss_fn.__class__.__name__}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path/'best_model', map_location=device, weights_only=False)
    model.eval()
    season = info["season"]

    groupby=info["groupby"]
    lat_lon=info["lat_lon"]
    full=info["full"]
    num_to_plot = info["num_to_plot"] if info["num_to_plot"] is not None else None

    rim = int(info.get("rim", 0))


    mld_res = float(info["mld_res"])
    feature_res = float(info["feature_res"])

    data = TemporalDataset(filepath=data_file, mld_res=mld_res, feature_res=feature_res, season=season, groupby=groupby, lat_lon=lat_lon, full=full, rim=rim)

    test_idx = test_indices(test_indices_file)
    test_data = TestSubsetRegression(data, test_idx)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

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

    if full:
        mld_labels, mld_preds, test_temps = plot_full(test_dataloader, model, device)
    else:
        mld_labels, mld_preds, test_temps = plot_grids(test_dataloader, model, device)

    total_mae, total_rmse, r2 = general_plot(mld_labels, mld_preds, test_temps, season, model.name(), model_path, num_to_plot=num_to_plot)
    update_values(info_path, {'rmse': total_rmse, 'mae': total_mae, 'r2': r2})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Continue training a model.")
    parser.add_argument('--model_relative_path', type=str, default="saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250603_220037>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_01.pt>", help="Relative path to the model directory.")
    parser.add_argument('--recalculate', type=bool, default=True, help="recompute inference forward pass")
    parser.add_argument('--num_to_plot', type=int, default=None, help="Number of time steps to plot. If None, all time steps will be plotted.")
    args = parser.parse_args()
    main(args)