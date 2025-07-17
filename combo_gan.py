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

def get_pred_and_mld_tensors(G, test_dataloader, dataset, device):
    test_months = len(test_dataloader.dataset.indices)
    pred_map = np.zeros((test_months, dataset.feature_map.shape[-2], dataset.feature_map.shape[-1]))
    mld_map = np.zeros((test_months, dataset.feature_map.shape[-2], dataset.feature_map.shape[-1]))
    dataset.sample_size = dataset.sample_size
    for i, (X, y, extra_info) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        i, (j, k) = extra_info
        preds = G(X)
        pred_map[i, j:j+dataset.sample_size, k:k+dataset.sample_size] = preds[0][0].item() * (dataset.std_label) + dataset.mean_label
        mld_map[i, j:j+dataset.sample_size, k:k+dataset.sample_size] = y[0][0].item() * (dataset.std_label) + dataset.mean_label
    return pred_map, mld_map

def plot_pred_and_mld_maps(pred_map, mld_map, save_dir, model_name, vmax, season):
    mae_loss = nn.L1Loss()
    time_steps = pred_map.shape[0]
    fig, ax = plt.subplots(max(1, time_steps), 2, figsize=(15, 8* time_steps))
    ax = np.atleast_2d(ax)
    total_rmse = 0
    for i, t in enumerate(range(time_steps)):
        pred = pred_map[i]
        label = mld_map[i]
        im_0 = ax[i, 0].imshow(label, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        ax[i, 0].set_title('Target Map for Time Step ' + str(t))
        ax[i, 0].set_xlabel('Longitude Index')
        ax[i, 0].set_ylabel('Latitude Index')
        fig.colorbar(im_0, label='Actual MLD (m)', ax=ax[i, 0])
        im_1 = ax[i, 1].imshow(pred, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
        mae = mae_loss(torch.tensor(label), torch.tensor(pred)).item()
        rmse = np.sqrt(np.mean((pred - label)**2))
        total_rmse += rmse
        ax[i, 1].set_title('Prediction Map for Time Step ' + str(t) + " MAE: " + f"{mae:.7f}" + " RMSE: " + f"{rmse:.2f}")
        ax[i, 1].set_xlabel('Longitude Index')
        ax[i, 1].set_ylabel('Latitude Index')
        fig.colorbar(im_1, label="Predicted MLD (m)", ax=ax[i, 1])
    total_rmse = total_rmse / len(time_steps)
    plt.suptitle(f"Season: {season}\n{model_name} \n \n RMSE: f{total_rmse:.2f}")
    plt.subplots_adjust(hspace=0.5)
    fig.savefig(save_dir / "results.png", dpi=600)

def plot_from_test_dataloader(G, test_dataloader, dataset, device, vmax, season, save_dir, model_name):
    test_temp_units = len(test_dataloader.dataset.indices)
    fig, ax = plt.subplots(nrows=test_temp_units, ncols=2, figsize=(12, 4 * test_temp_units))
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
        ax[i, 0].set_title("Real MLD | Date: {} | Season: {} | Model: {}".format(dataset.full_dates[real_idx], season, model_name))
        ax[i, 1].imshow(fake_y[0][0], cmap='viridis', vmin=vmin, vmax=vmax)
        ax[i, 1].set_title("Generated MLD | RMSE: {:.2f}, MAE: {:.2f}".format(rmse, mae))

    total_rmse /= len(test_dataloader)
    total_mae /= len(test_dataloader)
    fig.suptitle(f"Test Results | Average RMSE: {total_rmse:.2f}, Average MAE: {total_mae:.2f}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    plt.savefig(save_dir / "results.png", dpi=300)
    plt.show()



def train_loop(G, D, train_dataloader, opt_G, opt_D, device):
    λ = 50
    size = len(train_dataloader)
    G.train()
    D.train()
    total_loss_G = 0
    total_loss_D = 0
    total_fake_loss_D = 0
    total_real_loss_D = 0
    normal_loss_G = 0  # Track the normal loss for G
    for i, (X, y) in enumerate(train_dataloader):
        X, y = X.float().to(device), y.float().to(device)
        fake_y = G(X)

        opt_D.zero_grad()           
        D_real = D(X, y)
        D_fake = D(X, fake_y.detach())

        D_real_loss = F.mse_loss(D_real, torch.ones_like(D_real))
        D_fake_loss = F.mse_loss(D_fake, torch.zeros_like(D_fake))

        loss_D = 0.5 * (D_real_loss + D_fake_loss)
        loss_D.backward()
        opt_D.step() 

        opt_G.zero_grad()
        D_pred = D(X, fake_y)

        adv_loss = F.mse_loss(D_pred, torch.ones_like(D_pred))#How good is G at fooling D?
        l1_loss = F.mse_loss(fake_y, y)#`How close is G's prediction to the real value?
        normal_loss_G += l1_loss.item()  # Track the normal loss for G
        loss_G = adv_loss + λ * l1_loss# Loss function for G combines ability to fool D and closeness to the real value
        
        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()
        total_real_loss_D += D_real_loss.item()
        total_fake_loss_D += D_fake_loss.item()

        loss_G.backward()
        opt_G.step()
    
    total_loss_G /= size
    total_loss_D /= size
    total_real_loss_D /= size
    total_fake_loss_D /= size
    normal_loss_G /= size
    loss_dict = {
        'G_loss': total_loss_G,
        'D_loss': total_loss_D,
        'D_real_loss': total_real_loss_D,
        'D_fake_loss': total_fake_loss_D,
        'normal_loss_G': normal_loss_G
    }
    return loss_dict

def val_loop(G, val_dataloader, device):
    G.eval()
    total_val_loss_G = 0
    for i, (X, y) in enumerate(val_dataloader):
        X, y = X.float().to(device), y.float().to(device)
        with torch.no_grad():
            fake_y = G(X)
            l1_loss = F.mse_loss(fake_y, y)
            loss_G_val = l1_loss
            total_val_loss_G += loss_G_val.item()
    total_val_loss_G /= len(val_dataloader)
    return total_val_loss_G


def main(args):

    cfg = RELEVANT_CONFIG
    root = Path(PROJECT_ROOT)
    submode = RELEVANT_CONFIG["submode"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    season = args.season
    filepath = args.filepath
    batch_size = args.batch_size
    G_lr = args.G_lr
    D_lr = args.D_lr
    groupby = args.groupby

    # data_aug = RescaledRotationTransform(scaling_interval=(1, 1.2), degree_range=0)
    dataset = GANDataset(filepath = filepath, normalize=True, season=season, groupby=groupby)
    sample_image = dataset[0][0]
    dataset_aug = GANTransform(size=sample_image[0][0].shape)
    dataset.transform = dataset_aug
    epochs = 1000
    
    D = PatchDiscriminatorConditional(in_channels=dataset[0][0].shape[0] + dataset[0][1].shape[0]).to(device)
    G = GeneratorUNetRegressionSEConditional(in_channels=dataset[0][0].shape[0], out_channels=1).to(device)

    def initialize_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    D.apply(initialize_weights)
    G.apply(initialize_weights)

    D = D.to(device)
    G = G.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=G_lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=D_lr, betas=(0.5, 0.999))

    # test_indices_path = Path((cfg['data'][submode]["test_indices"]).replace('.pt', f'{str(season)}'+'.pt'))
    test_indices_path = Path((cfg['data'][submode]["test_indices"]).replace('.pt', f'{dataset.filepath.split("/")[-1].replace(".nc", "")}+_{str(season)}_{groupby}'+'.pt'))
    # train_idx, val_idx, test_idx = train_val_test_split_temp(data, seed=42, test_indices_path=test_indices_path, gen_new=True)
    train_idx, val_idx, test_idx = train_val_test_split_temp(dataset, seed=42, test_frac=0.1, val_frac=0.135, gen_new=True)

    train_data, val_data, test_data = Subset(dataset, train_idx), TestSubset(dataset, val_idx), TestSubset(dataset, test_idx)

    print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")
    print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Val dataset shape: img: {val_data[0][0].shape}, lbl: {val_data[0][1].shape}")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

    best_loss = float('inf')
    G_train_loss = float('inf')
    D_train_loss = float('inf')
    best_epoch=0

    start_timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_name = f"SEASON:{season}>G:{G.name()}>D:{D.name()}>TRAINSTART:{start_timestamp}>DATAFILE:{str(filepath).split('/')[-1].replace('.nc', '')}>"
    model_dir = 'gan_models'
    save_dir = root / model_dir / model_name
    os.makedirs(save_dir, exist_ok=True)

    writer= SummaryWriter(save_dir / 'tensorboard_logs')
    info_path =  save_dir / 'training_info.txt'
    info_bef = {
        "start_time": start_timestamp,
        "data_file": str(filepath),
        "test_indices": f"{test_indices_path}",
        "total_epochs": epochs,
        "batch_size": batch_size,
        "G": G.__repr__().replace("\n", r"\n"),
        "D": D.__repr__().replace("\n", r"\n"),
        "train_dataset_size": len(train_data),
        "val_dataset_size": len(val_data),
        "test_dataset_size": len(test_idx),
        "transform": dataset.transform.__class__.__name__ if dataset.transform else None,
        "target_transform": dataset.target_transform.__class__.__name__ if dataset.target_transform else None,
        "current_epoch": 0,
        "training_completed": False,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "corresponding_G_train_loss": G_train_loss,
        "corresponding_D_train_loss": D_train_loss,
        "G_lr": G_lr,
        "D_lr": D_lr,
        "season": season,
        "groupby": args.groupby,
        }

    with open(info_path, 'w') as f:
        for key, value in info_bef.items():
            f.write(f"{key}:: {value}\n")

    num_epochs = args.num_epochs

    if num_epochs is None:
        num_epochs = epochs

    checkpoint = {
                'G_state':G.state_dict(),
                'D_state':D.state_dict(),
                'G_opt_state': opt_G.state_dict(),
                'D_opt_state': opt_D.state_dict(),
                }
    best_checkpoint = {
                'best_G_state':G.state_dict(),
                'best_D_state':D.state_dict(),
                'best_G_opt_state': opt_G.state_dict(),
                'best_D_opt_state': opt_D.state_dict(),
                }
    torch.save(checkpoint, save_dir / 'checkpoint.pt')
    torch.save(best_checkpoint, save_dir / 'best_checkpoint.pt')
    torch.save(G, save_dir / 'best_G_model.pt')
    torch.save(D, save_dir / 'best_D_model.pt')
    torch.save(G, save_dir / 'model_G.pt')
    torch.save(D, save_dir / 'model_D.pt')

    # for epoch in range(0, epochs):
    for epoch in range(0, num_epochs):
        # print('Epoch {}:'.format(epoch + 1))
        loss_dict = train_loop(G, D, train_dataloader, opt_G, opt_D, device)
        total_loss_G = loss_dict['G_loss']
        total_loss_D = loss_dict['D_loss']
        total_real_loss_D = loss_dict['D_real_loss']
        total_fake_loss_D = loss_dict['D_fake_loss']
        normal_loss_G= loss_dict["normal_loss_G"]
        val_loss = val_loop(G, val_dataloader, device)
        writer.add_scalars('Loss', {'val': val_loss, 'train_G': loss_dict['G_loss'], 'train_D': loss_dict['D_loss']}, epoch)
        update_values(info_path, {'current_epoch': epoch})
        checkpoint = {
                    'G_state':G.state_dict(),
                    'D_state':D.state_dict(),
                    'G_opt_state': opt_G.state_dict(),
                    'D_opt_state': opt_D.state_dict(),
                    }
        torch.save(checkpoint, save_dir / 'checkpoint.pt')
        torch.save(G, save_dir / 'model_G.pt')
        torch.save(D, save_dir / 'model_D.pt')
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            print(f"New best validation loss: {best_loss:.4f} | Saving models...")
            G_train_loss = loss_dict['G_loss']
            D_train_loss = loss_dict['D_loss']
            best_checkpoint = {
                'best_G_state':G.state_dict(),
                'best_D_state':D.state_dict(),
                'best_G_opt_state': opt_G.state_dict(),
                'best_D_opt_state': opt_D.state_dict(),
                }
            torch.save(best_checkpoint, save_dir / 'best_checkpoint.pt')
            torch.save(G, save_dir / 'best_G_model.pt')
            torch.save(D, save_dir / 'best_D_model.pt')
        update_values(info_path, {"best_epoch": best_epoch, "best_val_loss": best_loss, "corresponding_G_train_loss": G_train_loss, "corresponding_D_train_loss": D_train_loss})
        print(f"END OF EPOCH {epoch+1} \n| Average_D_loss: {total_loss_D:.4f} (Average_D_real: {total_real_loss_D:.4f}, Average_D_fake: {total_fake_loss_D:.4f}) |\n| Average_G_loss: {total_loss_G:.4f} | Normal_G_loss: {normal_loss_G:.4f} | Average_G_val_loss: {val_loss:.4f} |\n| BEST VAL LOSS: {best_loss:.4f}\n")


    update_values(info_path, {'training_completed': True})
    print(f"Best loss: {best_loss}")
    print(f"Training completed. Best model saved at {save_dir / 'best_model'}")

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
    plot_from_test_dataloader(G, test_dataloader, dataset, device, vmax, season, save_dir, model_name)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model on 1/12-degree mld data")
    parser.add_argument('--num_epochs', default = 1000, type=int, help="number of epochs to train for")
    parser.add_argument('--G_lr', default = 1e-4, type=float)
    parser.add_argument('--D_lr', default = 1e-6, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--season', default='all', type=str)
    parser.add_argument('--filepath', default='data/WaterOnlyMonthlySmall/WaterOnlyMonthlyExtendedSeasonalitySmall.nc', type=str, help="Path to the dataset file")
    parser.add_argument('--groupby', default='months', type=str, choices=['days', 'months', 'years'], help="Group by days, months, or years")
    args = parser.parse_args()
    main(args)