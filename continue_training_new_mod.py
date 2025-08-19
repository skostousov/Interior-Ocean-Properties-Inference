from begin_training import train_loop, val_loop, update_values, fetch_data_processor
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from pathlib import Path
from torch.utils.data import Subset, DataLoader
from utils.datasettemporal_new_mod import TemporalDatasetNewMod as TemporalDataset, TestSubsetRegressionNewMod as TestSubsetRegression
from utils.transforms import RescaledRotationTransform
from utils.config import PROJECT_ROOT
from torch.optim import AdamW
from utils.splitter import train_val_test_split_temp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys, importlib

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
    sys.modules.setdefault("numpy._core", importlib.import_module("numpy.core"))
    torch.backends.cudnn.benchmark = True

    project_root = Path(PROJECT_ROOT)
    model_relative_path = args.model_relative_path
    save_dir = project_root / model_relative_path

    info_path = save_dir / "training_info.txt"
    info = fetch_info(info_path)

    training_completed = info["training_completed"]
    if not training_completed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

        model = torch.load(save_dir/'model', map_location=device, weights_only=False)
        optimizer = AdamW(model.parameters(), lr=float(info['lr'] if hasattr(info, 'lr') else 1e-4), weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        checkpoint = torch.load(save_dir / 'checkpoint.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        loss_fn = nn.L1Loss()
        assert info['loss_fn'] == loss_fn.__class__.__name__, f"Loss function mismatch: {info['loss_fn']} != {loss_fn.__class__.__name__}"

        if info['transform']:
            data_aug = RescaledRotationTransform()
            assert data_aug.__class__.__name__ == info['transform'], f"Transform mismatch: {data_aug.__class__.__name__} != {info['transform']}"
        else:
            data_aug = None

        groupby = info["groupby"]
        lat_lon = info["lat_lon"]
        full = info["full"]
        rim = int(info.get("rim", 0))
        custom_features = eval(info.get("features", False))


        data = TemporalDataset(filepath=project_root / info['data_file'], transform=data_aug, season=info["season"], mld_res=float(info['mld_res']), feature_res=float(info['feature_res']), groupby=groupby, lat_lon=lat_lon, full=full, rim=rim, custom_features=custom_features)


        batch_size = int(info['batch_size']) 
        train_idx, val_idx, _ = train_val_test_split_temp(data, seed=42, test_indices_path=Path(info['test_indices']), gen_new=False)
        train_data, val_data = Subset(data, train_idx), TestSubsetRegression(data, val_idx)
        print(f"Train dataset size: {len(train_data)}, Val dataset size: {len(val_data)}")

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

        start_epoch = int(info['current_epoch']) + 1
        if args.num_epochs is None:
            num_epochs = int(info['total_epochs']) - start_epoch
        else:
            num_epochs = args.num_epochs

        end_epoch = start_epoch + num_epochs
        max_epochs = int(info['total_epochs'])
        best_epoch = int(info['best_epoch'])
        best_loss = float(info['best_val_loss'])
        early_stopping_thresh = int(info['early_stopping_thresh'])

        writer= SummaryWriter(save_dir / 'tensorboard_logs')

        for epoch in range(start_epoch, end_epoch):
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
                corresponding_train_loss = train_loss
                best_checkpoint = {
                    'best_model_state':model.state_dict(),
                    'best_optimizer_state': optimizer.state_dict(),
                    'best_scheduler_state': scheduler.state_dict(),
                    }
                torch.save(best_checkpoint, save_dir / 'best_checkpoint.pt')
                torch.save(model, save_dir / 'best_model')
                update_values(info_path, {"best_epoch": best_epoch, "best_val_loss": best_loss, "corresponding_train_loss": corresponding_train_loss})
                print("achieved best val loss")
            if epoch - best_epoch >= early_stopping_thresh:
                print(f"Early stopping on epoch {epoch}")
                update_values(info_path, {'training_completed': True})
                break
            elif epoch == max_epochs:
                print(f"finished all epochs")
                update_values(info_path, {'training_completed': True})
                break
        print(f"Best loss: {best_loss}")
        print(f"Training completed. Best model saved at {save_dir / 'best_model'}")
    else:
        print("Training already completed. Skipping training process.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Continue training a model.")
    parser.add_argument('--model_relative_path', type=str, default="saved_models/saved_daily_alternative_small_models/MODEL:UNetRegressionSE>TRAINSTART:20250603_220037>DATAFILE:small_daily_alternative_sample_1993-1993.nc>STRAT:test_indices_daily_alternative_small_small_01.pt>", help="Relative path to the model directory.")
    parser.add_argument('--num_epochs', type=int, default=5, help="number of epochs to train for")
    args = parser.parse_args()
    main(args)




