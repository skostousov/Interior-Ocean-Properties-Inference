import torch
import sys
from models.Unet import UNet
from torchvision.transforms import Compose
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from utils.dataset import GLORYSDS
from utils.transforms import RescaledRotationTransform, ToTensor
from utils.config import DATA_DIR, RAW_CONFIG
from torch.optim import Adam

cfg = RAW_CONFIG

def train_test_split(dataset, test_frac=0.2, seed=42):
    groups = dataset.index_region
    indexes = list(range(len(dataset)))
    gss = GroupShuffleSplit(test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(indexes, groups=groups))
    return(
        Subset(dataset, train_idx),
        Subset(dataset, test_idx)
    )

def train_one_epoch(model, train_dataloader, optimizer, loss_fn, device):
    running_loss = 0
    last_loss = 0

    model.train()
    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(' batch {} loss: {}'.format(i+1, last_loss))
            running_loss = 0.
    return last_loss

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_aug = Compose([ToTensor(), RescaledRotationTransform()])
ds_path = DATA_DIR/"cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_156.00W-121.42W_41.83N-63.08N_0.49m_2021-06-25-2021-06-30.nc"
data = GLORYSDS(ds_path, data_aug)
train_ds, test_ds = train_test_split(data)
print(len(train_ds), len(test_ds))


batch_size = RAW_CONFIG["batch_size"]
epochs = RAW_CONFIG["epochs"]

model = UNet(data[0][0].shape[1], data[0][1].shape[1])
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    print('Epoch {}:'.format(epoch + 1))
    model.train(True)
    avg_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
    running_val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (val_images, val_labels) in enumerate(val_dataloader):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = loss_fn(val_outputs, val_labels)
            running_val_loss += val_loss
    avg_val_loss = running_val_loss / (i + 1)
    print("Loss train {} validation {}".format(avg_loss, avg_val_loss))

    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict()
    #     'optimizer_state_dict': optimizer.state_dict()
    #     'loss'
    # })

    if avg_val_loss < best_vloss:
        best_vloss = avg_val_loss
        model_path = 'model_{}_{}'.format(timestamp, epoch+1)




