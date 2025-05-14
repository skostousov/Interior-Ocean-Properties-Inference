import torch
from torch import nn
import sys
from models.Unet import UNet
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from utils.dataset import GLORYSDS
from utils.transforms import RescaledRotationTransform, ToTensor, Compose
from utils.config import DATA_DIR, RAW_CONFIG, SAVED_MODELS_DIR
from torch.optim import Adam
import time

cfg = RAW_CONFIG

def train_test_split(dataset, test_frac=0.2, seed=42):
     groups = dataset.index_region
     print(len(groups))
     print(len(dataset))
     indexes = list(range(len(dataset)))
     gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
     train_idx, test_idx = next(gss.split(indexes, groups=groups))
     print(f"Max train_idx: {max(train_idx)}, Max test_idx: {max(test_idx)}, Dataset size: {len(dataset)}")
     assert max(train_idx) < len(dataset), f"Max train_idx ({max(train_idx)}) >= dataset size ({len(dataset)})"
     assert max(test_idx) < len(dataset), f"Max test_idx ({max(test_idx)}) >= dataset size ({len(dataset)})"
     return (
         Subset(dataset, train_idx),
         Subset(dataset, test_idx)
     )

def train_loop(model, train_dataloader, optimizer, loss_fn, device):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        if batch % 2 == 0:
            loss, current = loss.item(), batch * batch_size + len(images)
            print(f"loss: {loss:>7f} [{current:>6d}/{size:>5d}]")

def test_loop(test_dataloader, model, loss_fn, device):
    model.eval()
    size=len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for image, label in test_dataloader:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_aug = Compose([ToTensor(), RescaledRotationTransform()])
ds_path = DATA_DIR/"cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_156.00W-121.42W_41.83N-63.08N_0.49m_2021-06-25-2021-06-30.nc"
# data = GLORYSDS(ds_path, data_aug, days = False)
data = GLORYSDS(ds_path, days = False)

batch_size = RAW_CONFIG["batch_size"]
epochs = RAW_CONFIG["epochs"]

model = UNet(data[0][0].shape[0], data[0][1].shape[0])
model = model.to(device)
optimizer = Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
loss_fn = nn.MSELoss()

train_data, test_data = train_test_split(data)
print(f"Train dataset size: {len(train_data)}, Test dataset size: {len(test_data)}")
print(f"Train dataset shape: img: {train_data[0][0].shape}, lbl: {train_data[0][1].shape}, Test dataset shape: img: {test_data[0][0].shape}, lbl: {test_data[0][1].shape}")

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

best_loss = 1000000000000000000

for epoch in range(epochs):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        print('Epoch {}:'.format(epoch + 1))
        model.train(True)
        train_loop(model, train_dataloader, optimizer, loss_fn, device)
        test_loss = test_loop(test_dataloader, model, loss_fn, device)
        if test_loss < best_loss:
            best_loss = test_loss
            model_path = SAVED_MODELS_DIR/'model_{}_{}'.format(timestamp, epoch+1)

