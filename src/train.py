import torch
import sys
from torchvision.transforms import Compose
from torch.utils.data import Subset
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from utils.dataset import GLORYSDS
from utils.transforms import RescaledRotationTransform, ToTensor
from utils.config import DATA_DIR


def train_test_split(dataset, test_frac=0.2, seed=42):
    groups = dataset.index_region
    indexes = list(range(len(dataset)))
    gss = GroupShuffleSplit(test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(indexes, groups=groups))
    return(
        Subset(dataset, train_idx),
        Subset(dataset, test_idx)
    )



data_aug = Compose([ToTensor(), RescaledRotationTransform()])
ds_path = DATA_DIR/"cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_156.00W-121.42W_41.83N-63.08N_0.49m_2021-06-25-2021-06-30.nc"
data = GLORYSDS(ds_path, data_aug)
train_ds, test_ds = train_test_split(data)
print(len(train_ds), len(test_ds))



