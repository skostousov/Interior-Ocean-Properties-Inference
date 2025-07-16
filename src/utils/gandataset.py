import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from utils.datasettemporal import TemporalDataset, TestSubsetRegression

class GANDataset(TemporalDataset):
    def __init__(self, season="all", groupby="days", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.season = season
        self.relevant_months = self.season_months.get(season, range(1, 13))
        self.data = self.feature_map
        self.target = self.annotations_map
        self.groupby = groupby
        if self.groupby not in ["days", "months", "years"]:
            raise ValueError("groupby must be one of 'days', 'months', or 'years'")
        self.dates = self.dataset.time.values
        self.groupby_map = {
            "days": [dt.astype('datetime64[D]').astype(int) for dt in self.dates],
            "months": [dt.astype('datetime64[M]').astype(int) for dt in self.dates],
            "years": [dt.astype('datetime64[Y]').astype(int) + 1970 for dt in self.dates]
        }
        self.groups = self.groupby_map[self.groupby]
        self.indices = range(len(self.data))
        assert len(self.indices) == len(self.groups), "Data and groups must have the same length"
        self.indices = [i for i in self.indices if (self.groupby_map["months"][i] % 12) + 1 in self.relevant_months]
        self.full_dates = [self.dates[i] for i in self.indices]
        self.groups = [self.groups[i] for i in self.indices]
    def __len__(self):
       return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        X = self.data[idx]
        y = self.target[idx]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform:
            combined = torch.cat((X, y), dim=0)
            combined_transformed = self.transform(combined)
            X, y = combined_transformed[:X.shape[0]], combined_transformed[X.shape[0]:]

        if self.normalize:
            X = (X - self.mean) / self.std
            y = (y - self.mean_label) / self.std_label
        
        return X, y

class TestSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset

    def __getitem__(self, idx):
        old_transform = self.dataset.transform
        self.dataset.transform = None
        X, y = super().__getitem__(idx)
        self.dataset.transform = old_transform
        return X, y


if __name__ == "__main__":
    transform = None  # Define your transform here if needed
    dataset = GANDataset(filepath="data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc", transform=transform, normalize=True, season="autumn", groupby="months")
    print(f"Dataset length: {len(dataset)}")
    print(dataset[0][0].shape, dataset[0][1].shape)