import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from utils.datasettemporal import TemporalDataset, TestSubsetRegression

class GANDataset(TemporalDataset):
    def __init__(self, season="all", full_frames=True, sample_size=50, auto_specs=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_frames = full_frames
        self.season = season
        self.months = self.season_months.get(season, range(1, 13))
        self.data = self.feature_map
        self.target = self.annotations_map
        self.sample_size = sample_size
        if self.sample_size == "auto":
            self._auto_specs = auto_specs
            self.sample_size = min(self.data.shape[1], self.data.shape[2])//self._auto_specs
        if self.full_frames:
            self.indices = [i for i in range(len(self.data)) if (i % 12) + 1 in self.months]
            self.indices_corresponding_months = [(i%12)+1 for i in self.indices]
            self.groups = self.indices
        else:
            self.indices = [(i, (j, k)) for i in range(len(self.data)) 
                            for j in range(0, self.data.shape[1]-self.sample_size, self.sample_size) 
                            for k in range(0, self.data.shape[2]-self.sample_size, self.sample_size) 
                            if (i % 12) + 1 in self.months]
            self.indices_corresponding_months = [(i[0]%12)+1 for i in self.indices]
            self.groups = [i[0] for i in self.indices]

    def __len__(self):
       return len(self.indices)

    def __getitem__(self, idx):
        if self.full_frames:
            idx = self.indices[idx]
            X = self.data[idx]
            y = self.target[idx]
        else:
            i, (j, k) = self.indices[idx]
            X = self.data[i, j:j+self.sample_size, k:j+self.sample_size]
            y = self.target[i, j:j+self.sample_size, k:j+self.sample_size]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform:
            combined = torch.cat((X, y), dim=0)
            combined_transformed = self.transform(combined)
            X, y = combined_transformed[:X.shape[0]], combined_transformed[X.shape[0]:]

        if self.normalize:
            X = (X - self.mean) / self.std
            y = (y - self.mean_label) / self.std_label
        
        if self.full_frames:
            return X, y
        else:
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
        if not self.dataset.full_frames:
            i, (j, k) = self.dataset.indices[idx]
            ijk = (i, j, k)
            return X, y, ijk
        else:
            return X, y


if __name__ == "__main__":
    transform = None  # Define your transform here if needed
    dataset = GANDataset(filepath="data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc", transform=transform, normalize=True, season="autumn")
    print(f"Dataset length: {len(dataset)}")