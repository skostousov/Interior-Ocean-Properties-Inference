from torch.utils.data import Dataset, Subset
import xarray as xr
from utils.config import PROJECT_ROOT
from pathlib import Path
import numpy as np
import torch
from models.simple_CNN_regression import PixelWiseRegressor

class myDataset(Dataset):
    def __init__(self, transform=None, season=None, coarsen=False):
        self.coarsen = coarsen
        self.glorys, self.argo = self.fetch_argo_and_glorys()
        self.sla = self.fetch_sla()
        self.sst = self.fetch_sst()
        self.season= season
        season_months = {
                "winter": [12, 1, 2],
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11]
            }
        self.relevant_months = season_months.get(season, range(1, 13))
        self.ds_merged = xr.merge([self.glorys, self.argo, self.sla, self.sst])
        self.ds_renamed = self.ds_merged.rename({'zos': 'ssh_reanalysis', 'thetao': 'sst_reanalysis', 'so' : 'sss_reanalysis', 'mlotst': 'mld_reanalysis', 'sla': 'sla', 'sst': 'sst' })
        self.step_lat = (self.argo_cut["latitude"].values[-1] - self.argo_cut["latitude"].values[0]) // (self.argo_cut["latitude"].shape[0] - 1)
        self.step_lon = (self.argo_cut["longitude"].values[-1] - self.argo_cut["longitude"].values[0]) // (self.argo_cut["longitude"].shape[0] - 1)
        self.indexes = [(k_idx, i, j) for k_idx, _ in enumerate(self.ds_renamed["time"].values) for i in range(self.argo_cut["latitude"].shape[0]) for j in range(self.argo_cut["longitude"].shape[0])]
        self.indexes = [idx for idx in self.indexes if not np.isnan(self.argo_cut["mld"][idx[0]][idx[1]][idx[2]].values)]
        self.indexes = [idx for idx in self.indexes if idx[0] % 12 in self.relevant_months]
        self.groups = [idx[0] for idx in self.indexes]
        self.transform = transform
        self.full_groups = [(i%12)+1 for i in range(len(self.ds_renamed["time"].values)) if (i%12)+1 in self.relevant_months]
        self.full_indexes = [i for i in range(len(self.ds_renamed["time"].values)) if (i%12)+1 in self.relevant_months]
        # Normalize each variable in self.ds_renamed
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}
        for var in self.ds_renamed.data_vars:
            data = self.ds_renamed[var]
            self.means[var] = data.mean(skipna=True).item()
            self.stds[var] = data.std(skipna=True).item() if data.std(skipna=True).item() != 0 else 1.0
            self.mins[var] = data.min(skipna=True).item()
            self.maxs[var] = data.max(skipna=True).item()
            self.ds_renamed[var] = (data - self.means[var]) / self.stds[var]
            # self.ds_renamed[var] = (data - self.mins[var]) / (self.maxs[var] - self.mins[var])  # Normalize to [0, 1]
    def fetch_sst(self):
        sst = xr.open_dataset(Path(PROJECT_ROOT) / "data/argo/sst.mon.mean.nc")
        sst = sst.assign_coords(lon=(((sst.lon + 180) % 360) - 180))
        sst = sst.assign_coords(lat=sst.lat.astype("float64"), lon=sst.lon.astype("float64"))
        sst = sst.sel(time=slice("2005-01-01", "2020-12-31"))
        sst_cut = sst.where((sst["lat"] >= self.min_lat) & (sst["lat"] <= self.max_lat) & (sst["lon"] >= self.min_long) & (sst["lon"] <= self.max_long), drop=True)
        sst_cut = sst_cut.rename({'lat': 'latitude', 'lon': 'longitude'})
        sst_interpolated = sst_cut.interp(longitude=self.glorys_cut["longitude"], latitude=self.glorys_cut["latitude"], method='nearest', assume_sorted=True)
        return sst_interpolated
    def fetch_sla(self):
        sla = xr.open_dataset(Path(PROJECT_ROOT) / "data/argo/BCsla.nc")
        sla = sla.assign_coords(latitude=sla.latitude.astype("float64"), longitude=sla.longitude.astype("float64"))
        sla_aligned = sla.assign_coords(time=self.glorys_time)
        sla_cut = sla_aligned.where(
            (sla_aligned["latitude"] >= self.min_lat) & (sla_aligned["latitude"] <= self.max_lat) &
            (sla_aligned["longitude"] >= self.min_long) & (sla_aligned["longitude"] <= self.max_long),
            drop=True)
        sla_interpolated = sla_cut.interp(longitude=self.glorys_cut["longitude"], latitude=self.glorys_cut["latitude"], method='nearest', assume_sorted=True)
        return sla_interpolated
    def fetch_argo_and_glorys(self):
        glorys = xr.open_dataset(Path(PROJECT_ROOT) / "data/argo/BCGlorys.nc")
        argo = xr.open_dataset(Path(PROJECT_ROOT) / "data/argo/BCArgo.nc")

        glorys = glorys.isel(depth=0, drop=True)

        if self.coarsen:
            glorys = glorys.coarsen(latitude=self.coarsen, longitude=self.coarsen, boundary='trim').mean()

        self.glorys_time = glorys["time"]
        argo_aligned = argo.assign_coords(time = self.glorys_time)

        lon = argo_aligned["longitude"]
        lon180 = lon - 360
        argo_aligned = argo_aligned.assign_coords(longitude=lon180)

        glorys = glorys.assign_coords(
            latitude=glorys.latitude.astype("float64"),
            longitude=glorys.longitude.astype("float64")
        )
        self.min_long = max(glorys.longitude.min(), argo_aligned.longitude.min())
        self.max_long = min(glorys.longitude.max(), argo_aligned.longitude.max())
        self.min_lat = max(glorys.latitude.min(), argo_aligned.latitude.min())
        self.max_lat = min(glorys.latitude.max(), argo_aligned.latitude.max())

        self.argo_cut = argo_aligned.where(
            (argo_aligned["latitude"] >= self.min_lat) & (argo_aligned["latitude"] <= self.max_lat) &
            (argo_aligned["longitude"] >= self.min_long) & (argo_aligned["longitude"] <= self.max_long),
            drop=True)

        self.glorys_cut = glorys.where(
            (glorys["latitude"] >= self.min_lat) & (glorys["latitude"] <= self.max_lat) &
            (glorys["longitude"] >= self.min_long) & (glorys["longitude"] <= self.max_long),
            drop=True)
        argo_interpolated = self.argo_cut.interp(longitude=self.glorys_cut["longitude"], latitude=self.glorys_cut["latitude"], method='nearest', assume_sorted=True)
        return self.glorys_cut, argo_interpolated
    def return_full_dataset(self):
        X = np.stack([self.ds_renamed["sla"].values, 
                    self.ds_renamed["uo"].values, 
                    self.ds_renamed["vo"].values, 
                    # ds_square["ssh_reanalysis"].values, 
                    self.ds_renamed["sst"].values, 
                    self.ds_renamed["sss_reanalysis"].values, 
                    self.ds_renamed["bottomT"].values,], axis=-1)
        y = self.ds_renamed["mld"].values
        
        return X, y

    def __len__(self):
        return len(self.indexes)
    
    def prep_square(self, square):
        ds_square = square[0]
        lat = square[2]["lat"]
        lon = square[2]["lon"]
        time = square[2]["time"]
        time_idx = square[2]["time_idx"]
        mld = square[1]
        lat_grid, lon_grid = np.meshgrid(
            ds_square.latitude.values, ds_square.longitude.values, indexing='ij'
        )

        min_lat_grid = np.full(lat_grid.shape, self.min_lat)
        min_lon_grid = np.full(lon_grid.shape, self.min_long)
        max_lat_grid = np.full(lat_grid.shape, self.max_lat)
        max_lon_grid = np.full(lon_grid.shape, self.max_long)
        lat_grid = (lat_grid - min_lat_grid) / (max_lat_grid - min_lat_grid)
        lon_grid = (lon_grid - min_lon_grid) / (max_lon_grid - min_lon_grid)

        X = np.stack([ds_square["sla"].values, 
                    ds_square["uo"].values, 
                    ds_square["vo"].values, 
                    # ds_square["ssh_reanalysis"].values, 
                    ds_square["sst"].values, 
                    ds_square["sss_reanalysis"].values, 
                    ds_square["bottomT"].values, 
                    lat_grid, lon_grid], axis=-1)

        y = mld
        extra_info = [lat, lon, time, time_idx]

        X = X.transpose(2, 0, 1)  # Reshape to (H, W, C)
        shape = 13//self.coarsen if self.coarsen else 13
        if X.shape[1] != shape:
            X = np.pad(X, ((0, 0), (0, shape - X.shape[1]), (0, 0)), mode='constant', constant_values=np.nan)
        if X.shape[2] != shape:
            X = np.pad(X, ((0, 0), (0, 0), (0, shape - X.shape[2])), mode='constant', constant_values=np.nan)
        return X, y, extra_info

    def __getitem__(self, idx):
        k_idx, i, j = self.indexes[idx]
        k = self.ds_renamed["time"].values[k_idx]
        # argo_mld_norm = (self.argo_cut["mld"][k_idx][i][j] - self.mins["mld"]) / (self.maxs["mld"] - self.mins["mld"])
        argo_mld_norm = (self.argo_cut["mld"][k_idx][i][j] - self.means["mld"]) / self.stds["mld"]
        square = (self.ds_renamed.loc[dict(
                latitude=slice(self.argo_cut["latitude"].values[i]-self.step_lat/2, self.argo_cut["latitude"].values[i]+self.step_lat/2),
                longitude=slice(self.argo_cut["longitude"].values[j]-self.step_lon/2, self.argo_cut["longitude"].values[j]+self.step_lon/2),
                time=k)], argo_mld_norm.values, {"lat": float(self.argo_cut["latitude"].values[i]), "lon": float(self.argo_cut["longitude"].values[j]), "time": str(k), "time_idx": k_idx})
        X, y, extra_info = self.prep_square(square)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)
        if self.transform:
            X = self.transform(X)
        return X, y, extra_info
    
class TestSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        original_transform = self.dataset.transform
        self.dataset.transform = None  
        image, label, extra_info = self.dataset[original_idx]
        self.dataset.transform = original_transform
        return image, label, extra_info
    def __getitems__(self, indices: list[int]):
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.__getitem__(idx) for idx in indices]

if __name__ == "__main__":
    Dataset = myDataset(season="Summer")
    print(len(Dataset))
    X, y, extra_info = Dataset[43]
    print(X.max(), X.min(), y.max(), y.min())
    print(X.shape, y.shape, extra_info)
    model = PixelWiseRegressor(in_channels=X.shape[0], out_channels=1)
    pred = model(X.unsqueeze(0))  # Add batch dimension
    print(pred.shape)  # Should be (1, 1) for a single pixel prediction
    print(pred)