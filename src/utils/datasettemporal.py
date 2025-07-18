from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from netCDF4 import Dataset as NETCDF4Dataset
import numpy as np
from pathlib import Path
# import copernicusmarine
from utils.config import RAW_CONFIG, RELEVANT_CONFIG, PROJECT_ROOT
import torch
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import xarray as xr

cfg = RELEVANT_CONFIG
project_root = PROJECT_ROOT

class TemporalDataset(TorchDataset):
    def __init__(self, transform = None, target_transform = None, normalize=True, filepath=None, downsample=False, grid_size = cfg['data']['grid_size'], season=None, data_processor="xarray", target_coarsen=None, feature_coarsen=None, lat_lon=True):
        self.cfg = cfg
        self.lat_lon=lat_lon
        self.target_coarsen = target_coarsen
        self.feature_coarsen = feature_coarsen
        self.data_processor = data_processor
        self.downsample = downsample
        self.features = cfg['data']['features']
        self.project_root = project_root
        self.transform = transform
        self.submode = cfg['submode']
        self.submode_cfg = cfg['data'][self.submode]
        self.target_transform = target_transform
        self.grid_size = int(grid_size)
        self.filepath = filepath
        self.season_months = {
                "winter": [12, 1, 2],
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11]
            }
        self.relevant_months = self.season_months.get(season, range(1, 13))
        if filepath is not None:
            if self.data_processor == "netcdf4":
                self.dataset = NETCDF4Dataset(filepath)
            elif self.data_processor == "xarray":
                self.dataset = xr.open_dataset(filepath)
            else:
                raise ValueError(f"Unsupported data processor: {self.data_processor}")

        else:
            if self.data_processor == "netcdf4":
                self.dataset = NETCDF4Dataset(Path(self._download()))
            elif self.data_processor == "xarray":
                self.dataset = xr.open_dataset(Path(self._download()))

        if self.data_processor == "xarray" and self.feature_coarsen is not None:
            self.dataset = self.dataset.coarsen(latitude=self.feature_coarsen, longitude=self.feature_coarsen, boundary="trim").mean()
            self.grid_size = self.grid_size // self.feature_coarsen
        
        if self.data_processor == "xarray" and self.target_coarsen is not None:
            mlotst_coarse = self.dataset["mlotst"].coarsen(latitude=self.target_coarsen, longitude=self.target_coarsen, boundary="pad").mean()
            self.dataset["mlotst"] = mlotst_coarse.interp_like(self.dataset["mlotst"], method="nearest")


        self.relevant_variables = {k:v[:] for (k, v) in self.dataset.variables.items() if k in self.features}
        # delete dimensional variables
        if self.data_processor == "netcdf4":
            for key, _ in self.dataset.dimensions.items():
                if key in self.relevant_variables.keys():
                    del self.relevant_variables[key]
        elif self.data_processor == "xarray":
            for key, _ in self.dataset.dims.items():
                if key in self.relevant_variables.keys():
                    del self.relevant_variables[key]

        # reshape variables into (months, 1, long, lat)
        for key, value in self.relevant_variables.items():
            if len (self.relevant_variables[key].shape) != 4: self.relevant_variables[key] = np.expand_dims(value, 1)

        self.all_variables = self.relevant_variables.copy()
        #creation of label_map
        self.annotations_map = self.relevant_variables.pop("mlotst")
        #creation of feature_map
        self.feature_map = np.concatenate(list(self.relevant_variables.values()), axis=1)

        lat_range = self.feature_map.shape[-2]
        lon_range = self.feature_map.shape[-1]

        if self.lat_lon: 
            lat_grid, lon_grid = np.meshgrid(range(lat_range), range(lon_range), indexing='ij')
            self.feature_map = np.concatenate((self.feature_map, np.repeat(lat_grid[np.newaxis, np.newaxis, ...], self.feature_map.shape[0], axis=0), np.repeat(lon_grid[np.newaxis, np.newaxis, ...], self.feature_map.shape[0], axis=0)), axis=1)

        grid_coords = [(i, j) for i in range(0, lat_range-self.grid_size) for j in range(0, lon_range-self.grid_size)]
        centre_coords = [(i+self.grid_size//2, j+self.grid_size//2) for i in range(0, lat_range-self.grid_size) for j in range(0, lon_range-self.grid_size)]
        assert len(grid_coords) == len(centre_coords), "Grid coordinates and centre coordinates do not match in length"
        self.grid_and_centre_coords = [(grid_coords[i], centre_coords[i]) for i in range(len(grid_coords))]
        self.grid_and_centre_coords_and_temp_unit = [(grid_coords[i], centre_coords[i], j) for i in range(len(grid_coords)) for j in range(self.feature_map.shape[0]) if j%12 in self.relevant_months]
        self.groups = [datapoint[-1] for datapoint in self.grid_and_centre_coords_and_temp_unit]

        self.normalize = normalize
        if self.normalize:
            self.feature_scaler = RobustScaler()
            self.annotation_scaler = RobustScaler()
            self.feature_scaler.fit(self._convert_to_scaler_format(self.feature_map))
            self.annotation_scaler.fit(self._convert_to_scaler_format(self.annotations_map))
            self.mean = torch.tensor(self.feature_scaler.center_, dtype=torch.float32)
            self.std = torch.tensor(self.feature_scaler.scale_, dtype=torch.float32)
            self.mean_label = torch.tensor(self.annotation_scaler.center_, dtype=torch.float32)
            self.std_label = torch.tensor(self.annotation_scaler.scale_, dtype=torch.float32)
            self.mean = self.mean.view(-1, 1, 1)
            self.std = self.std.view(-1, 1, 1)
            self.mean_label = self.mean_label.view(-1)
            self.std_label = self.std_label.view(-1)


        # self.monthly_features = np.split(self.feature_map, self.feature_map.shape[0], axis=0)
        # self.monthly_annotations = np.split(self.annotations_map, self.annotations_map.shape[0], axis=0)
    def _convert_to_scaler_format(self, data):
        #input data is a 4D array (months, channels, lat, lon)
        #convert to 2D array (months * lat * lon, channels)
        n_temp_units, n_channels, lat_size, lon_size = data.shape
        reshaped_data = data.reshape(n_temp_units * lat_size * lon_size, n_channels)
        return reshaped_data
    
    def _download(self):
        data_dir = Path(self.project_root) / self.cfg["data"]["data_dir"]
        print(self.submode)
        filename = self.submode_cfg['output_file']
        download_dest = (data_dir / filename).resolve()
        download_dest.parent.mkdir(parents=True, exist_ok=True)
        # alt_path = (Path(_REPO_ROOT)/self.config['output_dir'] / download_dest.name).resolve()
        # print(alt_path)
        print(download_dest)
        if not download_dest.is_file():
            self.data_cfg = self.cfg['data']
            min_latitude = min(self.data_cfg['latitude_range'])
            max_latitude = max(self.data_cfg['latitude_range'])
            min_longitude = min(self.data_cfg['longitude_range'])
            max_longitude = max(self.data_cfg['longitude_range'])
            start_date = self.submode_cfg['start_date'].isoformat() + "T00:00:00"
            end_date = self.submode_cfg['end_date'].isoformat() + "T00:00:00"
            copernicusmarine.subset(
            dataset_id=self.data_cfg['dataset_id'],
            dataset_version="202311",
            variables=self.features,
            minimum_longitude=min_longitude,
            maximum_longitude=max_longitude,
            minimum_latitude=min_latitude,
            maximum_latitude=max_latitude,
            start_datetime=start_date,
            end_datetime=end_date,
            output_directory=data_dir,
            output_filename=filename,
            minimum_depth=0.49402499198913574,
            maximum_depth=0.49402499198913574,
            coordinates_selection_method="strict-inside",
            netcdf_compression_level=0,
            disable_progress_bar=False,)
        return download_dest
        
    def __getitem__(self, index):
        grid_coords, centre_coords, temp_unit = self.grid_and_centre_coords_and_temp_unit[index]
        image = self.feature_map[temp_unit, :, grid_coords[0]:grid_coords[0]+self.grid_size, grid_coords[1]:grid_coords[1]+self.grid_size]
        # label = self.annotations_map[temp_unit, :, centre_coords[0], centre_coords[1]]
        label = self.annotations_map[temp_unit, :, grid_coords[0]:grid_coords[0]+self.grid_size, grid_coords[1]:grid_coords[1]+self.grid_size].mean(axis=(1, 2))  


        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.normalize:
            image = (image - self.mean) / self.std
            label = (label - self.mean_label) / self.std_label
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def __len__(self):
        return len(self.grid_and_centre_coords_and_temp_unit)

    def name(self):
        return self.data_processor

class TestSubsetRegression(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        grid_coords, centre_coords, temp_unit = self.dataset.grid_and_centre_coords_and_temp_unit[original_idx]
        image = self.dataset.feature_map[temp_unit, :, grid_coords[0]:grid_coords[0]+self.dataset.grid_size, grid_coords[1]:grid_coords[1]+self.dataset.grid_size]
        label = self.dataset.annotations_map[temp_unit, :, centre_coords[0], centre_coords[1]]

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.dataset.normalize:
            image = (image - self.dataset.mean) / self.dataset.std
            label = (label - self.dataset.mean_label) / self.dataset.std_label

        return image, label, (grid_coords, centre_coords, temp_unit)

    def __getitems__(self, indices: list[int]):
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.__getitem__(idx) for idx in indices]

if __name__ == "__main__":
    dataset = TemporalDataset(season="summer", data_processor="xarray", feature_coarsen=None, target_coarsen=None)
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset shape: {dataset[0][0].shape}, label shape: {dataset[0][1].shape}")
    print(dataset.grid_and_centre_coords_and_temp_unit[1])
    print(dataset[1][1])
    print(dataset.grid_and_centre_coords_and_temp_unit[100])
    print(dataset[100][1])