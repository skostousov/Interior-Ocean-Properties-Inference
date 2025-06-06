from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
import numpy as np
from pathlib import Path
import copernicusmarine
from utils.config import RAW_CONFIG, RELEVANT_CONFIG, PROJECT_ROOT
import torch
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import xarray as xr

cfg = RELEVANT_CONFIG
project_root = PROJECT_ROOT

class XArrayDataset(TorchDataset):
    def __init__(self, transform = None, target_transform = None, normalize=True, filepath=None, downsample=False):
        self.cfg = cfg
        self.downsample = downsample
        self.features = cfg['data']['features']
        self.project_root = project_root
        self.transform = transform
        self.submode = cfg['submode']
        self.submode_cfg = cfg['data'][self.submode]
        self.target_transform = target_transform
        if filepath is not None:
            self.dataset = xr.open_dataset(filepath, mask_and_scale=False)
        else:
            self.dataset = xr.open_dataset(Path(self._download()), mask_and_scale=False)

        self.relevant_variables = {k:v[:] for (k, v) in self.dataset.variables.items() if k in self.features}
        # delete dimensional variables
        for key in self.dataset._coord_names:
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

        self.grid_size = self.cfg['data']['grid_size']
        if self.downsample:
            self.grid_size = self.grid_size / 3
            self.feature_map = F.avg_pool2d(torch.tensor(self.feature_map, dtype=torch.float32), kernel_size=3, stride=3).numpy()
            self.annotations_map = F.avg_pool2d(torch.tensor(self.annotations_map, dtype=torch.float32), kernel_size=3, stride=3).numpy()

        lat_range = self.feature_map.shape[-2]
        lon_range = self.feature_map.shape[-1]
        grid_coords = [(i, j) for i in range(0, lat_range-self.grid_size) for j in range(0, lon_range-self.grid_size)]
        centre_coords = [(i+self.grid_size//2, j+self.grid_size//2) for i in range(0, lat_range-self.grid_size) for j in range(0, lon_range-self.grid_size)]
        assert len(grid_coords) == len(centre_coords), "Grid coordinates and centre coordinates do not match in length"
        self.grid_and_centre_coords = [(grid_coords[i], centre_coords[i]) for i in range(len(grid_coords))]
        self.grid_and_centre_coords_and_temp_unit = [(grid_coords[i], centre_coords[i], j) for i in range(len(grid_coords)) for j in range(self.feature_map.shape[0])]
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
    def _inversescale(self, label):
        #inverse scale the label
        label = label.view(-1, 1)
        label = self.annotation_scaler.inverse_transform(label)
        return label

    def generate_mean_and_std(self, temp_unit_indices):
        X = self.feature_map[temp_unit_indices]
        mu = X.mean(axis=(0, 2, 3))
        std = X.std(axis=(0, 2, 3))
        return mu, std
    def generate_mean_and_std_partial(self, temp_unit_indices):
        n_pix = 0
        mean  = None
        M2    = None   # sum of squared diffs

        for t in temp_unit_indices:
            # feature_map[t] has shape (C, H, W)
            X = np.asarray(self.feature_map[t], dtype=np.float64)
            C, H, W = X.shape
            X = X.reshape(C, -1)               # now (C, H*W)
            
            batch_mean = X.mean(axis=1)        # (C,)
            batch_var  = X.var(axis=1)         # (C,)
            batch_n    = X.shape[1]            # H*W pixels per channel

            if mean is None:
                # first batch: init accumulators
                mean  = batch_mean
                M2    = batch_var * batch_n
                n_pix = batch_n
            else:
                # Welford update
                delta = batch_mean - mean
                total = n_pix + batch_n

                mean += delta * (batch_n / total)
                M2   += batch_var * batch_n + (delta**2) * (n_pix * batch_n / total)
                n_pix  = total

        std = np.sqrt(M2 / n_pix)
        return mean.astype(np.float32), std.astype(np.float32)
    def generate_mean_and_std_labels(self, temp_unit_indices):
        n_pix = 0
        mean  = None
        M2    = None
        for t in temp_unit_indices:
            # feature_map[t] has shape (C, H, W)
            X = np.asarray(self.annotations_map[t], dtype=np.float64)
            C, H, W = X.shape
            X = X.reshape(C, -1)               # now (C, H*W)
                
            batch_mean = X.mean(axis=1)
            batch_var  = X.var(axis=1)         # (C,)
            batch_n    = X.shape[1]            # H*W pixels per channel
            if mean is None:
                    # first batch: init accumulators
                    mean  = batch_mean
                    M2    = batch_var * batch_n
                    n_pix = batch_n
            else:
                    # Welford update
                    delta = batch_mean - mean
                    total = n_pix + batch_n

                    mean += delta * (batch_n / total)
                    M2   += batch_var * batch_n + (delta**2) * (n_pix * batch_n / total)
                    n_pix  = total
        std = np.sqrt(M2 / n_pix)
        return mean.astype(np.float32), std.astype(np.float32)
        
    def __getitem__(self, index):
        grid_coords, centre_coords, temp_unit = self.grid_and_centre_coords_and_temp_unit[index]
        image = self.feature_map[temp_unit, :, grid_coords[0]:grid_coords[0]+self.cfg['data']['grid_size'], grid_coords[1]:grid_coords[1]+self.cfg['data']['grid_size']]
        label = self.annotations_map[temp_unit, :, centre_coords[0], centre_coords[1]]

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

class TestSubsetRegression(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        grid_coords, centre_coords, temp_unit = self.dataset.grid_and_centre_coords_and_temp_unit[original_idx]
        image = self.dataset.feature_map[temp_unit, :, grid_coords[0]:grid_coords[0]+self.dataset.cfg['data']['grid_size'], grid_coords[1]:grid_coords[1]+self.dataset.cfg['data']['grid_size']]
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
    dataset = XArrayDataset()
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset shape: {dataset[0][0].shape}, label shape: {dataset[0][1].shape}")
    print(dataset.grid_and_centre_coords_and_temp_unit[1])
    print(dataset[1][1])
    print(dataset.grid_and_centre_coords_and_temp_unit[0])
    print(dataset[0][1])