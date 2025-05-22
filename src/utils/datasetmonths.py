from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from netCDF4 import Dataset as NETCDF4Dataset
import numpy as np
from pathlib import Path
import copernicusmarine
from utils.config import RAW_CONFIG, DATA_DIR, _REPO_ROOT
import torch
from sklearn.preprocessing import RobustScaler
cfg = RAW_CONFIG

class DatasetOverMonths(TorchDataset):
    def __init__(self, months=False, transform = None, target_transform = None, full_dataset=False, normalize=True):
        self.full_dataset = full_dataset
        self.config = cfg["monthly"]
        self.features = self.config['features']
        self.transform = transform
        self.target_transform = transform
        self.dataset = self._load(self._download())
        self._get_features_and_labels()
        self._grid_coords()

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
        n_months, n_channels, lat_size, lon_size = data.shape
        reshaped_data = data.reshape(n_months * lat_size * lon_size, n_channels)
        return reshaped_data
    
    def _download(self):
        if self.full_dataset:
            mode = 'large'
            download_dest = (DATA_DIR / self.config['output_dir'] / self.config['large']['output_file']).resolve()
        else:
            mode = 'small'
            download_dest = (DATA_DIR / self.config['output_dir'] / self.config['small']['output_file']).resolve()
        download_dest.parent.mkdir(parents=True, exist_ok=True)
        alt_path = (Path(_REPO_ROOT)/self.config['output_dir'] / download_dest.name).resolve()
        print(alt_path)
        if not alt_path.is_file():
            min_latitude = min(self.config['latitude_range'])
            max_latitude = max(self.config['latitude_range'])
            min_longitude = min(self.config['longitude_range'])
            max_longitude = max(self.config['longitude_range'])
            start_date = self.config[mode]['start_date'].isoformat() + "T00:00:00"
            end_date = self.config[mode]['end_date'].isoformat() + "T00:00:00"
            copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",
            dataset_version="202311",
            variables=self.features,
            minimum_longitude=min_longitude,
            maximum_longitude=max_longitude,
            minimum_latitude=min_latitude,
            maximum_latitude=max_latitude,
            start_datetime=start_date,
            end_datetime=end_date,
            output_directory=self.config['output_dir'],
            output_filename=self.config[mode]['output_file'],
            minimum_depth=0.49402499198913574,
            maximum_depth=0.49402499198913574,
            coordinates_selection_method="strict-inside",
            netcdf_compression_level=0,
            disable_progress_bar=False,)
        return alt_path
    def _load(self, path):
        dataset = NETCDF4Dataset(Path(path))
        return dataset
    def _get_features_and_labels(self):
        #extract relevant variables
        relevant_variables = {k:v[:] for (k, v) in self.dataset.variables.items() if k in self.features}
        # delete dimensional variables
        for key, _ in self.dataset.dimensions.items():
            if key in relevant_variables.keys():
                del relevant_variables[key]
        # reshape variables into (months, 1, long, lat)
        for key, value in relevant_variables.items():
            if len (relevant_variables[key].shape) != 4: relevant_variables[key] = np.expand_dims(value, 1)
        #creation of label_map
        self.annotations_map = relevant_variables.pop("mlotst")
        #creation of feature_map
        self.feature_map = np.concatenate(list(relevant_variables.values()), axis=1)
    def _grid_coords(self):
        grid_size = self.config['grid_size']
        lat_range = self.feature_map.shape[-2]
        lon_range = self.feature_map.shape[-1]
        grid_coords = [(i, j) for i in range(0, lat_range-grid_size) for j in range(0, lon_range-grid_size)]
        centre_coords = [(i+grid_size//2, j+grid_size//2) for i in range(0, lat_range-grid_size) for j in range(0, lon_range-grid_size)]
        assert len(grid_coords) == len(centre_coords), "Grid coordinates and centre coordinates do not match in length"
        self.grid_and_centre_coords = [(grid_coords[i], centre_coords[i]) for i in range(len(grid_coords))]
        self.grid_and_centre_coords_and_months = [(grid_coords[i], centre_coords[i], j) for i in range(len(grid_coords)) for j in range(self.feature_map.shape[0])]
        self.groups = [datapoint[-1] for datapoint in self.grid_and_centre_coords_and_months]
    def generate_mean_and_std(self, month_indices):
        X = self.feature_map[month_indices]
        mu = X.mean(axis=(0, 2, 3))
        std = X.std(axis=(0, 2, 3))
        return mu, std
    def generate_mean_and_std_partial(self, month_indices):
        n_pix = 0
        mean  = None
        M2    = None   # sum of squared diffs

        for m in month_indices:
            # feature_map[m] has shape (C, H, W)
            X = np.asarray(self.feature_map[m], dtype=np.float64)
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
    def generate_mean_and_std_labels(self, month_indices):
        n_pix = 0
        mean  = None
        M2    = None
        for m in month_indices:
            # feature_map[m] has shape (C, H, W)
            X = np.asarray(self.annotations_map[m], dtype=np.float64)
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
        grid_coords, centre_coords, month = self.grid_and_centre_coords_and_months[index]
        image = self.feature_map[month, :, grid_coords[0]:grid_coords[0]+self.config['grid_size'], grid_coords[1]:grid_coords[1]+self.config['grid_size']]
        label = self.annotations_map[month, :, centre_coords[0], centre_coords[1]]

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.normalize:
            image = (image - self.mean) / self.std
            label = (label - self.mean_label) / self.std_label

        if self.transform:
            image = self.transform(image)
            label = self.target_transform(label)

        return image, label
    
    def __len__(self):
        return len(self.grid_and_centre_coords_and_months)

if __name__ == "__main__":
    dataset = DatasetOverMonths()
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset shape: {dataset[0][0].shape}, label shape: {dataset[0][1].shape}")
    print(dataset.grid_and_centre_coords_and_months[1])
    print(dataset[1][1])
    print(dataset.grid_and_centre_coords_and_months[0])
    print(dataset[0][1])