from torch.utils.data import Dataset as TorchDataset
from netCDF4 import Dataset as NETCDF4Dataset
import numpy as np
from pathlib import Path
import yaml
import copernicusmarine
from utils.config import RAW_CONFIG, DATA_DIR

def download_dataset(username: str, output_dir: str, start_day: tuple, end_day: tuple, longitude: tuple, latitude: tuple):
    """
    Download dataset.

    Parameters:
        username (str): username to Copernicus Marine Service account
        output_dir (str): directory to download data to
        start_day (tuple): (year: int, month: int, day: int)
    """
    start_datetime = f"{start_day[0]}-{start_day[1]}-{start_day[2]}:T00:00:00"
    end_datetime = f"{end_day[0]}-{end_day[1]}-{end_day[2]}:T00:00:00"

    copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    username=username,
    dataset_version="202311",
    variables=["bottomT", "mlotst", "siconc", "sithick", "so", "thetao", "uo", "usi", "vo", "vsi", "zos"],
    minimum_longitude=min(longitude),
    maximum_longitude=max(longitude),
    minimum_latitude=min(latitude),
    maximum_latitude=max(latitude),
    output_directory= output_dir,
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    minimum_depth=0.49402499198913574,
    maximum_depth=0.49402499198913574,
    coordinates_selection_method="strict-inside",
    disable_progress_bar=False,
)

class GLORYSDS(TorchDataset):
    def __init__(self, dataset_dir, transform = None, grid_size = 40, days = True, normalize = True):
        self.days = days
        self.data = NETCDF4Dataset(dataset_dir)
        self.data_variables = {k:v[:] for (k, v) in self.data.variables.items()}
        for key, _ in self.data.dimensions.items():
            if key in self.data_variables.keys():
                del self.data_variables[key]

        self.transform = transform
        self.annotations_map = self.data_variables.pop("mlotst")

        for key, value in self.data_variables.items():
            if len (self.data_variables[key].shape) != 4: self.data_variables[key] = np.expand_dims(value, 1)
        self.feature_map = np.concatenate(list(self.data_variables.values()), axis=1)
        
        self.normalize = normalize
        self.feature_stats = {}
        self.label_stats = {}

        if normalize:
            for i in range(self.feature_map.shape[1]):
                channel = self.feature_map[:, i, :, :]
                self.feature_stats[i] = {
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel) + 1e-8)
                }
            self.label_stats = {
                'mean': float(np.mean(self.annotations_map)),
                'std': float(np.std(self.annotations_map) + 1e-8)
            }

        self.grid_size = grid_size
        self.offset_size = 2
        self.images_in_region_one_axis = 2
        self.region_size = grid_size + self.offset_size*(self.images_in_region_one_axis-1)
        self.regions = [
            (i, j)
            for i in range(0, self.annotations_map.shape[1], self.region_size)
            for j in range(0, self.annotations_map.shape[2], self.region_size)
        ]
        self.indices_regionified = {}
        for region in self.regions:
            self.indices_regionified[region] = [
                (i, j) 
                for i in range(region[0], region[0]+self.region_size, self.offset_size)
                for j in range(region[1], region[1]+self.region_size, self.offset_size)
                ]
        self.all_indices = []
        for region_indices in self.indices_regionified.values():
            self.all_indices.extend(region_indices)
        self.index_region = []
        for j, value in enumerate(self.indices_regionified.values()):
            self.index_region.extend([j for i in range(len(value))])

    def __len__(self):
        return len(self.all_indices)
    def __getitem__(self, idx):
        if idx >= len(self.all_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.all_indices)}")
        coordinates = self.all_indices[idx]

        max_x = min(coordinates[0] + self.grid_size, self.annotations_map.shape[1])
        max_y = min(coordinates[1] + self.grid_size, self.annotations_map.shape[2])


        image = self.feature_map[:, :, coordinates[0]:coordinates[0]+self.grid_size, coordinates[1]:coordinates[1]+self.grid_size]
        label = self.annotations_map[..., coordinates[0]:coordinates[0]+self.grid_size, coordinates[1]:coordinates[1]+self.grid_size]
        if max_x <= coordinates[0] or max_y <= coordinates[1]:
            padded_image = np.zeros((self.feature_map.shape[0], self.feature_map.shape[1], self.grid_size, self.grid_size))
            padded_label = np.zeros((self.annotations_map.shape[0], self.grid_size, self.grid_size))
            image = padded_image
            label = padded_label
        else:
            image = self.feature_map[:, :, coordinates[0]:max_x, coordinates[1]:max_y]
            label = self.annotations_map[..., coordinates[0]:max_x, coordinates[1]:max_y]
            if max_x - coordinates[0] < self.grid_size or max_y - coordinates[1] < self.grid_size:
                padded_image = np.zeros((image.shape[0], image.shape[1], self.grid_size, self.grid_size))
                padded_label = np.zeros((label.shape[0], self.grid_size, self.grid_size))
                padded_image[:, :, :(max_x-coordinates[0]), :(max_y-coordinates[1])] = image
                padded_label[:, :(max_x-coordinates[0]), :(max_y-coordinates[1])] = label
                
                image = padded_image
                label = padded_label
        
        if self.normalize:
            for i in range(image.shape[1]):
                mean = self.feature_stats[i]['mean']
                std = self.feature_stats[i]['std']
                image[:, i, :, :] = (image[:, i, :, :] - mean) / std
            label = (label - self.label_stats['mean']) / self.label_stats['std']

        label = np.expand_dims(label, axis=1)
        if self.transform:
            image, label = self.transform(image, label)
        if not self.days:
            image = np.squeeze(image[0:1], axis=0)
            label = np.squeeze(label[0:1], axis=0)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label
    
    def unormalize(self, predictions):
        if not self.normalize:
            return predictions
        mean = self.label_stats['mean']
        std = self.label_stats['std']
        return predictions*std + mean

if __name__ == "__main__":
    ds_name = "cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_156.00W-121.42W_41.83N-63.08N_0.49m_2021-06-25-2021-06-30.nc"
    ds = GLORYSDS(DATA_DIR/ds_name, days = False)
    sample_image, sample_label = ds[1]
    print(f"{sample_image.shape}, {sample_label.shape}")
    print(ds.data_variables)
        