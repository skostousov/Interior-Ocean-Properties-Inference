from torch.utils.data import Dataset as TorchDataset
from netCDF4 import Dataset as NETCDF4Dataset
import numpy as np
from pathlib import Path
import yaml
import copernicusmarine

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
    def __init__(self, dataset_dir, transform = None, target_transform=None, grid_size = 40):
        self.data = NETCDF4Dataset(dataset_dir)
        self.data_variables = {k:v[:] for (k, v) in self.data.variables.items()}
        for key, _ in self.data.dimensions.items():
            if key in self.data_variables.keys():
                del self.data_variables[key]
        self.transform = transform
        self.target_transform = target_transform
        self.annotations_map = self.data_variables.pop("mlotst")
        for key, value in self.data_variables.items():
            if len (self.data_variables[key].shape) != 4: self.data_variables[key] = np.expand_dims(value, 1)
        self.feature_map = np.concatenate(list(self.data_variables.values()), axis=1)

        self.grid_size = grid_size
        self.offset_size = 2
        self.indices = [
            (i, j)
            for i in range(0, self.annotations_map.shape[1], self.offset_size)
            for j in range(0, self.annotations_map.shape[2], self.offset_size)
        ]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        coordinates = self.indices[idx]
        image = self.feature_map[:, :, coordinates[0]:coordinates[0]+self.grid_size, coordinates[1]:coordinates[1]+self.grid_size]
        label = self.annotations_map[..., coordinates[0]:coordinates[0]+self.grid_size, coordinates[1]:coordinates[1]+self.grid_size]
        return image, label

if __name__ == "__main__":
    ds_name = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m_multi-vars_164.33W-122.08W_35.58N-62.75N_2025-05-01-2025-05-04.nc"
    repo_root = Path.cwd()
    config_path = repo_root / "configs" / "config.yaml"
    with open(config_path, 'r') as config_file:
        config_file = yaml.safe_load(config_file)
    ds_dir = (repo_root / config_file["data_dir_relative_to_project_root"]/ds_name).resolve()
    ds = GLORYSDS(ds_dir, grid_size=20)
    print(ds.indices)
    sample_image, sample_label = ds[1]
    print(f"{sample_image.shape}, {sample_label.shape}")
        