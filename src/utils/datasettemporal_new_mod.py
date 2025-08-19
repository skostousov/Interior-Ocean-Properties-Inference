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

class TemporalDatasetNewMod(TorchDataset):
    def __init__(self, transform = None, target_transform = None, normalize=True, filepath=None, grid_size = cfg['data']['grid_size'], season=None, mld_res=1/12, feature_res=1/12, weird_param_not_sure_what_it_does = True, include_mld_in_input = False, groupby="days", lat_lon=True, full=False, rim=0, custom_features=False):
        self.cfg = cfg
        self.full = full
        self.rim = rim
        self.groupby = groupby
        self.lat_lon = lat_lon
        self.mld_res = mld_res
        self.feature_res = feature_res
        self.target_coarsen = int(self.mld_res / self.feature_res)
        self.feature_coarsen = int(12*self.feature_res)

        if custom_features:
            self.features = custom_features
        else:
            self.features = cfg['data']['features']
        self.project_root = project_root
        self.transform = transform
        self.submode = cfg['submode']
        self.submode_cfg = cfg['data'][self.submode]
        self.target_transform = target_transform
        self.filepath=filepath
        if weird_param_not_sure_what_it_does:
            self.grid_size = int(self.mld_res / self.feature_res)
            # self.grid_size = self.grid_size // self.feature_coarsen

        else:
            self.grid_size = int(grid_size)        
        self.season_months = {
                "winter": [12, 1, 2],
                "spring": [3, 4, 5],
                "summer": [6, 7, 8],
                "autumn": [9, 10, 11]
            }
        self.relevant_months = self.season_months.get(season, range(1, 13))
        if filepath is not None:
            self.dataset = xr.open_dataset(filepath)
        else:
            self.dataset = xr.open_dataset(Path(self._download()))

        self.dataset = self.dataset.coarsen(latitude=self.feature_coarsen, longitude=self.feature_coarsen, boundary="pad").mean()        
        mlotst_coarse = self.dataset["mlotst"].coarsen(latitude=self.target_coarsen, longitude=self.target_coarsen, boundary="pad").mean()
        self.dataset["mlotst_2"] = mlotst_coarse.interp_like(self.dataset["mlotst"], method="nearest")

        self.relevant_variables = {k:v[:] for (k, v) in self.dataset.variables.items() if k in self.features + ["mlotst_2"] }
        # delete dimensional variables
        for key, _ in self.dataset.dims.items():
            if key in self.relevant_variables.keys():
                del self.relevant_variables[key]

        # reshape variables into (months, 1, long, lat)
        for key, value in self.relevant_variables.items():
            if len (self.relevant_variables[key].shape) != 4: self.relevant_variables[key] = np.expand_dims(value, 1)

        self.all_variables = self.relevant_variables.copy()
        #creation of label_map
        self.annotations_map = self.relevant_variables.pop("mlotst_2")
        if not include_mld_in_input:
            self.relevant_variables.pop("mlotst")
        #creation of feature_map

        self.lat_grid, self.lon_grid = np.meshgrid(self.dataset.latitude, self.dataset.longitude, indexing='ij')
        self.lat_grid = np.expand_dims(self.lat_grid, axis=(0, 1))
        self.lon_grid = np.expand_dims(self.lon_grid, axis=(0, 1))
        self.lat_grid = np.repeat(self.lat_grid, len(self.dataset.time.values), axis=0)
        self.lon_grid = np.repeat(self.lon_grid, len(self.dataset.time.values), axis=0)
        print(self.lat_grid.shape, self.lon_grid.shape)
        self.feature_map = np.concatenate(list(self.relevant_variables.values()), axis=1)
        print(self.feature_map.shape)
        if self.lat_lon:
            self.feature_map = np.concatenate((self.feature_map, self.lat_grid, self.lon_grid), axis=1)


        lat_range = self.feature_map.shape[-2]
        lon_range = self.feature_map.shape[-1]

        print(lat_range)
        print(lon_range)
        print(self.grid_size)

        lat_list = range(self.rim, lat_range-lat_range%(self.grid_size), self.grid_size)
        lon_list = range(self.rim, lon_range-lon_range%(self.grid_size), self.grid_size)

        # lat_list = range(self.rim, (lat_range)-(lat_range)%self.grid_size, self.grid_size)
        # lon_list = range(self.rim, (lon_range)-(lon_range)%self.grid_size, self.grid_size)

        lat_list_list = [i for i in lat_list]
        lon_list_list = [j for j in lon_list]
        print(f"Lat list: {max(lat_list_list), min(lat_list_list)}  Lon list: {max(lon_list_list), min(lon_list_list)}")
        print(f"Lat range: {lat_range}, Lon range: {lon_range}")

        # self.feature_map = F.pad(self.feature_map, (self.rim, self.rim, self.rim, self.rim), mode='replicate')
        # self.annotations_map = F.pad(self.annotations_map, (self.rim, self.rim, self.rim, self.rim), mode='replicate')

        self.feature_map = np.pad(self.feature_map, ((0, 0), (0, 0), (self.rim, self.rim), (self.rim, self.rim)), mode='edge')
        self.annotations_map = np.pad(self.annotations_map, ((0, 0), (0, 0), (self.rim, self.rim), (self.rim, self.rim)), mode='edge')

        # lat_list = range(0, lat_range, self.grid_size)
        # lon_list = range(0, lon_range, self.grid_size)
        print(len(lat_list), len(lon_list))
        print(lat_list)
        print(lon_list)
        print(self.feature_map.shape)

        if self.groupby not in ["days", "months", "years"]:
            raise ValueError("groupby must be one of 'days', 'months', or 'years'")
        self.dates = self.dataset.time.values
        self.groupby_map = {
            "days": [dt.astype('datetime64[D]').astype(int) for dt in self.dates],
            "months": [dt.astype('datetime64[M]').astype(int) for dt in self.dates],
            "years": [dt.astype('datetime64[Y]').astype(int) + 1970 for dt in self.dates]
        }
        self.groups = self.groupby_map[self.groupby]

        # lat_list = lat_list[self.rim:-self.rim]
        # lon_list = lon_list[self.rim:-self.rim]

        if not self.full:
            grid_coords = [(i, j) for i in lat_list for j in lon_list]
            centre_coords = [(i+self.grid_size//2, j+self.grid_size//2) for i in lat_list for j in lon_list]
            assert len(grid_coords) == len(centre_coords), "Grid coordinates and centre coordinates do not match in length"
            self.grid_and_centre_coords = [(grid_coords[i], centre_coords[i]) for i in range(len(grid_coords))]
            self.grid_and_centre_coords_and_temp_unit_full = [(grid_coords[i], centre_coords[i], j) for i in range(len(grid_coords)) for j in range(self.feature_map.shape[0])]
            self.groups = [self.groups[i[-1]] for i in self.grid_and_centre_coords_and_temp_unit_full]
            self.grid_and_centre_coords_and_temp_unit = [datapoint for datapoint in self.grid_and_centre_coords_and_temp_unit_full if (datapoint[2]%12)+1 in self.relevant_months]
            self.groups = [self.groups[i[-1]] for i in self.grid_and_centre_coords_and_temp_unit]
            self.indices = range(len(self.grid_and_centre_coords_and_temp_unit))
        else:
            self.indices = range(len(self.feature_map))
            self.groups = self.groups

        

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

    # def generate_mean_and_std(self, temp_unit_indices):
    #     X = self.feature_map[temp_unit_indices]
    #     mu = X.mean(axis=(0, 2, 3))
    #     std = X.std(axis=(0, 2, 3))
    #     return mu, std
    # def generate_mean_and_std_partial(self, temp_unit_indices):
    #     n_pix = 0
    #     mean  = None
    #     M2    = None   # sum of squared diffs

    #     for t in temp_unit_indices:
    #         # feature_map[t] has shape (C, H, W)
    #         X = np.asarray(self.feature_map[t], dtype=np.float64)
    #         C, H, W = X.shape
    #         X = X.reshape(C, -1)               # now (C, H*W)
            
    #         batch_mean = X.mean(axis=1)        # (C,)
    #         batch_var  = X.var(axis=1)         # (C,)
    #         batch_n    = X.shape[1]            # H*W pixels per channel

    #         if mean is None:
    #             # first batch: init accumulators
    #             mean  = batch_mean
    #             M2    = batch_var * batch_n
    #             n_pix = batch_n
    #         else:
    #             # Welford update
    #             delta = batch_mean - mean
    #             total = n_pix + batch_n

    #             mean += delta * (batch_n / total)
    #             M2   += batch_var * batch_n + (delta**2) * (n_pix * batch_n / total)
    #             n_pix  = total

    #     std = np.sqrt(M2 / n_pix)
    #     return mean.astype(np.float32), std.astype(np.float32)
    # def generate_mean_and_std_labels(self, temp_unit_indices):
    #     n_pix = 0
    #     mean  = None
    #     M2    = None
    #     for t in temp_unit_indices:
    #         # feature_map[t] has shape (C, H, W)
    #         X = np.asarray(self.annotations_map[t], dtype=np.float64)
    #         C, H, W = X.shape
    #         X = X.reshape(C, -1)               # now (C, H*W)
                
    #         batch_mean = X.mean(axis=1)
    #         batch_var  = X.var(axis=1)         # (C,)
    #         batch_n    = X.shape[1]            # H*W pixels per channel
    #         if mean is None:
    #                 # first batch: init accumulators
    #                 mean  = batch_mean
    #                 M2    = batch_var * batch_n
    #                 n_pix = batch_n
    #         else:
    #                 # Welford update
    #                 delta = batch_mean - mean
    #                 total = n_pix + batch_n

    #                 mean += delta * (batch_n / total)
    #                 M2   += batch_var * batch_n + (delta**2) * (n_pix * batch_n / total)
    #                 n_pix  = total
    #     std = np.sqrt(M2 / n_pix)
    #     return mean.astype(np.float32), std.astype(np.float32)
        
    def __getitem__(self, index):
        if not self.full:
            grid_coords, centre_coords, temp_unit = self.grid_and_centre_coords_and_temp_unit[index]

            image = self.feature_map[temp_unit, :, grid_coords[0]-self.rim:grid_coords[0]+self.grid_size+self.rim, grid_coords[1]-self.rim:grid_coords[1]+self.grid_size+self.rim]
            label = self.annotations_map[temp_unit, :, centre_coords[0], centre_coords[1]]
        else:
            # index = self.indices[index]
            image = self.feature_map[index]

            label = self.annotations_map[index]

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        image = torch.nan_to_num(image, nan=0.0)
        label = torch.nan_to_num(label, nan=0.0)

        if self.normalize:
            image = (image - self.mean) / self.std
            label = (label - self.mean_label) / self.std_label
        if self.transform:
            if self.full:
                combined = torch.cat((image, label), dim=0)
                combined_transformed = self.transform(combined)
                image, label = combined_transformed[:-1], combined_transformed[-1].squeeze()
            else:
                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
        
    def __len__(self):
        return len(self.indices)

    def name(self):
        return self.data_processor

class TestSubsetRegressionNewMod(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
    def __getitem__(self, idx):
        original_idx = self.indices[idx]

        if not self.dataset.full:
            grid_coords, centre_coords, temp_unit = self.dataset.grid_and_centre_coords_and_temp_unit[original_idx]
            image = self.dataset.feature_map[temp_unit, :, grid_coords[0]-self.dataset.rim:grid_coords[0]+self.dataset.grid_size+self.dataset.rim, grid_coords[1]-self.dataset.rim:grid_coords[1]+self.dataset.grid_size+self.dataset.rim]
            label = self.dataset.annotations_map[temp_unit, :, centre_coords[0], centre_coords[1]]
        else:
            # temp_unit = original_idx
            # index = self.dataset.indices[original_idx]
            image = self.dataset.feature_map[original_idx]
            label = self.dataset.annotations_map[original_idx]
        # label = self.dataset.annotations_map[temp_unit, :, grid_coords[0]:grid_coords[0]+self.dataset.grid_size, grid_coords[1]:grid_coords[1]+self.dataset.grid_size].mean(axis=(1, 2))


        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        image = torch.nan_to_num(image, nan=0.0)
        label = torch.nan_to_num(label, nan=0.0)

        if self.dataset.normalize:
            image = (image - self.dataset.mean) / self.dataset.std
            label = (label - self.dataset.mean_label) / self.dataset.std_label
        
        if not self.dataset.full:
            return image, label, (grid_coords, centre_coords, temp_unit)
        else:
            return image, label,

    def __getitems__(self, indices: list[int]):
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.__getitem__(idx) for idx in indices]

if __name__ == "__main__":
    dataset = TemporalDatasetNewMod(season="summer", mld_res=1, feature_res=1/12, groupby="months", rim=True, filepath="/mnt/c/Users/samue/SynologyDrive/OceanPropInfSatImg/data/WaterOnlyMonthly/WaterOnlyMonthlyExtendedSeasonality.nc")
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset shape: {dataset[0][0].shape}, label shape: {dataset[0][1].shape}")
    print(dataset.grid_and_centre_coords_and_temp_unit[1])
    print(dataset[1][1])
    print(dataset.grid_and_centre_coords_and_temp_unit[100])
    print(dataset[100][1])