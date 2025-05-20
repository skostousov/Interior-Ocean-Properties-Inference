from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from netCDF4 import Dataset as NETCDF4Dataset
import numpy as np
from pathlib import Path
import yaml
import copernicusmarine
from utils.config import RAW_CONFIG, DATA_DIR, FEATURES
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import torch

class GLORYSDS2(TorchDataset):
    def __init__(self, dataset_dir, transform = None, grid_size = 40, normalize = True):
        self.transform = transform
        self.normalize = normalize
        self.data = NETCDF4Dataset(dataset_dir)

        
        # relevant variable extraction
        self.data_variables = {k:v[:] for (k, v) in self.data.variables.items() if k in FEATURES}
        for key, _ in self.data.dimensions.items():
            if key in self.data_variables.keys():
                del self.data_variables[key]

        #ensuring each variable is of shape (days, 1, long, lat)
        for key, value in self.data_variables.items():
            if len (self.data_variables[key].shape) != 4: self.data_variables[key] = np.expand_dims(value, 1)
        
        # creation of label map -> shape (days, 1, long, lat)
        self.annotations_map = self.data_variables.pop("mlotst")

        # creation of feature map by concatenating channels together -> shape (days, channels, long, lat)
        self.feature_map = np.concatenate(list(self.data_variables.values()), axis=1)
        
        self.num_days = self.feature_map.shape[0]

        #fit scalers
        if normalize:
            self.feature_scalers = {}
            self.annotation_scalers = {}
            for i in range(self.num_days):
                self.feature_scalers[f"day {i}"] = RobustScaler()
                self.annotation_scalers[f"day {i}"] = RobustScaler()
            for i in range(self.num_days):
                full_day_img = self.feature_map[i]
                full_day_lbl = self.annotations_map[i]
                full_day_img, _ = self._convert_to_scaler_fmt(full_day_img)
                full_day_lbl, _ = self._convert_to_scaler_fmt(full_day_lbl)
                self.feature_scalers[f"day {i}"].fit(full_day_img)
                self.annotation_scalers[f"day {i}"].fit(full_day_lbl)

        self.grid_size = grid_size
        self.offset_size = 2
        self.images_in_region_one_axis = 4
        self.region_size = grid_size + self.offset_size*(self.images_in_region_one_axis-1)
        self.regions = [
            (i, j)
            for i in range(0, self.annotations_map.shape[-2], self.region_size)
            for j in range(0, self.annotations_map.shape[-1], self.region_size)
            if not self._just_land(self.feature_map[:, :, i:i+self.region_size, j:j+self.region_size])
        ]
        self.indices_regionified = {}
        for region in self.regions:
            self.indices_regionified[region] = [
                (day, i, j) 
                for i in range(region[0], region[0]+self.region_size - self.grid_size, self.offset_size)
                for j in range(region[1], region[1]+self.region_size - self.grid_size, self.offset_size)
                for day in range(0, self.num_days)
                ]
        self.all_indices = []
        for indices_of_region in self.indices_regionified.values():
            self.all_indices.extend(indices_of_region)
        self.indexed_region = []
        for j, value in enumerate(self.indices_regionified.values()):
            self.indexed_region.extend([j for i in range(len(value))])
    def _convert_to_scaler_fmt(self, tensor):
        #takes in shape (C, W H)
        C, W, H = tensor.shape
        reversed_tensor = np.transpose(tensor, (1, 2, 0))
        flattened_tensor = reversed_tensor.reshape(-1, C)
        return flattened_tensor, (C, W, H)
    def _convert_to_normal_fmt(self, flattened_tensor, shape):
        C, W, H = shape
        normal_tensor = flattened_tensor.reshape(W, H, C).transpose(2, 0, 1)
        return normal_tensor

    def _just_land(self, data):
        result = (data == 0)
        if result.size == 0 or result.sum() == 0:
            land = True
        else:
            land = bool(result.all())
        return land
    def unnormalize(self, annotation, day):
        ##takes as input dataset of size (1, 1, long, lat)
        scaler = self.annotation_scalers[f"day {day}"]
        scaled_fmt, original_shape = self._convert_to_scaler_fmt(annotation[0])
        unnormalized = scaler.inverse_transform(scaled_fmt)
        reshaped = self._convert_to_normal_fmt(unnormalized, original_shape)
        reshaped_expanded = np.expand_dims(reshaped, axis=0)
        return reshaped_expanded

    def _pad(self, coordinates, image, label):
        max_x = min(coordinates[1] + self.grid_size, self.annotations_map.shape[-2])
        max_y = min(coordinates[2] + self.grid_size, self.annotations_map.shape[-1])
        if max_x <= coordinates[1] or max_y <= coordinates[2]:
            padded_image = np.zeros((1, self.feature_map.shape[1], self.grid_size, self.grid_size))
            padded_label = np.zeros((1, self.annotations_map.shape[1], self.grid_size, self.grid_size))
            image = padded_image
            label = padded_label
        else:
            image = self.feature_map[coordinates[0], :, coordinates[1]:max_x, coordinates[2]:max_y]
            label = self.annotations_map[coordinates[0], :, coordinates[1]:max_x, coordinates[2]:max_y]
            if max_x - coordinates[1] < self.grid_size or max_y - coordinates[2] < self.grid_size:
                padded_image = np.zeros((image.shape[0], self.grid_size, self.grid_size))
                padded_label = np.zeros((label.shape[0], self.grid_size, self.grid_size))
                padded_image[:, :(max_x-coordinates[1]), :(max_y-coordinates[2])] = image
                padded_label[:, :(max_x-coordinates[1]), :(max_y-coordinates[2])] = label
                image = padded_image
                label = padded_label
        return image, label
    
    def __len__(self):
        return len(self.all_indices)
    def __getitem__(self, idx):
        if idx >= len(self.all_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.all_indices)}")
        
        coordinates = self.all_indices[idx]

        image = self.feature_map[coordinates[0], :, coordinates[1]:coordinates[1]+self.grid_size, coordinates[2]:coordinates[2]+self.grid_size]
        label = self.annotations_map[coordinates[0], :, coordinates[1]:coordinates[1]+self.grid_size, coordinates[2]:coordinates[2]+self.grid_size]

        image, label = self._pad(coordinates, image, label)

        #normalize images and labels
        if self.normalize:
            imgtobescaled = image
            imgtobescaled, original_img_shape = self._convert_to_scaler_fmt(imgtobescaled)
            scaled_img = self.feature_scalers[f"day {coordinates[0]}"].transform(imgtobescaled)
            normal_scaled_img = self._convert_to_normal_fmt(scaled_img, original_img_shape)
            lbltobescaled = label
            lbltobescaled, original_lbl_shape = self._convert_to_scaler_fmt(lbltobescaled)
            scaled_lbl = self.annotation_scalers[f"day {coordinates[0]}"].transform(lbltobescaled)
            normal_scaled_lbl = self._convert_to_normal_fmt(scaled_lbl, original_lbl_shape)
            image = normal_scaled_img
            label = normal_scaled_lbl
        
        if self.transform:
            image, label = self.transform(image, label)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label
    
class TestSubset(Subset):
    def __init__(self, dataset, indices, days=False):
        super().__init__(dataset, indices)
        self.days = days
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        # Get the day information from the original dataset
        coordinates = self.dataset.all_indices[original_idx]
        day = coordinates[0]  # This is the day
        
        if self.dataset.transform:
            original_transform = self.dataset.transform
            self.dataset.transform = None
            image, label = self.dataset[original_idx]
            self.dataset.transform = original_transform
        else:
            image, label = self.dataset[original_idx]
        if self.days:
            result = (image, label, torch.tensor(day, dtype=torch.int64))
            assert len(result) == 3, f"Got {len(result)} pieces at idx={idx}"
            return result
        else:
            result = (image, label)
            return result

if __name__ == "__main__":
    ds_name = RAW_CONFIG["datafile"]
    ds = GLORYSDS2(DATA_DIR/ds_name, normalize=True)
    sample_image, sample_label = ds[1]
    print(f"{sample_image.shape}, {sample_label.shape}")
    print(len(ds))