from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from netCDF4 import Dataset as NETCDF4Dataset
import numpy as np
from pathlib import Path
import yaml
import copernicusmarine
from utils.config import RAW_CONFIG, DATA_DIR, FEATURES
from sklearn.preprocessing import RobustScaler

class GLORYSDS(TorchDataset):
    def __init__(self, dataset_dir, transform = None, grid_size = 40, days = True, normalize = True):
        self.include_several_days = days
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
            for j in range(self.num_days):
                self.annotation_scalers[f"day {j}"] = RobustScaler()
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
                (i, j) 
                for i in range(region[0], region[0]+self.region_size - self.grid_size, self.offset_size)
                for j in range(region[1], region[1]+self.region_size - self.grid_size, self.offset_size)
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
        max_x = min(coordinates[0] + self.grid_size, self.annotations_map.shape[-2])
        max_y = min(coordinates[1] + self.grid_size, self.annotations_map.shape[-1])
        if max_x <= coordinates[0] or max_y <= coordinates[1]:
            padded_image = np.zeros((self.feature_map.shape[0], self.feature_map.shape[1], self.grid_size, self.grid_size))
            padded_label = np.zeros((self.annotations_map.shape[0], self.annotations_map.shape[1], self.grid_size, self.grid_size))
            image = padded_image
            label = padded_label
        else:
            image = self.feature_map[:, :, coordinates[0]:max_x, coordinates[1]:max_y]
            label = self.annotations_map[:, :, coordinates[0]:max_x, coordinates[1]:max_y]
            if max_x - coordinates[0] < self.grid_size or max_y - coordinates[1] < self.grid_size:
                padded_image = np.zeros((image.shape[0], image.shape[1], self.grid_size, self.grid_size))
                padded_label = np.zeros((label.shape[0], label.shape[1], self.grid_size, self.grid_size))
                padded_image[:, :, :(max_x-coordinates[0]), :(max_y-coordinates[1])] = image
                padded_label[:, :, :(max_x-coordinates[0]), :(max_y-coordinates[1])] = label
                image = padded_image
                label = padded_label
        return image, label
    
    def __len__(self):
        return len(self.all_indices)
    def __getitem__(self, idx):
        if idx >= len(self.all_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.all_indices)}")
        
        coordinates = self.all_indices[idx]

        image = self.feature_map[:, :, coordinates[0]:coordinates[0]+self.grid_size, coordinates[1]:coordinates[1]+self.grid_size]
        label = self.annotations_map[:, :, coordinates[0]:coordinates[0]+self.grid_size, coordinates[1]:coordinates[1]+self.grid_size]
        
        image, label = self._pad(coordinates, image, label)

        #normalize images and labels
        if self.normalize:
            scaled_imgs = []
            scaled_lbls = []
            for i in range(self.num_days):
                imgtobescaled = image[i]
                imgtobescaled, original_img_shape = self._convert_to_scaler_fmt(imgtobescaled)
                scaled_img = self.feature_scalers[f"day {i}"].transform(imgtobescaled)
                normal_scaled_img = self._convert_to_normal_fmt(scaled_img, original_img_shape)
                scaled_imgs.append(normal_scaled_img)

                lbltobescaled = label[i]
                lbltobescaled, original_lbl_shape = self._convert_to_scaler_fmt(lbltobescaled)
                scaled_lbl = self.annotation_scalers[f"day {i}"].transform(lbltobescaled)
                normal_scaled_lbl = self._convert_to_normal_fmt(scaled_lbl, original_lbl_shape)
                scaled_lbls.append(normal_scaled_lbl)
            image = np.stack(scaled_imgs, axis=0)
            label = np.stack(scaled_lbls, axis=0)

        if self.transform:
            image, label = self.transform(image, label)
            
        if not self.include_several_days:
            image = np.squeeze(image[0:1], axis=0)
            label = np.squeeze(label[0:1], axis=0)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label
    
class TestSubset(Subset):
    def __getitem__(self, idx):
        if self.dataset.transform:
            original_idx = self.indices[idx]
            original_transform = self.dataset.transform
            self.dataset.transform = None
            item = self.dataset[original_idx]
            self.dataset.transform = original_transform
            return item
        else:
            return super().__getitem__(self, idx)


if __name__ == "__main__":
    ds_name = "lat:30-60_long:-190--120_date:1993-10-11-1993-10-12.nc"
    ds = GLORYSDS(DATA_DIR/ds_name, days = False, normalize=True)
    sample_image, sample_label = ds[1]
    print(f"{sample_image.shape}, {sample_label.shape}")
        