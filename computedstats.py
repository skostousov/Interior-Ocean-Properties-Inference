from utils.datasettemporal import TemporalDataset
from utils.config import PROJECT_ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = Path(PROJECT_ROOT)

filepath = root / "data/monthly/ten_sample_1993-2003.nc"

ds = TemporalDataset(filepath=filepath)

annotation_map, feature_map = ds.annotations_map, ds.feature_map

print(annotation_map.shape, feature_map.shape)

fig, ax = plt.subplots(10, annotation_map.shape[0])
annotation_map_to_plot = np.squeeze(annotation_map)

for month, image in enumerate(annotation_map_to_plot):
    ax[month%10, month//10].imshow(image)