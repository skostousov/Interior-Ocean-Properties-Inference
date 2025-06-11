from utils.datasettemporal import TemporalDataset
from utils.datasettemporalxarray import XArrayDataset
from utils.config import PROJECT_ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math

data_class = TemporalDataset

root = Path(PROJECT_ROOT)

relative_file = "data/daily/very_small_daily_sample_1993-1994.nc"
filepath = root / relative_file
filepath_no_nc = root/relative_file.replace(".nc", f"_{data_class.name()}.png")

ds = data_class(filepath=filepath)

annotation_map, feature_map = ds.annotations_map, ds.feature_map

print(annotation_map.shape, feature_map.shape)

annotation_map_to_plot = np.squeeze(annotation_map)


fig, ax = plt.subplots(math.ceil(annotation_map.shape[0]/12), 12, figsize=(math.ceil(annotation_map.shape[0]/12) * 5, 50))
for month, image in enumerate(annotation_map_to_plot):
    im = ax[month//12, month%12].imshow(image, cmap='viridis', vmax=90, vmin=0)
    ax[month//12, month%12].set_title(f"Month {month+1}")
    ax[month//12, month%12].axis('off')
    fig.colorbar(im, label='MLD (m)', ax=ax[month//12, month%12])
plt.savefig(filepath_no_nc, dpi=300)
