from utils.datasettemporal import TemporalDataset
from utils.config import PROJECT_ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math

root = Path(PROJECT_ROOT)

relative_file = "data/monthly/ten_sample_1993-2003.nc"
filepath = root / relative_file
filepath_no_nc = root/relative_file.replace(".nc", ".png")

ds = TemporalDataset(filepath=filepath)

annotation_map, feature_map = ds.annotations_map, ds.feature_map

print(annotation_map.shape, feature_map.shape)

annotation_map_to_plot = np.squeeze(annotation_map)


fig, ax = plt.subplots(10, math.ceil(annotation_map.shape[0]/10), figsize=(50, math.ceil(annotation_map.shape[0]/10) * 5))
for month, image in enumerate(annotation_map_to_plot):
    im = ax[month%10, month//10].imshow(image, cmap='viridis')
    ax[month%10, month//10].set_title(f"Month {month+1}")
    ax[month%10, month//10].axis('off')
    fig.colorbar(im, label='MLD (m)', ax=ax[month%10, month//10])
plt.savefig(filepath_no_nc, dpi=300)
