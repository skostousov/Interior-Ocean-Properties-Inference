from utils.datasettemporal import TemporalDataset
from utils.datasettemporalxarray import XArrayDataset
from utils.dataset025 import PaperlikeDataset
from utils.config import PROJECT_ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math

data_class = PaperlikeDataset

root = Path(PROJECT_ROOT)

relative_file = "data/monthly/ten_sample_1993-2003.nc"
filepath = root / relative_file
filepath_no_nc = root/relative_file.replace(".nc", f"_{data_class.name()}.png")

ds = data_class(filepath=filepath)

annotation_map, feature_map = ds.annotations_map, ds.feature_map

print(annotation_map.shape, feature_map.shape)

annotation_map_to_plot = np.squeeze(annotation_map)

#unit = 12
unit = 2

fig, ax = plt.subplots(math.ceil(annotation_map.shape[0]/unit), unit, figsize=(math.ceil(annotation_map.shape[0]/unit) * 5, 5 * unit))
ax = np.atleast_2d(ax)


for month, image in enumerate(annotation_map_to_plot):
    im = ax[month//unit, month%unit].imshow(image, cmap='viridis', vmax=90, vmin=0)
    ax[month//unit, month%unit].set_title(f"Temporal Unit {month+1} ")
    ax[month//unit, month%unit].axis('off')
    fig.colorbar(im, label='MLD (m)', ax=ax[month//unit, month%unit])
plt.savefig(filepath_no_nc, dpi=300)
