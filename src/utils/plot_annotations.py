import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import numpy as np


slider = widgets.IntSlider(value=0, min=0, max=mlds.shape[0]-1, step=1, description='Day')

def plot_var(axis, fig, day, var, var_name):
    vmin = 0
    vmax = np.max(var)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    im = axis.imshow(var[day], origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(im, ax=axis, label=var_name)

def plot_day(day, param1, param2):
    fig, axs = plt.subplots(figsize=(15, 5), ncols=2)
    plot_var(axs[1], fig, day, param1, "Depth of Mixed Ocean Layer (m)")
    plot_var(axs[0], fig, day, param2, "Sea Water Pressure at Sea Floor (dbar)")


widgets.interact(plot_day, day=slider)