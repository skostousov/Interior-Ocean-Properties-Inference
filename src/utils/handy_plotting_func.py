import numpy as np

def plot_var(axis, fig, var, var_name):
    vmin = 0
    vmax = np.max(var)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    im = axis.imshow(var, origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(im, ax=axis, label=var_name)