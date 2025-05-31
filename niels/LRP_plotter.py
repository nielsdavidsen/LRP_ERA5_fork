import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def normalize_lrp(lrp_maps):
    
    global_max = max(
        np.abs(lrp_maps[0]).max(),
        np.abs(lrp_maps[1]).max()
    )

    return lrp_maps / global_max  # scales to [-1, 1] if global_max is abs-max


def plot_LRP(lrp_attr_sum):

    lrp_norm = normalize_lrp(lrp_attr_sum)
    shared_norm = mcolors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
    cmap = 'coolwarm'

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    im0 = ax[0].imshow(lrp_norm[0], cmap=cmap, norm=shared_norm, aspect='auto')
    ax[0].set_title('T850 LRP Attribution Map')
    im1 = ax[1].imshow(lrp_norm[1], cmap=cmap, norm=shared_norm, aspect='auto')
    ax[1].set_title('MSL LRP Attribution Map')
    
    ax_list = ax.ravel().tolist()
    cbar = fig.colorbar(im1, ax=ax_list, orientation='horizontal', fraction=0.05, pad=0.15)
    cbar.set_label('LRP Attribution Value', fontsize=12)

    for a in ax:
        a.set_xlabel('X-grid index (longitude)')
        a.set_ylabel('Y-grid index (latitude)')
        a.grid(True, linestyle='--', alpha=0.5)


    plt.suptitle('LRP Attribution Maps for T850 and MSL', fontsize=16)
    return fig, ax
