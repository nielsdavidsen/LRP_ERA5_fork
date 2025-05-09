
import sys 
home_dir = '../../../../'
sys.path.append(home_dir)

import postprocessing as post
import preprocessing as prep
import models as mod
import os
import pickle
import argparse
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import binary_f1_score

from sklearn.model_selection import train_test_split
import pandas as pd
import xarray as xr 
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs


networktype = 'FFNN'
datatype = 'CMIP6'

args = prep.setup_argperse(datatype, networktype).parse_args()

experiment_name = args.experiment_name
precision = torch.float

og_shape = torch.tensor([2, 56, 141])
down_sample_rate = 0
split = 0.2
num_workers_dataloader = 1

data_load_path, model_save_path, fig_save_path = prep.setup_paths(
    experiment_name, datatype, networktype, home_dir=home_dir)

vars_to_analyse = args.vars.split(',')
var_names_dict = {'ts': 'Temperature', 'psl': 'Pressure', 'zg': 'Geopotential height', 'zg_grad': 'Geopotential height gradient'}

print('Loading X  :', data_load_path + 'X_' + vars_to_analyse[0] +'_'+ args.X_file_postfix + '.pkl')
print('Loading X  :', data_load_path + 'X_' + vars_to_analyse[1] +'_'+ args.X_file_postfix + '.pkl')

print('Loading y  :', data_load_path + args.y_file)
months_zero_indexed = np.array(args.months.split(',')).astype(int)
print('\n')
crop = None
flatten = True
if args.crop:
    crop = (slice(0,None),slice(0,90),)
    flatten = False
X_train, y_train_old, X_val, y_val_old, _, idx_train, idx_val = prep.format_CMIP_X_y(
    path=data_load_path,
    og_shape=og_shape,
    down_sample_rate=down_sample_rate,
    dtype=precision,
    split=split,
    only_over=True,
    flatten=flatten,
    verbose=True,
    vars = vars_to_analyse,
    X_file = args.X_file_postfix,
    y_file = args.y_file,
    months= months_zero_indexed,
    regresion = False,
    return_idx = True,
    time_shift= args.time_shift,
    crop=crop
    )

X_tot, y_test_old, idx_test, _ = prep.format_CMIP_X_y(
    path=data_load_path,
    og_shape=og_shape,
    down_sample_rate=down_sample_rate,
    dtype=precision,
    split=split,
    only_over=True,
    flatten=flatten,
    verbose=False,
    vars = args.vars.split(','),
    X_file=args.X_file_postfix,
    y_file=args.y_file,
    months= months_zero_indexed,
    regresion = False,
    test = True,    
    time_shift= args.time_shift,
    crop=crop
    )


if experiment_name == 'detrend_reg_extended_winter_ts_psl_east_canada':
    target_region = (slice(50,54),slice(-62,-58)) # east_canada

elif experiment_name == 'detrend_reg_extended_winter_ts_psl_Bergen':
    target_region = (slice(58,62),slice(3,7)) # Bergen

elif experiment_name == 'detrend_reg_extended_winter_ts_psl_Bergen_cropped':
    target_region = (slice(58,62),slice(3,7)) # Bergen

else:
    target_region = (slice(53,58),slice(7,15))

if args.crop:
    og_shape = torch.tensor([2, 56, 90])

tot_index_class = torch.cat([torch.tensor(idx_train), torch.tensor(idx_val), torch.tensor(idx_test)])

X_tot = torch.cat([X_train, X_val, X_tot])
if args.crop:
    X_tot = X_tot.reshape(-1, torch.prod(og_shape).item())
folds = 10

y = []
for i in range(folds):
    with open(model_save_path + '/output_fold_' + str(i) + '.pkl', 'rb') as f:
        fold_data = pickle.load(f)  # This is already a list of tuples
        if isinstance(fold_data, tuple):  # Check if pickle.dump wrapped it in an extra tuple
            fold_data = fold_data[0]
        y.extend(fold_data)

# Map the sequential indices back to original indices
y_with_original_idx = [(tot_index_class[x[0]].item(), x[1], x[2]) for x in y]

# Sort by original indices
y_with_original_idx.sort(key=lambda x: x[0])

# Unpack into arrays
indexes_from_folds = np.array([x[0] for x in y_with_original_idx])
y_test_pr = np.array([x[1] for x in y_with_original_idx])
output = np.array([x[2] for x in y_with_original_idx])

nc_file_name = 'CMIP6_target_combined'
if args.X_file_postfix != '':
    nc_file_name += '_' + args.X_file_postfix + '.nc'


# pick out the correct months 
combined_data_set_nc = xr.open_dataset(data_load_path + nc_file_name)
combined_data_set_nc = combined_data_set_nc.sel(time=combined_data_set_nc.time.dt.month.isin(months_zero_indexed+1))
pr_nc = combined_data_set_nc['pr'].values.flatten()
target_nc = combined_data_set_nc['target'].values.flatten()
target_nc[target_nc>1] = 0
target_nc = target_nc.astype(bool)
bins_count = plt.hist(pr_nc, bins = 13)
plt.close('all')

lats_for_plot = combined_data_set_nc['lat'].values
lons_for_plot = combined_data_set_nc['lon'].values
if args.crop:
    lons_for_plot = lons_for_plot[:90]

plt.hist(pr_nc[target_nc], bins = bins_count[1], alpha = 0.5, label = 'True')
plt.hist(pr_nc[~target_nc], bins = bins_count[1], alpha = 0.5, label = 'False')
plt.savefig(fig_save_path + '/histogram_old_pr.png')
plt.close('all')



plt.hist(output, bins=100, density=True)
plt.hist(y_test_pr, bins=100, alpha=0.5, density=True)
plt.title('Histogram of model output')
plt.xlabel('Model output')
plt.ylabel('Frequency')
plt.savefig('model_output_hist.png')
plt.savefig(fig_save_path + '/model_output_hist.png')

from scipy.stats import ks_2samp

ks_stat, ks_p = ks_2samp(output, y_test_pr)

mse_loss_on_test = nn.MSELoss()(torch.tensor(output), torch.tensor(y_test_pr))

#sort output and y_test by tot_index


y_old = np.concatenate((y_train_old, y_val_old,y_test_old))

y_old = y_old[tot_index_class.argsort()].astype(bool)


output = output > 1
y_test = y_test_pr > 1

bins_count = plt.hist(y_test_pr , bins = 13)
plt.close('all')    

plt.hist(y_test_pr[y_old] , bins = bins_count[1], alpha = 0.5, label = 'True')
plt.hist(y_test_pr[~y_old] , bins = bins_count[1], alpha = 0.5, label = 'False')
plt.savefig(fig_save_path + '/histogram_old_pr_test.png')
plt.close('all')
print('KS test:', ks_stat, ' p-value: ', ks_p)
print('MSE loss on test set:', mse_loss_on_test.item())
print('F1 score on test set:', binary_f1_score(torch.tensor(output), torch.tensor(y_test)))
print('Heidke skill score on test set:', post.heidke_skill_score(
    y_test, output))

print('Recall on test set:', recall_score(y_test, output))


print('Precision on test set:', precision_score(y_test, output, zero_division = 0), end = '\n\n')


#latex formated output
print(f'& {ks_stat:.3f} & {ks_p:.3e} & {mse_loss_on_test.item():.3f} & {binary_f1_score(torch.tensor(output), torch.tensor(y_test)):.3f} & {post.heidke_skill_score(y_test, output):.3f} & {recall_score(y_test, output):.3f} & {precision_score(y_test, output, zero_division=0):.3f} \\\\', end='\n\n')



fig, confmatrix = post.confusion_matrix(y_test, output)

# add colors around the squares in the matrix to indicate the TN, TP, FN, FP color in the histogram
ax = plt.gca()
color_palette = sns.color_palette("colorblind", 4)

plt.savefig(fig_save_path + '/confusion_matrix.png')
plt.close('all')


conf_list = np.zeros_like(output, dtype = int)

string_tp_fp_tn_fn = ['TN', 'TP', 'FN', 'FP']
for i in range(len(output)):
    if output[i] == 0:
        if output[i] != y_test[i]:
            conf_list[i] = 2
    else:
        if y_test[i] == 1:
            conf_list[i] = 1
        else:
            conf_list[i] = 3

bins_count = plt.hist(y_test_pr, bins = 13)


plt.close('all')

fig, ax = plt.subplots(figsize = (30,20))
font_size = 30
plt.rcParams.update({'font.size': font_size})

print('number of samples ', len(conf_list))
for_stack_hist = np.zeros((len(y_test_pr[np.where(conf_list  == 0)]),4))
for_stack_hist[:] = np.nan


for i in range(4):
    into_array = y_test_pr[np.where(conf_list  == i)]
    for_stack_hist[:len(into_array),i] = into_array


idx = np.argmin(np.abs(1-bins_count[1]))
bins_shifted = bins_count[1]+1-bins_count[1][idx]
for i in range(4):
    ax.hist(for_stack_hist[:,i], bins = bins_shifted, label = string_tp_fp_tn_fn[i], color = color_palette[i], alpha = 0.5,
histtype='bar')

ax.set_xlabel('Normalised Precipitation', fontsize = font_size)
ax.set_ylabel('Frequency', fontsize = font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.tick_params(axis='both', which='minor', labelsize=font_size)
#
plt.legend()
# set all labels size to 20
plt.savefig(fig_save_path + '/histogram.png', bbox_inches = 'tight', dpi = 100)


fig, ax = plt.subplots(figsize = (30,20))
font_size = 30
plt.rcParams.update({'font.size': font_size})

print('number of samples ', len(conf_list))
for_stack_hist = np.zeros((len(y_test_pr[np.where(conf_list  == 0)]),4))
for_stack_hist[:] = np.nan


for i in range(4):
    into_array = y_test_pr[np.where(conf_list  == i)]
    for_stack_hist[:len(into_array),i] = into_array


idx = np.argmin(np.abs(1-bins_count[1]))
bins_shifted = bins_count[1]+1-bins_count[1][idx]
for i in range(4):
    if i == 0 or i == 3:
        colour = color_palette[0]
    else:
        colour = color_palette[1]
    ax.hist(for_stack_hist[:,i], bins = bins_shifted, label = string_tp_fp_tn_fn[i], color = colour, 
histtype='bar',stacked = True)

ax.set_xlabel('Normalised Precipitation', fontsize = font_size)
ax.set_ylabel('Frequency', fontsize = font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.tick_params(axis='both', which='minor', labelsize=font_size)
#
# set all labels size to 20
plt.savefig(fig_save_path + '/histogram_plain.png', bbox_inches = 'tight', dpi = 100)




plt.close('all')

fig, ax = plt.subplots(figsize = (10,10))
font_size = 15
plt.rcParams.update({'font.size': font_size})


for_stack_hist = np.zeros((len(y_test_pr[np.where(conf_list  == 0)]),4))
for_stack_hist[:] = np.nan


for i in range(4):
    into_array = y_test_pr[np.where(conf_list  == i)]
    for_stack_hist[:len(into_array),i] = into_array


idx = np.argmin(np.abs(1-bins_count[1]))
bins_shifted = bins_count[1]+1-bins_count[1][idx]
for i in range(4):
    if i == 0 or i == 3:
        i_color = 0
    else:
        i_color = 1
    ax.hist(for_stack_hist[:,i], bins = bins_shifted, color = color_palette[i_color], stacked = True, histtype='bar')

ax.set_xlabel('Normalised Precipitation', fontsize = font_size)
ax.set_ylabel('Frequency', fontsize = font_size)
ax.set_tick_size = font_size
#
plt.legend()
# set all labels size to 20
plt.savefig(fig_save_path + '/histogram_for_data.png', bbox_inches = 'tight', dpi = 300)




# Define the map projection and extent
predict_region = (slice(30, 69.5), slice(-64.5, 35))
map_proj = ccrs.LambertConformal(central_longitude=(35-64.5)/2, central_latitude=(69.5+30)/2)


def make_countour_plot(ax, data, title, levels, transform=ccrs.PlateCarree(), title_size=25, cmap='coolwarm', label_size=20, predict_region=[-64.5, 35, 30, 69.5]):
    ax.set_title(title, fontsize=title_size)
    ax.coastlines()
    ax.set_extent(predict_region, crs=transform)

    # Add gridlines and set tick parameters
    gl = ax.gridlines(draw_labels=True, crs=transform, linestyle="--")

    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels
    gl.xlocator = plt.FixedLocator([-60, -30, 0, 30])

    gl.label_offset = 100  # Set the label offset from the gridlines
    gl.xlabel_style = {'size': label_size}
    gl.ylabel_style = {'size': label_size}
    ax.contour(lons_for_plot, lats_for_plot, data, levels=levels, colors='k', transform=transform, linewidths=0.5)
    im = ax.contourf(lons_for_plot, lats_for_plot, data, cmap=cmap, origin='lower', extend='both', transform=transform, levels=levels)

    west = target_region[1].start
    east = target_region[1].stop
    south = target_region[0].start
    north = target_region[0].stop
    ax.plot([west, east, east, west, west], [south, south, north, north, south], 'k--', transform=transform)

    return im

total_gridpoints = torch.prod(torch.tensor(og_shape[1:])).item()

X_tot = X_tot.cpu().detach().numpy()
mean_confs = np.zeros((8,  X_tot.shape[1]//2))
sigma_confs = np.zeros((8,  X_tot.shape[1]//2))
for i in range(4):
    mask = np.where(conf_list == i)
    print(string_tp_fp_tn_fn[i], np.sum(conf_list == i))    
    print(string_tp_fp_tn_fn[i], 'mean ', var_names_dict[vars_to_analyse[0]], X_tot[mask, :total_gridpoints].mean(),
    var_names_dict[vars_to_analyse[1]], X_tot[mask, total_gridpoints:].mean())
    mean_confs[i] =  X_tot[mask, :total_gridpoints].mean(axis = 1)
    sigma_confs[i] =  X_tot[mask, :total_gridpoints].std(axis = 1)
    mean_confs[i+4] =  X_tot[mask, total_gridpoints:].mean(axis = 1)
    sigma_confs[i+4] =  X_tot[mask, total_gridpoints:].std(axis = 1)

label_size = 25
plt.rcParams.update({'font.size': label_size})

fig, axes = plt.subplots(
    nrows=2, ncols=2,  # Two side-by-side plots
    subplot_kw={'projection': map_proj},
    figsize=(30, 15)
)
ax = axes.flatten()
vmin = -0.15
vmax = 0.15

for i in range(4):
    im = make_countour_plot(ax[i], mean_confs[i].reshape(*og_shape[1:]), string_tp_fp_tn_fn[i] + ' ' + var_names_dict[vars_to_analyse[0]],
    np.linspace(vmin, vmax, 19))

cbar = fig.colorbar(im, ax=ax,  pad=0.05, aspect=50)
cbar.mappable.set_clim(vmin, vmax)  #

plt.savefig(fig_save_path + '/mean_heatmaps_' + vars_to_analyse[0] + '.png', bbox_inches = 'tight', dpi = 300)
plt.close('all')


label_size = 25
plt.rcParams.update({'font.size': label_size})
fig, ax = plt.subplots(
    nrows=2, ncols=2,  # Two side-by-side plots
    subplot_kw={'projection': map_proj},
    figsize=(30, 15)
)
ax = ax.flatten()

for i in range(4):
    im = make_countour_plot(ax[i], mean_confs[i+4].reshape(*og_shape[1:]),
    string_tp_fp_tn_fn[i] + ' ' + var_names_dict[vars_to_analyse[1]], np.linspace(vmin, vmax, 19))

cbar = fig.colorbar(im, ax=ax,  pad=0.05, aspect=50)
cbar.mappable.set_clim(vmin, vmax)  #


plt.savefig(fig_save_path + '/mean_heatmaps_' + vars_to_analyse[1] + '.png', bbox_inches = 'tight', dpi = 300)
plt.close('all')




plt.rcParams.update({'font.size': label_size})

fig, axes = plt.subplots(
    nrows=2, ncols=2,  # Two side-by-side plots
    subplot_kw={'projection': map_proj},
    figsize=(30, 15)
)
ax = axes.flatten()



for i in range(2):
    to_be_plotted = mean_confs[i].reshape(*og_shape[1:]) - mean_confs[i+2].reshape(*og_shape[1:])
    to_be_plotted /= np.sqrt(sigma_confs[i].reshape(*og_shape[1:])**2 + sigma_confs[i+2].reshape(*og_shape[1:])**2)
    im1 = make_countour_plot(ax[i], to_be_plotted, string_tp_fp_tn_fn[i] + ' - ' + string_tp_fp_tn_fn[i+2] + ' '+var_names_dict[vars_to_analyse[0]], np.linspace(vmin, vmax, 19))



for i in range(2):
    to_be_plotted = mean_confs[i+4].reshape(*og_shape[1:]) - mean_confs[i+6].reshape(*og_shape[1:])
    to_be_plotted /= np.sqrt(sigma_confs[i+4].reshape(*og_shape[1:])**2 + sigma_confs[i+6].reshape(*og_shape[1:])**2)
    im2 = make_countour_plot(ax[i+2], to_be_plotted, string_tp_fp_tn_fn[i] + ' - ' + string_tp_fp_tn_fn[i+2] + ' '+var_names_dict[vars_to_analyse[1]], np.linspace(vmin, vmax, 19))


cbar = fig.colorbar(im2, ax=ax,  pad=0.05, aspect=50)
cbar.mappable.set_clim(vmin, vmax)  #




plt.savefig(fig_save_path + '/mean_heatmaps_diff.png', bbox_inches = 'tight', dpi = 300)
plt.close('all')
