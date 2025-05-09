import sys 
from captum.attr import LRP
from captum.attr._utils.lrp_rules import EpsilonRule, IdentityRule
import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from torcheval.metrics.functional import binary_f1_score
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
home_dir = '../../../../'
sys.path.append(home_dir)

import preprocessing as prep
import postprocessing as post
import training as train
import models as mod
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
import pickle

networktype = 'FFNN'
datatype = 'CMIP6'

args = prep.setup_argperse(datatype, networktype).parse_args()

experiment_name = args.experiment_name
precision = torch.float

og_shape = torch.tensor([2, 56,141])
down_sample_rate = 0
split = 0.2
num_workers_dataloader = 1

epochs = 500
min_loss = 1e9
since_best = 0
early_stop = 50
save_every = 10
number_of_ensembles = 19
predict_region = (slice(30,69.5),slice(-64.5,35),) # in degrees



data_load_path, model_save_path, _ = prep.setup_paths(
    experiment_name, datatype, networktype)
# is there GPU available?
device = torch.device('cuda:'+str(args.gpu_idx) if torch.cuda.is_available() else 'cpu') 



og_shape = torch.tensor([2,56,141])
down_sample_rate = 0


vars_to_analyse = args.vars.split(',')
var_names_dict = {'ts': 'Temperature', 'psl': 'Pressure', 'zg': 'Geopotential height', 'zg_grad': 'Geopotential height gradient'}

crop = None
flatten = True

if args.crop:
    crop = (slice(0,None),slice(0,90),)
    flatten = False

X_train, y_train, X_val, y_val, pos_weight, idx_train, idx_val = prep.format_CMIP_X_y(
    path=data_load_path,
    og_shape=og_shape,
    down_sample_rate=down_sample_rate,
    dtype=precision,
    split=split,
    only_over=False,
    flatten=flatten,
    verbose=True,
    vars = args.vars.split(','),
    X_file = args.X_file_postfix,
    y_file = args.y_file,
    months= np.array(args.months.split(',')).astype(int),
    regresion = True,
    return_idx = True,
    time_shift = args.time_shift,
    crop=crop
    )

X_test, y_test, test_idx, pos_weight = prep.format_CMIP_X_y(
    path=data_load_path,
    og_shape=og_shape,
    down_sample_rate=down_sample_rate,
    dtype=precision,
    split=split,
    only_over=False,
    flatten=flatten,
    verbose=False,
    vars = args.vars.split(','),
    X_file = args.X_file_postfix,
    y_file = args.y_file,
    months= np.array(args.months.split(',')).astype(int),
    regresion = True,
    return_idx = True,
    test = True,
    time_shift = args.time_shift,
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

nc_file_name = 'CMIP6_target_combined'
if args.X_file_postfix != '':
    nc_file_name += '_' + args.X_file_postfix + '.nc'


data = xr.open_dataset(data_load_path + nc_file_name)


lats_for_plot = data['lat'].values
lons_for_plot = data['lon'].values
if args.crop:
    lons_for_plot = lons_for_plot[:90]

## combine X and y
X = torch.cat([X_train, X_val, X_test], dim=0)
y = torch.cat([y_train, y_val, y_test], dim=0)
total_index_for_save = torch.cat([torch.tensor(idx_train), torch.tensor(idx_val), torch.tensor(test_idx)], dim=0)

print(torch.sum(X))
print(torch.sum(y))
print(torch.sum(total_index_for_save))

if args.crop:
    X = X.reshape(-1, torch.prod(og_shape).item())

def make_countour_plot(ax, data, title, levels, transform=ccrs.PlateCarree(), title_size=25, cmap='coolwarm', label_size=20, plot_region=[-64.5, 35, 30, 69.5], target_region = (slice(53,58),slice(7,15))):
    ax.set_title(title, fontsize=title_size)
    ax.coastlines()
    ax.set_extent(plot_region, crs=transform)

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



n_layers = 12
layers = [4096, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 256]

activation = nn.ELU()
batch_norm = True # Sould definitely be true all cases without it cannot get below 0.3 MSEloss
dropout = 0.10345778206213857


folds = 10


def construct_TP_mask(y_true, y_pred):
    return (y_true == 1) & (y_pred == 1)



indexes = np.array_split(total_index_for_save, folds)   

# Create a dictionary mapping index to value before splitting
index_to_value = {idx.item(): val.item() for idx, val in zip(total_index_for_save, y)}



rules_dict = {
    nn.ReLU: EpsilonRule(epsilon=1e-6), 
    nn.Sigmoid: EpsilonRule(epsilon=1e-6), 
    nn.ELU: EpsilonRule(epsilon=1e-6),
    nn.BatchNorm1d: post.StableBatchNormRule(epsilon=1e-6, stabilizer=1e-3)
}

number_of_grid_points = torch.prod(og_shape).item()

lrp_of_data_sum = np.zeros(number_of_grid_points)
tot_TP = 0
tot_sum_output = 0
print('Sample LRP plots:', args.sample_lrp_plots)
print('Turn of TP mask:', args.turn_of_TP_mask)
if args.sample_lrp_plots:
    # find 3 random indexes
    random_fold = np.random.randint(0, folds)
    three_random_indexes = np.random.randint(0, 100, 20)

lrp_for_save = []
for i in range(folds):

    model = mod.FFNN(number_of_grid_points, [*layers], activation=activation, last_activation=None, last_number_of_nodes=1, dropout=0, batch_norm=batch_norm)

    if i == 0:
        print(model)
        summary(model, input_size=(100, number_of_grid_points,))
    state_dict = torch.load(model_save_path + f'/model_fold_{i}', map_location=device, weights_only=False)
    # Load the state_dict into the model (assuming the model architecture is already defined)
    model.load_state_dict(state_dict)
    model.eval()

    all_folds = np.arange(folds)
    current_test_indexes = indexes[i]
    X_test = X[current_test_indexes]
    y_test = y[current_test_indexes]

    train_folds = np.delete(all_folds, [i, (i+1)%folds])

    X_test = X_test.to(device)
    y_test = y_test.to(device)
    model = model.to(device)





    # Run the test fold and save the output
    output = model(X_test).flatten()
    tot_sum_output += torch.sum(output)
    TP_mask = construct_TP_mask(y_test>1, output > 1)
    tot_TP += torch.sum(TP_mask)
    if torch.sum(TP_mask) == 0:
        print('No TP in this fold')
        continue  
    # Apply rules with stabilization

    lrp_for_model = LRP(model)
    lrp_of_data = []
    
    for j, layer in enumerate(model.modules()):
        for key, value in rules_dict.items():
            if isinstance(layer, key):
                layer.rule = value



    # Compute LRP with stabilization. Here all are summed up including FP and FN
    if not args.turn_of_TP_mask:
        lrp_of_data = lrp_for_model.attribute(X_test[TP_mask])
    else:
        lrp_of_data = lrp_for_model.attribute(X_test)
    # Apply stabilization before summing
    lrp_of_data = post.stabilize_relevance(lrp_of_data)  
    lrp_of_data_sum += torch.sum(lrp_of_data, dim=0).cpu().detach().numpy()
    lrp_for_save.append(lrp_of_data.cpu().detach().numpy())

    if args.sample_lrp_plots:
        if i == random_fold:
            for idx_to_test in three_random_indexes:
                # find the matching index
                idx = idx_to_test 
                idx_LRP = lrp_of_data[idx].cpu().detach().numpy().reshape(*og_shape)
                var_1_LRP = idx_LRP[0]
                var_2_LRP = idx_LRP[1]

                #check for nan values in var_1_LRP, var_2_LRP
                if np.isnan(var_1_LRP).any() or np.isnan(var_2_LRP).any():
                    print('Nan values in LRP')
                    continue
                
                map_proj = ccrs.LambertConformal(central_longitude=(35-64.5)/2, central_latitude=(69.5+30)/2)
                plt.close('all')
                fig,axes = plt.subplots(1, 2, subplot_kw={'projection': map_proj}, figsize=(30, 30))
                label_size = 25
                plt.rcParams.update({'font.size': label_size})
                target_region = (slice(53,58),slice(7,15))
                cmin_max = np.max(np.abs(np.array([var_1_LRP.min(), var_1_LRP.max(), var_2_LRP.min(), var_2_LRP.max()])))
                im = make_countour_plot(axes[0], var_1_LRP, var_names_dict[vars_to_analyse[0]], np.linspace(-cmin_max, cmin_max, 21),cmap='coolwarm', target_region = target_region)
                im = make_countour_plot(axes[1], var_2_LRP, var_names_dict[vars_to_analyse[1]], np.linspace(-cmin_max, cmin_max, 21),cmap='coolwarm', target_region = target_region)
                # add colorbar
                cbar_ax = fig.add_axes([0.15, 0.25, 0.7, 0.02])  # [left, bottom, width, height]
                # Add the colorbar using the new axis
                cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
                fig.savefig(f'Figures/lrp_examples/lrp_map_{idx_to_test}_map_formattet.png')
                plt.close('all')
lrp_for_save = np.concatenate(lrp_for_save)
np.save('lrp_out',lrp_for_save)
lrp_of_data_sum /= tot_TP.item()
print('Total TP:', tot_TP)
print('Total sum output:', tot_sum_output)

lrp_of_data_sum = lrp_of_data_sum.reshape(*og_shape)

# Define the map projection and extent

map_proj = ccrs.LambertConformal(central_longitude=(35-64.5)/2, central_latitude=(69.5+30)/2)
predict_region = (slice(30, 69.5), slice(-64.5, 35))

# Set color bar min and max values
label_size = 25
text_size = 30
plt.rcParams.update({'font.size': label_size})

cbar_limit = np.percentile(np.abs(lrp_of_data_sum), 99)  # Use 99th percentile instead of max
vmin, vmax = -cbar_limit, cbar_limit
# if lrp < 0.2 * cbar_limit set to 0
# lrp_of_data_sum[np.abs(lrp_of_data_sum) < 0.2 * cbar_limit] = 0
# Create the figure and axes with shared labels





label_size = 25
plt.rcParams.update({'font.size': label_size})

fig, axes = plt.subplots(
    nrows=1, ncols=2,  # Two side-by-side plots
    subplot_kw={'projection': map_proj},
    figsize=(30, 15)
)
ax = axes.flatten()
print(cbar_limit)
vmin = -.425
vmax = .425




    
# Plot each 'lrp' index on a separate axis
for i, ax in enumerate(axes):
    im = make_countour_plot(ax, lrp_of_data_sum[i], var_names_dict[vars_to_analyse[i]], np.linspace(vmin, vmax, 21), cmap='coolwarm', target_region = target_region)

cbar_ax = fig.add_axes([0.15, 0.25, 0.7, 0.02])  # [left, bottom, width, height]

# Add the colorbar using the new axis
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.mappable.set_clim(vmin, vmax)
cbar.set_label('LRP Value', size=label_size)
cbar.ax.tick_params(labelsize=label_size)

# Add shared labels for longitude and latitude
# fig.text(0.516,.29, 'Longitude', ha='center')  # Shared x-axis label
# fig.text(0, 0.53, 'Latitude', va='center', rotation='vertical')  # Shared y-axis label

fig.savefig('Figures/lrp_map.png')
