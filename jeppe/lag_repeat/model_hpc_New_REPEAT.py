import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import optuna

import torch
import torch.optim as optim
import torch.nn as nn
from torchinfo import summary
from torcheval.metrics.functional import binary_f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# import preprocessing as prep
# import postprocessing as post
# import training as train
# import models as mod
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import gamma as gamma_dist
from captum.attr._utils.lrp_rules import EpsilonRule

from captum.attr import LRP
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ----- Borrowed from Melcher -----
class StableBatchNormRule(EpsilonRule):
    def __init__(self, epsilon=1e-6, stabilizer=1e-3):
        super().__init__(epsilon)
        self.stabilizer = stabilizer
    
    def forward_hook(self, module, inputs, outputs):
        """Store normalized inputs for better relevance propagation"""
        self.normalized_input = (inputs[0] - module.running_mean[None, :]) / torch.sqrt(module.running_var[None, :] + module.eps)
        return super().forward_hook(module, inputs, outputs)
    
    def backward_hook(self, module, grad_input, grad_output):
        """Stabilize the gradient propagation"""
        # Apply stabilization to prevent explosion
        grad_modified = grad_input[0] / (torch.norm(grad_input[0], dim=1, keepdim=True) + self.stabilizer)
        return (grad_modified,) + grad_input[1:]
    
rules_dict = {
    nn.ReLU: EpsilonRule(epsilon=1e-6), 
    nn.Sigmoid: EpsilonRule(epsilon=1e-6), 
    nn.ELU: EpsilonRule(epsilon=1e-6),
    nn.BatchNorm1d: StableBatchNormRule(epsilon=1e-6, stabilizer=1e-3)
}
# ------------------------------------




# class gamma_activation(nn.Module):
#     def forward(self, x):
#         return torch.gamma_dist.cdf(x, a=1.0, scale=1.0)  # Example parameters, adjust as needed

def standard_scale_day(group, eps = 1e-6):
    mean_val = group.mean(dim='valid_time')
    std_val = group.std(dim='valid_time') + eps
    return (group - mean_val) / std_val


def sample_data(X, n):
    t, v, x, y = X.shape
    X_flat = X.reshape(t, v, x * y)

    # Sample every n-th element along the last dimension
    sampled_X = X_flat[:, :, ::n]

    # Split the sampled data into n parts along the last dimension
    split_X = np.array_split(sampled_X, n, axis=2)
    

    # Reshaping into (t, v, x, y) format    new_x, new_y = x // n, y // n
    new_x = x // n
    new_y = y // n

    result_datasets = []
    for i, part in enumerate(split_X):
        reshaped = part.reshape(t, v, new_x, new_y)
        # Create an xarray Dataset or DataArray from this
        da = xr.DataArray(reshaped, dims=('time', 'variable', 'x', 'y'))
        result_datasets.append(da)

    # Concatenate the datasets along time dimension
    result_datasets = xr.concat(result_datasets, dim='time')
    return result_datasets

class Model_FFN:
    def __init__(self, latitude_range=(70, 40.25), longitude_range=(-80, 21.75), DEVICE = 'cpu'):
        self.latitude_range = latitude_range
        self.longitude_range = longitude_range
        self.X = None
        self.target = None
        self.lag = 0
        self.dataset = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.best_model_state = None
        self.ypred = None
        self.DEVICE = DEVICE
        self.OG_shape = None  # Original shape of the data before reshaping for LRP
        self.lat_for_plot = None
        self.lon_for_plot = None

    def load_data(self, sub_sampling = False, sub_sample_dim = 4, five_year_test = False):
        file_paths_msl = "/dmidata/users/jpgp/AppML/ERA5/all_years/ERA5_mean_sea_level_pressure_merged.nc"
        ds_msl = xr.open_dataset(file_paths_msl).sel(
            latitude=slice(self.latitude_range[0], self.latitude_range[1]),
            longitude=slice(self.longitude_range[0], self.longitude_range[1])
        )
        self.lat_for_plot = ds_msl.latitude.values
        self.lon_for_plot = ds_msl.longitude.values
        doy = ds_msl['valid_time'].dt.dayofyear
        ds_msl = ds_msl.assign_coords(day_of_year=doy) ## 28 feb is 366. i have 2 of these in the 5 years.
        msl_stand = ds_msl['msl'].groupby('day_of_year').map(standard_scale_day) ## scale the data
        ds_msl['msl_stand'] = msl_stand
        if five_year_test:
            print("Using 5-year test data...")
            ds_msl = ds_msl.isel(valid_time=slice(0, 1827))
            msl_input = ds_msl.msl_stand.values 
        if not five_year_test:
            print("Using full dataset...")
            msl_input = ds_msl.msl_stand.values


        if np.isnan(msl_input).any():
            raise ValueError("NaN values found in the input data. Please check the dataset for missing values.")

        print("Msl input shape:", msl_input.shape)


##################################T2m#############################################


        file_paths_t2m = "/dmidata/users/jpgp/AppML/ERA5/all_years/ERA5_temperature_merged.nc"
        ds_t2m = xr.open_dataset(file_paths_t2m).sel(
                                latitude=slice(self.latitude_range[0], self.latitude_range[1]),
                                longitude=slice(self.longitude_range[0], self.longitude_range[1])
                                        )
        doy_t2m = ds_t2m['valid_time'].dt.dayofyear
        ds_t2m = ds_t2m.assign_coords(day_of_year=doy_t2m)
        t2m_stand = ds_t2m['t2m'].groupby('day_of_year').map(standard_scale_day)
        ds_t2m['t2m_stand'] = t2m_stand

        if five_year_test:
            print("Using 5-year test data...")
            ds_t2m = ds_t2m.isel(valid_time=slice(0, 1827))
            t2m_input = ds_t2m.t2m_stand.values
        if not five_year_test:
            print("Using full dataset...")
            t2m_input = ds_t2m.t2m_stand.values
        if np.isnan(t2m_input).any():
            raise ValueError("NaN values found in the input data. Please check the dataset for missing values.")

        print("t2m input shape:", t2m_input.shape)


    #################### precipitation ##########################################


        file_paths_prec = "/dmidata/users/jpgp/AppML/ERA5/all_years/ERA5_total_precipitation_merged.nc"
        ds_prec = xr.open_mfdataset(file_paths_prec)
        ds_prec = ds_prec.mean(dim= ['longitude', 'latitude'])


        prec_target = ds_prec.tp.values

        
        if np.isnan(prec_target).any():
                raise ValueError("NaN values found in the target data. Please check the dataset for missing values.")
        
        if five_year_test:
            print("Using 5-year test data...")
            prec_target = prec_target[:1827]
        if not five_year_test:
            print("Using full dataset...")

        print("Precipitation target shape:", prec_target.shape)





        X = np.stack([t2m_input, msl_input], axis=1)
        
        if sub_sampling:
            print("Sub-sampling data...")
            X_c = sample_data(X, n=sub_sample_dim)
            print("Sub-sampled X shape:", X_c.shape)
            prec_target_concat = np.repeat(prec_target, 4, axis=0)
            print("Concatenated precipitation target shape:", prec_target_concat.shape)
            self.X = X_c.values
            self.target = prec_target_concat
            #return X_c, prec_target_concat



        else:
            print("No sub-sampling applied.")
            print("X shape:", X.shape)
            print("Precipitation target shape:", prec_target.shape)
            self.X = X
            self.target = prec_target
            #return X, prec_target
         

    def lag_data(self, lag=1):
        self.lag = lag
        self.X = self.X[:-lag]            # Shape: (1827 - lag, 2, 121, 409)
        self.target = self.target[lag:]
        print("Lagged X shape:", self.X.shape)
        print("Lagged target shape:", self.target.shape)
        
             
    def prepare_data_for_tensorflow(self, test_size = .1 , print_shapes = True):
        ##take 250 random samples from X to test the model
        n = self.X.shape[0]
        test_size = int(test_size * n)

        random_indices = np.random.choice(n, size=test_size, replace=False)
    # Create a boolean mask for test data
        mask = np.zeros(self.X.shape[0], dtype=bool)
        mask[random_indices] = True

    # Split the data using the mask
        X_test = self.X[mask]
        X_train = self.X[~mask]
        scaler = MinMaxScaler()

        y_test = scaler.fit_transform(self.target[mask].reshape(-1, 1))
        y_train =scaler.transform(self.target[~mask].reshape(-1, 1))
        if print_shapes:
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)
        X_tensor = torch.from_numpy(X_train)
        y_tensor = torch.from_numpy(y_train)
        print(X_tensor.shape)
        print(y_tensor.shape)


        X = X_tensor.view(X_train.shape[0], -1).to(self.DEVICE)
        y = y_tensor.view(y_train.shape[0], 1).to(self.DEVICE)

    # Create dataset
        dataset = TensorDataset(X, y)
        self.dataset = dataset
        self.OG_shape = X_test.shape

        self.X_test = torch.from_numpy(X_test).view(X_test.shape[0], -1).to(self.DEVICE)
        self.y_test = torch.from_numpy(y_test).view(y_test.shape[0], 1)

        #return dataset, X_test, y_test, 

     
    def build_model(self, dropout_rate=0.1,
                     hidden_dims=[2048, 2048,1024, 1024,1024, 1024, 256],
                     Sigmoid_output = True,
                     ReLU_output = False,
                     gamma_output = False):
        class FFNN(nn.Module):
            def __init__(self, input_dim, dropout_rate=0.1):
                super().__init__()

                #hidden_dims = [4096, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 256]
                #hidden_dims = [2048, 2048,1024, 1024,1024, 1024, 256]
                #hidden_dims = [2048, 1024, 512, 256]
                #hidden_dims = [512, 512, 256, 256]


                layers = []
                prev_dim = input_dim
                for hdim in hidden_dims:
                    linear_layer = nn.Linear(prev_dim, hdim)

                    nn.init.xavier_uniform_(linear_layer.weight)
                    nn.init.zeros_(linear_layer.bias)

                    layers.append(linear_layer)
                    layers.append(nn.BatchNorm1d(hdim)) 
                    layers.append(nn.ELU())
                    layers.append(nn.Dropout(dropout_rate))
                    prev_dim = hdim

                # Final output layer
                final_later = nn.Linear(prev_dim, 1)
                nn.init.xavier_uniform_(final_later.weight)
                nn.init.zeros_(final_later.bias)
                layers.append(final_later)



                if Sigmoid_output:
                    layers.append(nn.Sigmoid())
                if ReLU_output:
                    layers.append(nn.ReLU()) if ReLU_output else None
             

                # Use Sequential for compactness
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        # Model setup
        input_dim = self.X.shape[1] * self.X.shape[2] * self.X.shape[3]

        model = FFNN(input_dim)



        self.model = model.to(self.DEVICE)

         
        print("Model summary:", summary(model, input_size=(1, input_dim), verbose=0))
         #return model
     
    def train_model(self,loss_fcn = 'mae', epochs=1000, batch_size=128, learning_rate=1e-4, validation_split=0.2,
                      weight_decay=1e-5, patience=5, factor=0.5, early_stopping_patience = 12):


        # Data split (already good)
        train_size = int((1-validation_split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss, optimizer, and scheduler
        if loss_fcn == 'mae':
            criterion = nn.L1Loss()
        if loss_fcn == 'mse': 
            criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                      )

        # Early stopping parameters
        early_stopping_patience = early_stopping_patience
        best_val_loss = float('inf')
        epochs_no_improve = 0

        num_epochs = epochs

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.DEVICE)
                y_batch = y_batch.to(self.DEVICE)
                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            train_loss /= train_size

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:            
                    pred = self.model(x_batch)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item() * x_batch.size(0)
            val_loss /= val_size

            print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")

            # Step the scheduler
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:  # small threshold for improvement
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_model_state = self.model.state_dict()  # save best model weights
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)

            #return self.model
   

    

        
    def optuna_trial(self, ntrials = 3):
        def objective(trial):
            hidden_dims = trial.suggest_categorical("hidden_dims", [
                                                                    [512, 256],
                                                                    [512, 512, 256],
                                                                    [1024, 512, 256],
                                                                    [1024, 1024, 512, 256],
                                                                    [2048, 1024, 512],
                                                                    [2048, 1024, 512, 256],
                                                                    [2048, 2048, 1024, 1024, 512],
                                                                    [2048, 2048, 1024, 1024, 1024, 1024, 256],
                                                                    [2048, 2048, 2048, 1024, 1024, 1024, 1024, 512, 256],
                                                                    [2048, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 512, 256],

                                                                    [4096, 2048, 1024],
                                                                    [4096, 2048, 1024, 512],
                                                                     [4096, 2048, 1024, 512, 256],
                                                                    [4096, 2048, 1024, 512, 256, 128],
                                                                    [4096, 2048, 1024, 512, 256, 128, 64],
                                                                    [4096, 2048, 2048, 1024, 512, 256, 128, 64],
                                                                    [4096, 4096, 2048, 1024, 512, 256],
                                                                    [4096, 4096, 2048, 1024, 512, 256, 128],
                                                                    [4096, 4096, 2048, 2048, 1024, 512, 256, 128, 64],
                                                                    [4096, 4096, 4096, 2048, 1024, 512, 256, 128, 64, 64],  # 10 layers

                                                                    [2048, 1024, 512, 256, 128, 64],
                                                                    [2048, 1024, 512, 512, 256, 128, 64],
                                                                    [2048, 2048, 1024, 512, 256, 256, 128, 64],
                                                                    [2048, 2048, 2048, 1024, 512, 256, 256, 128, 64],
                                                                    [1024, 512, 256, 128, 64],
                                                                    [1024, 1024, 512, 256, 128, 64],
                                                                    [512, 512, 256, 128, 64],
                                                                    [512, 256, 128, 64],
                                                                ]
            )


            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)


            # Data/model setup
           
            self.build_model(
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                
            )

            # Training
            self.train_model(
            
                learning_rate=learning_rate,
                weight_decay=weight_decay,
         
            )
            mse, mae = self.plot_model_on_test()
            return mse

        # 🚀 Run Optuna Study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=ntrials)
         ##save as pickle file
        with open('optuna_study_results.pkl', 'wb') as f:
            pickle.dump(study.best_trial, f)
        return study.best_trial
     


    
    
    def plot_model_on_test(self, ax_title= 'Model Performance on Test Data', save_name = None):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test)
        y_pred = y_pred.detach().cpu().numpy()  # Move to CPU and convert to numpy array
        
        
        rc = {
            "font.family": "serif",
            "mathtext.fontset": "stix"
        }
        plt.rcParams.update(rc)
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=500)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].plot(self.y_test.numpy() ,y_pred, label='True Values', color='cornflowerblue', ls='', marker='.', markersize=3)
        ax[0].plot([0,1], [0,1], color='indianred', linestyle='--',alpha = 0.7)

        mse = np.mean((self.y_test.numpy() - y_pred) ** 2)
        mae = np.mean(np.abs(self.y_test.numpy() - y_pred))
        
        bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.5, edgecolor='lightgrey')
        ax[0].text(0.31, 0.85, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
                   transform=ax[0].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=bbox_props)

        ax[0].set_title(ax_title, fontsize=20)
        ax[0].set_xlabel('Sample Index (sorted by descending true value)', fontsize=15)
        ax[0].set_ylabel('Scaled Precip.', fontsize=15)
        ax[0].grid(True, linestyle='--', alpha=0.7, axis='both', color='white')
        ax[0].legend(fontsize=12, loc='upper left')
        ax[0].set_ylim(-0.1, 1.1)
        ax[0].set_xlim(-0.1, 1.1)

        residuals = self.y_test.numpy() - y_pred
        hist_range = [-1,1]
    
        ax[1].hist(residuals, range=hist_range, label='Residuals (truth - pred.)', bins=50, color='cornflowerblue',
                   alpha=0.5, histtype='stepfilled', edgecolor='dimgrey')
        ax[1].set_xlabel('Residuals', fontsize=15)
        ax[1].set_ylabel('Frequency', fontsize=15)
        ax[1].set_title('Residual Distribution', fontsize=20)
        ax[1].grid(True, linestyle='--', alpha=0.7, axis='y', color='white')
        ax[1].legend(fontsize=12, loc='upper left')
        ax[1].axvline(0, color='indianred', linestyle='--', alpha=0.7, linewidth=0.5)

        spine_args = ['top', 'right', 'left', 'bottom']

        for a in ax:
            a.set_facecolor('gainsboro')
            for spine in spine_args:
                a.spines[spine].set_visible(False)
            a.tick_params(axis='both', which='both', length=0)

        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name)

            return mse, mae





    def lrp_calc_and_plot(self, save_name=None, lag_title='LRP Attribution for FFNN Model (Lag = 0)'):

        rc = {
            "font.family": "serif",
            "mathtext.fontset": "stix"
        }
        plt.rcParams.update(rc)
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

        ##### below is prev code, testing the code above

        # fig_lrp, ax_lrp = plrp.plot_LRP(attr_sum)
        input_tensor = self.X_test.view(self.X_test.shape[0], -1).clone().detach().requires_grad_(True)

        # aplying rules to the model
        for layer in self.model.modules():
            for key, value in rules_dict.items():
                if isinstance(layer, key):
                    layer.rule = value
        

        

        lrp = LRP(self.model)

        attributions = lrp.attribute(input_tensor)
        attributions = attributions.detach().cpu().numpy()  # Convert to numpy array

        if self.OG_shape is not None:
            attr_sum = np.sum(attributions.reshape(self.OG_shape), axis=0)
        
        global_max = max(
            np.abs(attr_sum[0].max()),
            np.abs(attr_sum[1].max())
        )

        lrp_norm = attr_sum / global_max

        fig, axs = plt.subplots(ncols=2, 
                               nrows=1, 
                               figsize=(12,5), 
                               subplot_kw={'projection': ccrs.NorthPolarStereo() },
                               dpi = 300
        )


        axs[0].set_title('LRP Attribution for Temperature at 2m', fontsize=15)
        cbar = axs[0].pcolormesh(self.lon_for_plot,
                                self.lat_for_plot,
                                lrp_norm[0],
                                cmap='RdBu',
                                transform=ccrs.PlateCarree(),
                                vmin=-1, 
                                vmax=1

        )

        axs[1].set_title('LRP Attribution for Mean Sea Level Pressure', fontsize=15)
        axs[1].pcolormesh(self.lon_for_plot,
                                self.lat_for_plot,
                                lrp_norm[1],
                                cmap='RdBu',
                                transform=ccrs.PlateCarree(),
                                vmin=-1, 
                                vmax=1
        )
        # Add Colorbar to the plot
        fig.colorbar(cbar, 
                     ax=axs, 
                     orientation='horizontal', 
                     fraction=0.05, 
                     pad=0.04, 
                     label='LRP Attribution Value')
        
        for ax in axs.flatten():
            ax.coastlines()
            ax.gridlines(color='white', linestyle='--', alpha=1)
            ax.set_facecolor('gainsboro')
            ax.set_extent([-90, 30, 39, 90], ccrs.PlateCarree())
            for spine in ax.spines.values():
                spine.set_visible(False)        


        plt.suptitle(lag_title, fontsize=20)


        if save_name is not None:
            plt.savefig(save_name, dpi=300)


    def calc_mse(self):
        with torch.no_grad():
            y_pred = self.model(self.X_test)
        y_pred = y_pred.detach().cpu().numpy()
        mse = np.mean((self.y_test.numpy() - y_pred) ** 2)
        return mse










