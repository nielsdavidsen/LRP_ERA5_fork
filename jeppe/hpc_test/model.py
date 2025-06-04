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
     def __init__(self, data_path, latitude_range=(70, 40.25), longitude_range=(-80, 21.75)):
        self.data_path = data_path
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
        self.OG_shape = None
        self.lat_for_plot = None
        self.lon_for_plot = None

     def load_data(self, sub_sampling = False, sub_sample_dim = 4):
         '''
         the sorting key is wrong, as the dmi format is different.
         '''
         print("ipsaeigpasrk")
         ################################MSL##########################################
         input_files_msl = [f for f in os.listdir(self.data_path) if f.endswith('.nc') and 'mean_sea_level_pressure' in f]
         input_files_msl.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

         ##merge the files from the list
         file_paths_msl = [os.path.join(self.data_path, fname) for fname in input_files_msl]

        # Then open all files with xarray

         ds_msl = xr.open_mfdataset(file_paths_msl, combine='by_coords').sel(
                        latitude=slice(self.latitude_range[0], self.latitude_range[1]),
                        longitude=slice(self.longitude_range[0], self.longitude_range[1])

        )
         self.lat_for_plot = ds_msl.latitude.values
         self.lon_for_plot = ds_msl.longitude.values



         doy = ds_msl['valid_time'].dt.dayofyear
         ds_msl = ds_msl.assign_coords(day_of_year=doy) ## 28 feb is 366. i have 2 of these in the 5 years.
         msl_stand = ds_msl['msl'].groupby('day_of_year').map(standard_scale_day) ## scale the data
         ds_msl['msl_stand'] = msl_stand
         msl_input = ds_msl.msl_stand.values

    
         if np.isnan(msl_input).any():
                raise ValueError("NaN values found in the input data. Please check the dataset for missing values.")

         print("Msl input shape:", msl_input.shape)
         print("input_file_names:", input_files_msl)


        ##################################T2m#############################################


         input_files_t850 = [f for f in os.listdir(self.data_path) if f.endswith('.nc') and '850' in f]
         input_files_t850.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

         file_paths_t850 = [os.path.join(self.data_path, fname) for fname in input_files_t850]
         ds_t850 = xr.open_mfdataset(file_paths_t850, combine='by_coords').isel(pressure_level=0).sel(
    latitude=slice(self.latitude_range[0], self.latitude_range[1]),
    longitude=slice(self.longitude_range[0], self.longitude_range[1])
)
         doy_t850 = ds_t850['valid_time'].dt.dayofyear
         ds_t850 = ds_t850.assign_coords(day_of_year=doy_t850)
         t850_stand = ds_t850['t'].groupby('day_of_year').map(standard_scale_day)
         ds_t850['t850_stand'] = t850_stand
         t850_input = ds_t850.t850_stand.values
         if np.isnan(t850_input).any():
                raise ValueError("NaN values found in the input data. Please check the dataset for missing values.")

         print("T850 input shape:", t850_input.shape)
         print("input_file_names_t850:", input_files_t850)
         #################### precipitation ##########################################
         target_prec = [f for f in os.listdir(self.data_path) if f.endswith('.nc') and 'total' in f]
         target_prec.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
         file_paths_prec = [os.path.join(self.data_path, fname) for fname in target_prec]
         ds_prec = xr.open_mfdataset(file_paths_prec, combine='by_coords')
         ds_prec = ds_prec.mean(dim= ['longitude', 'latitude'])
       
      
         prec_target = ds_prec.tp.values

             
         if np.isnan(prec_target).any():
                raise ValueError("NaN values found in the target data. Please check the dataset for missing values.")
         
         print("Precipitation target shape:", prec_target.shape)
         print("target_file_names:", target_prec)
         X = np.stack([t850_input, msl_input], axis=1)
        
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


         X = X_tensor.view(X_train.shape[0], -1)
         y = y_tensor.view(y_train.shape[0], 1)

        # Create dataset
         dataset = TensorDataset(X, y)
         self.dataset = dataset
         self.OG_shape = X_test.shape
         self.X_test = torch.from_numpy(X_test).view(X_test.shape[0], -1)
         self.y_test = torch.from_numpy(y_test).view(y_test.shape[0], 1)

         #return dataset, X_test, y_test, 
    
     
     def build_model(self, dropout_rate=0.1,
                     hidden_dims=[2048, 2048,1024, 1024,1024, 1024, 256],
                     Sigmoid_output = False,
                     ReLU_output = False,
                     gamma_output = False):
         class FFNN(nn.Module):
             def __init__(self, input_dim, dropout_rate=dropout_rate):
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
                    layers.append(nn.ReLU()) 
                ##exponential activation
                #layers.append(gamma_activation()) if gamma_output else None


                # Use Sequential for compactness
                self.model = nn.Sequential(*layers)

             def forward(self, x):
                 return self.model(x)

        # Model setup
         input_dim = self.X.shape[1] * self.X.shape[2] * self.X.shape[3]

         model = FFNN(input_dim)

         self.model = model

         print("Model summary:", summary(model, input_size=(1, input_dim), verbose=0))

     def train_model(self,loss_fcn = 'mae', epochs=100, batch_size=128, learning_rate=1e-4, validation_split=0.2,
                      weight_decay=1e-5, patience=5, factor=0.5, early_stopping_patience = 5):


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
            # if self.best_model_state is not None:
            #     self.model.load_state_dict(self.best_model_state)

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
])


            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
            activation_choice = trial.suggest_categorical("output_activation", ["none", "sigmoid", "relu"])

            sigmoid_out = activation_choice == "sigmoid"
            relu_out = activation_choice == "relu"

            # Data/model setup
           
            self.build_model(
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                Sigmoid_output=sigmoid_out,
                ReLU_output=relu_out
            )

            # Training
            self.train_model(
                epochs=100,
                batch_size=128,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=10,
                factor=0.5,
                early_stopping_patience=7
            )
            mse, mae = self.plot_model_on_test()
            return mse

        # 🚀 Run Optuna Study
         study = optuna.create_study(direction="minimize")
         study.optimize(objective, n_trials=ntrials)
         ##save the study.best_trial wiht pickle
         import pickle
         with open('best_trial.pkl', 'wb') as f:
            pickle.dump(study.best_trial, f)
         return study.best_trial
     
     def plot_model_on_test(self, title = 'Model Performance on Test Data', save_name = None):
         plt.figure(figsize=(10, 5))
         plt.plot(self.y_test.numpy(), label='True Values', color='blue')
         plt.plot(self.model(self.X_test).detach().numpy(), label='Predicted Values', color='red')   
         mse = np.mean((self.y_test.numpy() - self.model(self.X_test).detach().numpy()) ** 2)
         mae = np.mean(np.abs(self.y_test.numpy() - self.model(self.X_test).detach().numpy()))
         title = f"{title} - MSE: {mse:.4f}, MAE: {mae:.4f}"
         plt.title(title)
         if save_name is not None:
            plt.savefig(save_name)

         return mse, mae
     
     
     def lrp_calc_and_plot(self, save_name = None, title = 'LRP Attribution for FFNN Model'):

    

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
                               figsize=(15,10), 
                               subplot_kw={'projection': ccrs.NorthPolarStereo() },
                               dpi = 300
        )
        
        for ax in axs.flatten():
            ax.coastlines()
            ax.gridlines()
            ax.set_extent([-90, 30, 39, 90], ccrs.PlateCarree()
        )

        axs[0].set_title('LRP Attribution for 850hPa Temperature')
        cbar = axs[0].pcolormesh(self.lon_for_plot,
                                self.lat_for_plot,
                                lrp_norm[0],
                                cmap='coolwarm',
                                transform=ccrs.PlateCarree(),
                                vmin=-1, vmax=1
        )

        axs[1].set_title('LRP Attribution for Mean Sea Level Pressure')
        axs[1].pcolormesh(self.lon_for_plot,
                                self.lat_for_plot,
                                lrp_norm[1],
                                cmap='coolwarm',
                                transform=ccrs.PlateCarree(),
                                vmin=-1, vmax=1
        )

        # Add Colorbar to the plot
        fig.colorbar(cbar, 
                     ax=axs, 
                     orientation='horizontal', 
                     fraction=0.02, 
                     pad=0.04, 
                     label='LRP Attribution Value')
        plt.suptitle('LRP Attribution for FFNN Model', fontsize=16)
        if save_name is not None:
            plt.savefig(save_name, dpi=300)







