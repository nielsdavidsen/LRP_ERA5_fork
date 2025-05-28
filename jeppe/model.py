import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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

class gamma_activation(nn.Module):
    def forward(self, x):
        return torch.gamma_dist.cdf(x, a=1.0, scale=1.0)  # Example parameters, adjust as needed

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

     def load_data(self, sub_sampling = False, sub_sample_dim = 4):
         '''
         the sorting key is wrong, as the dmi format is different.
         '''
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
        
             
     def prepare_data_for_tensorflow(self, test_size = 250 , print_shapes = True):
         ##take 250 random samples from X to test the model
         random_indices = np.random.choice(self.X.shape[0], size=test_size, replace=False)
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

         self.X_test = torch.from_numpy(X_test).view(X_test.shape[0], -1)
         self.y_test = torch.from_numpy(y_test).view(y_test.shape[0], 1)

         #return dataset, X_test, y_test, 
    
     
     def build_model(self, dropout_rate=0.1,
                     hidden_dims=[2048, 2048,1024, 1024,1024, 1024, 256],
                     Sigmoid_output = False,
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



                layers.append(nn.Sigmoid()) if Sigmoid_output else None
                layers.append(nn.ReLU()) if ReLU_output else None
                ##exponential activation
                layers.append(gamma_activation()) if gamma_output else None


                # Use Sequential for compactness
                self.model = nn.Sequential(*layers)

             def forward(self, x):
                 return self.model(x)

        # Model setup
         input_dim = self.X.shape[1] * self.X.shape[2] * self.X.shape[3]

         model = FFNN(input_dim)
         self.model = model
         print("Model summary:", summary(model, input_size=(1, input_dim), verbose=0))
         #return model
     
     def train_model(self, epochs=100, batch_size=128, learning_rate=1e-4, validation_split=0.2,
                      weight_decay=1e-5, patience=5, factor=0.5, early_stopping_patience = 5):


        # Data split (already good)
        train_size = int((1-validation_split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss, optimizer, and scheduler
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
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)

            #return self.model
     def plot_model_on_test(self):
         plt.figure(figsize=(10, 5))
         plt.plot(self.y_test.numpy(), label='True Values', color='blue')
         plt.plot(self.model(self.X_test).detach().numpy(), label='Predicted Values', color='red')   
         mse = np.mean((self.y_test.numpy() - self.model(self.X_test).detach().numpy()) ** 2)
         plt.title(f'Model Predictions vs True Values (MSE: {mse:.4f})')

        



