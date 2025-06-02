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
         #return model
     
     def train_model(self, loss_fcn = 'mae',epochs=100, batch_size=128, learning_rate=1e-4, validation_split=0.2,
                      weight_decay=1e-5, patience=5, factor=0.5, early_stopping_patience = 5):


        # Data split (already good)
        train_size = int((1-validation_split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss, optimizer, and scheduler
        if loss_fcn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fcn == 'mae':
            criterion = nn.L1Loss()
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
     def plot_model_on_test(self, safe=False, save_name = 'model_predictions.png'):
         plt.figure(figsize=(10, 5))
         plt.plot(self.y_test.numpy(), label='True Values', color='blue')
         plt.plot(self.model(self.X_test).detach().numpy(), label='Predicted Values', color='red')   
         mse = np.mean((self.y_test.numpy() - self.model(self.X_test).detach().numpy()) ** 2)
         mae = np.mean(np.abs(self.y_test.numpy() - self.model(self.X_test).detach().numpy()))
         plt.title(f'Model Predictions vs True Values (MSE: {mse:.4f})')
         if safe:
            plt.savefig(save_name)
         return mse, mae

        
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
            return mae

        # 🚀 Run Optuna Study
         study = optuna.create_study(direction="minimize")
         study.optimize(objective, n_trials=ntrials)
         return study.best_trial



                
##################################################################################################











class Model_CNN:  # Changed class name to reflect CNN architecture
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

    # Assume sample_data is already defined/imported above

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
         


    def lag_data(self, lag):
            self.lag = lag
            self.X = self.X[:-lag]            # Shape: (1827 - lag, 2, 121, 409)
            self.target = self.target[lag:]
            print("Lagged X shape:", self.X.shape)
            print("Lagged target shape:", self.target.shape)

    def prepare_data_for_tensorflow(self,
                                test_size=.1,
                                print_shapes=True,
                                sub_sampling=False,
                                sub_sample_dim=4):
    
    # 1) Random split indices
        n = self.X.shape[0]
        test_size = int(test_size * n)
        random_indices = np.random.choice(n, size=test_size, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[random_indices] = True

    # 2) Split raw arrays
        X_test = self.X[mask]
        X_train = self.X[~mask]
        y_test = self.target[mask]
        y_train = self.target[~mask]

    # 3) Scale targets
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

    # 4) Optional subsampling inside each split
        if sub_sampling:
            print("Applying spatial subsampling to train/test sets...")
            # Subsample spatial patches and repeat targets
            X_train_da = sample_data(X_train, n=sub_sample_dim)
            X_test_da = sample_data(X_test, n=sub_sample_dim)

            X_train = X_train_da.values
            X_test = X_test_da.values

        # Repeat target values for each patch
            y_train = np.repeat(y_train, sub_sample_dim)
            y_test = np.repeat(y_test, sub_sample_dim)

    # 5) Optionally print shapes
        if print_shapes:
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"X_test shape:  {X_test.shape}")
            print(f"y_test shape:  {y_test.shape}")

    # 6) Convert to tensors and dataset
        X_tensor = torch.from_numpy(X_train).float()
        y_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
        self.dataset = TensorDataset(X_tensor, y_tensor)

    # 7) Store test tensors for evaluation
        self.X_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    def build_model(self, dropout_rate=0.1, conv_layers=None, fc_layers=None):
        # 1) Grab actual input dims from your dataset tensor self.X of shape (N, C, H, W)
        input_channels = self.X.shape[1]
        H = self.X.shape[2]
        W = self.X.shape[3]

        # 2) Define the CNN class with dynamic flattening
        class FlexibleCNN(nn.Module):
            def __init__(self, input_channels, conv_layers, fc_layers, drop_rate):
                super().__init__()

                # Build convolutional blocks
                blocks = []
                in_ch = input_channels
                for out_ch, k_size, pool_size in conv_layers:
                    blocks += [
                        nn.Conv2d(in_ch, out_ch, k_size, padding=k_size // 2),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.MaxPool2d(pool_size),
                        nn.Dropout2d(drop_rate)
                    ]
                    in_ch = out_ch
                self.conv_layers = nn.Sequential(*blocks)

                # 3) Flatten layer
                self.flatten = nn.Flatten()

                # 4) Dummy forward to infer flat‐feature size
                with torch.no_grad():
                    dummy = torch.zeros(1, input_channels, H, W)
                    N = self.flatten(self.conv_layers(dummy)).shape[1]

                # 5) Build FC head dynamically
                fc_blocks = []
                in_features = N
                for out_features in fc_layers:
                    fc_blocks += [
                        nn.Linear(in_features, out_features),
                        nn.BatchNorm1d(out_features),
                        nn.ReLU(),
                        nn.Dropout(drop_rate)
                    ]
                    in_features = out_features
                # Final output (regression → 1)
                fc_blocks.append(nn.Linear(in_features, 1))
                self.fc_layers = nn.Sequential(*fc_blocks)

            def forward(self, x):
                x = self.conv_layers(x)
                x = self.flatten(x)
                return self.fc_layers(x)

        # 6) Instantiate & attach to self
        model = FlexibleCNN(input_channels, conv_layers, fc_layers, dropout_rate)
        self.model = model

        # 7) (Optional) Print a summary
        print("Model summary:",
              summary(model,
                      input_size=(1, input_channels, H, W),
                      verbose=0))

    def optuna_trial(self, ntrials=3):
        def objective(trial):
            # Search space for CNN architecture
            n_conv_layers = trial.suggest_int("n_conv_layers", 2, 5)

            # Build conv layer configurations
            conv_layers = []
            in_filters = 32  # Starting number of filters

            for i in range(n_conv_layers):
                filters = trial.suggest_int(f"conv_filters_{i}",
                                            in_filters,
                                            min(512, in_filters * 2))
                kernel = trial.suggest_int(f"kernel_size_{i}", 3, 7, step=2)
                pool = trial.suggest_int(f"pool_size_{i}", 2, 3)
                conv_layers.append((filters, kernel, pool))
                in_filters = filters

            # FC layer configurations
            n_fc_layers = trial.suggest_int("n_fc_layers", 1, 4)
            fc_layers = []

            for i in range(n_fc_layers):
                fc_size = trial.suggest_int(f"fc_size_{i}",
                                            64,
                                            512,
                                            step=64)
                fc_layers.append(fc_size)

            # Other hyperparameters
            dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            # Build and train model
            self.build_model(
                dropout_rate=dropout_rate,
                conv_layers=conv_layers,
                fc_layers=fc_layers
            )

            self.train_model(
                epochs=100,
                batch_size=128,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=10,
                factor=0.5,
                early_stopping_patience=7
            )

            mse,mae = self.plot_model_on_test()
            return mse, mae

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=ntrials)
        return study.best_trial

    def train_model(self, loss_fcn = 'mse',epochs=100, batch_size=128, learning_rate=1e-4, validation_split=0.2,
                      weight_decay=1e-5, patience=5, factor=0.5, early_stopping_patience = 5):


        # Data split (already good)
        train_size = int((1-validation_split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss, optimizer, and scheduler
        if loss_fcn == 'mse':
            criterion = nn.MSELoss()
        if loss_fcn == 'mae':
            criterion = nn.L1Loss()
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


    def plot_model_on_test(self, safe=False, save_name = 'model_predictions.png'):
         plt.figure(figsize=(10, 5))
         plt.plot(self.y_test.numpy(), label='True Values', color='blue')
         plt.plot(self.model(self.X_test).detach().numpy(), label='Predicted Values', color='red')   
         mse = np.mean((self.y_test.numpy() - self.model(self.X_test).detach().numpy()) ** 2)
         mae = np.mean(np.abs(self.y_test.numpy() - self.model(self.X_test).detach().numpy()))
         plt.title(f'Model Predictions vs True Values (MSE: {mse:.4f})')
         if safe:
            plt.savefig(save_name)
         return mse, mae



                



