import numpy as np
import matplotlib.pyplot as plt
import os
import optuna
import torch
import torch.nn as nn
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import xarray as xr

def standard_scale_day(group, eps=1e-6):
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

    # Reshaping into (t, v, x, y) format
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

# def prepare_data_for_tensorflow(self, test_size=250, print_shapes=True):
#     s = 42
#     print(f'SEED: {s}')
#     np.random.seed(s)  # For reproducibility
#     random_indices = np.random.choice(self.X.shape[0], size=test_size, replace=False)
#     mask = np.zeros(self.X.shape[0], dtype=bool)
#     mask[random_indices] = True
#
#     X_test = self.X[mask]
#     X_train = self.X[~mask]
#     scaler = MinMaxScaler()
#
#     y_test = scaler.fit_transform(self.target[mask].reshape(-1, 1))
#     y_train = scaler.transform(self.target[~mask].reshape(-1, 1))
#
#     if print_shapes:
#         print("X_train shape:", X_train.shape)
#         print("y_train shape:", y_train.shape)
#         print("X_test shape:", X_test.shape)
#         print("y_test shape:", y_test.shape)
#
#     X_tensor = torch.from_numpy(X_train).float()
#     y_tensor = torch.from_numpy(y_train).float()
#
#     # Create dataset without flattening X
#     dataset = TensorDataset(X_tensor, y_tensor)
#     self.dataset = dataset
#
#     self.X_test = torch.from_numpy(X_test).float()
#     self.y_test = torch.from_numpy(y_test).float()


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

        # --- NEW: Export to ONNX ---

    def export_to_onnx_from_wrapper(wrapper_model, path="flexible_cnn.onnx", input_shape=(1, 2, 120, 408)):
        dummy_input = torch.randn(input_shape)
        wrapper_model.model.eval()
        torch.onnx.export(
            wrapper_model.model,
            dummy_input,
            path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"ONNX model exported to {path}")

    def visualize_with_torchview_from_wrapper(wrapper_model, input_shape=(1, 2, 120, 408), output_file="model_graph"):
        try:
            from torchview import draw_graph
        except ImportError:
            raise ImportError("torchview is not installed. Run `pip install torchview`")

        graph = draw_graph(wrapper_model.model, input_size=input_shape, expand_nested=True)
        graph.visual_graph.render(output_file, format="png", cleanup=True)
        print(f"Model visualization saved as {output_file}.png")

    def load_data(self, sub_sampling=False, sub_sample_dim=4):
        ################################MSL##########################################
        input_files_msl = [f for f in os.listdir(self.data_path) if
                           f.endswith('.nc') and 'mean_sea_level_pressure' in f]
        input_files_msl.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        file_paths_msl = [os.path.join(self.data_path, fname) for fname in input_files_msl]
        ds_msl = xr.open_mfdataset(file_paths_msl, combine='by_coords').sel(
            latitude=slice(self.latitude_range[0], self.latitude_range[1]),
            longitude=slice(self.longitude_range[0], self.longitude_range[1])
        )

        doy = ds_msl['valid_time'].dt.dayofyear
        ds_msl = ds_msl.assign_coords(day_of_year=doy)
        msl_stand = ds_msl['msl'].groupby('day_of_year').map(standard_scale_day)
        ds_msl['msl_stand'] = msl_stand
        msl_input = ds_msl.msl_stand.values

        if np.isnan(msl_input).any():
            raise ValueError("NaN values found in the input data. Please check the dataset for missing values.")

        print("Msl input shape:", msl_input.shape)
        print("input_file_names:", input_files_msl)

        ##################################T2m#############################################
        input_files_t2m = [f for f in os.listdir(self.data_path) if f.endswith('.nc') and '2m' in f]
        input_files_t2m.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        print("input_file_names_t2m:", input_files_t2m)
        file_paths_t2m = [os.path.join(self.data_path, fname) for fname in input_files_t2m]
        ds_t2m = xr.open_mfdataset(file_paths_t2m, combine='by_coords').sel(
            latitude=slice(self.latitude_range[0], self.latitude_range[1]),
            longitude=slice(self.longitude_range[0], self.longitude_range[1])
        )

        doy_t2m = ds_t2m['valid_time'].dt.dayofyear
        ds_t2m = ds_t2m.assign_coords(day_of_year=doy_t2m)
        t2m_stand = ds_t2m['t2m'].groupby('day_of_year').map(standard_scale_day)
        ds_t2m['t2m_stand'] = t2m_stand
        t2m_input = ds_t2m.t2m_stand.values

        if np.isnan(t2m_input).any():
            raise ValueError("NaN values found in the input data. Please check the dataset for missing values.")

        print("T2m input shape:", t2m_input.shape)
        print("input_file_names_t2m:", input_files_t2m)

        #################### precipitation ##########################################
        target_prec = [f for f in os.listdir(self.data_path) if f.endswith('.nc') and 'total' in f]
        target_prec.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        file_paths_prec = [os.path.join(self.data_path, fname) for fname in target_prec]
        ds_prec = xr.open_mfdataset(file_paths_prec, combine='by_coords')
        ds_prec = ds_prec.mean(dim=['longitude', 'latitude'])
        prec_target = ds_prec.tp.values

        if np.isnan(prec_target).any():
            raise ValueError("NaN values found in the target data. Please check the dataset for missing values.")

        print("Precipitation target shape:", prec_target.shape)
        print("target_file_names:", target_prec)
        X = np.stack([t2m_input, msl_input], axis=1)

        if sub_sampling:
            print("Sub-sampling data...")
            X_c = sample_data(X, n=sub_sample_dim)
            print("Sub-sampled X shape:", X_c.shape)
            prec_target_concat = np.repeat(prec_target, 4, axis=0)
            print("Concatenated precipitation target shape:", prec_target_concat.shape)
            self.X = X_c.values
            self.target = prec_target_concat
        else:
            print("No sub-sampling applied.")
            print("X shape:", X.shape)
            print("Precipitation target shape:", prec_target.shape)
            self.X = X
            self.target = prec_target


    def prepare_data_for_tensorflow(self, test_size=250, print_shapes=True):
        # CHANGE: Removed flattening of input data since CNNs work with spatial dimensions
        s = 40
        print(f'SEED: {s}')
        np.random.seed(s)  # For reproducibility
        random_indices = np.random.choice(self.X.shape[0], size=test_size, replace=False)
        mask = np.zeros(self.X.shape[0], dtype=bool)
        mask[random_indices] = True

        X_test = self.X[mask]
        X_train = self.X[~mask]
        scaler = MinMaxScaler()

        y_test = scaler.fit_transform(self.target[mask].reshape(-1, 1))
        y_train = scaler.transform(self.target[~mask].reshape(-1, 1))

        if print_shapes:
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)

        # CHANGE: Keep spatial dimensions intact for CNN
        X_tensor = torch.from_numpy(X_train).float()
        y_tensor = torch.from_numpy(y_train).float()

        # Create dataset without flattening
        dataset = TensorDataset(X_tensor, y_tensor)
        self.dataset = dataset

        self.X_test = torch.from_numpy(X_test).float()
        self.y_test = torch.from_numpy(y_test).float()

    def build_model(self, dropout_rate=0.1, conv_layers=None, fc_layers=None,print_summary=True,sigmoid_output=False):
        # CHANGE: Completely new flexible CNN architecture
        if conv_layers is None:
            # Default CNN architecture if none provided
            conv_layers = [(64, 3, 2), (128, 3, 2), (256, 3, 2)]  # (filters, kernel_size, pool_size)
        if fc_layers is None:
            # Default fully connected layers if none provided
            fc_layers = [512, 128]

        class FlexibleCNN(nn.Module):
            def __init__(self, input_channels, conv_layers, fc_layers, dropout_rate,sigmoid_output=sigmoid_output):
                super().__init__()

                # Dynamic convolutional layers construction
                conv_blocks = []
                in_channels = input_channels
                h, w = 121, 409  # Initial spatial dimensions from your data

                # Build convolutional blocks dynamically
                for filters, k_size, pool_size in conv_layers:
                    if h < pool_size or w < pool_size:
                        raise ValueError(f"Pooling size {pool_size} is too large for current dimensions ({h}, {w}).")

                    conv_blocks.extend([
                        nn.Conv2d(in_channels, filters, k_size, padding=k_size // 2),
                        nn.BatchNorm2d(filters),
                        nn.ReLU(),
                        nn.MaxPool2d(pool_size),
                        nn.Dropout2d(dropout_rate)
                    ])
                    in_channels = filters
                    h = h // pool_size
                    w = w // pool_size

                self.conv_layers = nn.Sequential(*conv_blocks)

                # Calculate size after convolutions for FC layers
                self.flat_features = in_channels * h * w

                # Build fully connected layers dynamically
                fc_blocks = []
                in_features = self.flat_features

                for out_features in fc_layers:
                    fc_blocks.extend([
                        nn.Linear(in_features, out_features),
                        nn.BatchNorm1d(out_features),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    in_features = out_features

                # Final output layer
                fc_blocks.append(nn.Linear(in_features, 1))
                if sigmoid_output:
                    fc_blocks.append(nn.Sigmoid())

                self.fc_layers = nn.Sequential(*fc_blocks)

            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(-1, self.flat_features)
                x = self.fc_layers(x)
                return x

        # Model setup
        input_channels = self.X.shape[1]  # Number of input channels (2 in your case)
        model = FlexibleCNN(input_channels, conv_layers, fc_layers, dropout_rate)
        self.model = model
        if print_summary:
            print("Model summary:",
              summary(model, input_size=(1, input_channels, self.X.shape[2], self.X.shape[3]), verbose=0))

    def optuna_trial(self, ntrials=3):
        # CHANGE: Modified to search CNN hyperparameters
        def objective(trial):
            # Search space for CNN architecture
            n_conv_layers = trial.suggest_int("n_conv_layers", 2, 5)

            # Build conv layer configurations
            conv_layers = []
            in_filters = 32  # Starting number of filters

            for i in range(n_conv_layers):
                filters = trial.suggest_int(f"conv_filters_{i}", in_filters, min(256, in_filters * 2))  # Reduced max filters
                kernel = trial.suggest_int(f"kernel_size_{i}", 3, 5, step=2)  # Reduced kernel size range
                pool = trial.suggest_int(f"pool_size_{i}", 2, 2)  # Fixed pooling size
                conv_layers.append((filters, kernel, pool))
                in_filters = filters  # Ensure filters increase or remain constant

            # FC layer configurations
            n_fc_layers = trial.suggest_int("n_fc_layers", 1, 4)
            fc_layers = []

            for i in range(n_fc_layers):
                fc_size = trial.suggest_int(f"fc_size_{i}", 32, min(128, fc_layers[-1] if fc_layers else 128), step=32)  # Reduced max neurons
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

            mse, mae=  self.evaluate()
            return mse

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=ntrials)
        return study.best_trial

    def train_model(self, epochs=100, batch_size=128, learning_rate=1e-4, validation_split=0.2,
                      weight_decay=1e-5, patience=5, factor=0.5, early_stopping_patience = 5,loss_function = 'MAE'):


        # Data split (already good)
        train_size = int((1-validation_split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss, optimizer, and scheduler
        if loss_function == 'MAE':
            criterion = nn.L1Loss()
        elif loss_function == 'MSE':
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

    def plot_model_on_test(self, ax_title='Model Performance on Test Data', save_name=None):
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

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=1200)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].plot(self.y_test.numpy(), y_pred, color='cornflowerblue', ls='', marker='.',
                   markersize=3)
        ax[0].plot([0, 1], [0, 1], color='indianred', linestyle='--', alpha=0.7)

        mse = np.mean((self.y_test.numpy() - y_pred) ** 2)
        mae = np.mean(np.abs(self.y_test.numpy() - y_pred))

        bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.5, edgecolor='lightgrey')
        ax[0].text(0.31, 0.85, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
                   transform=ax[0].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=bbox_props)

        ax[0].set_title(ax_title, fontsize=20)
        ax[0].set_xlabel('True Scaled Precip.', fontsize=15)
        ax[0].set_ylabel('Predicted Scaled Precip.', fontsize=15)
        ax[0].grid(True, linestyle='--', alpha=0.7, axis='both', color='white')
        ax[0].set_ylim(-0.1, 1.1)
        ax[0].set_xlim(-0.1, 1.1)

        residuals = self.y_test.numpy() - y_pred
        hist_range = [-1, 1]

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

    def evaluate(self):
        self.model.eval()
        mse = np.mean((self.y_test.numpy() - y_pred) ** 2)
        mae = np.mean(np.abs(self.y_test.numpy() - y_pred))
        return mse, mae
