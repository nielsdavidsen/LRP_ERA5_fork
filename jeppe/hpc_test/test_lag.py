import os
import sys

import torch

device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

from model import Model_FFN
lags =[1,2,5,20]

def apply_best_hyperparams(model, best_params):
    # Unpack parameters
    hidden_dims = best_params["hidden_dims"]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    output_activation = best_params.get("output_activation", "none")

    # Map activation string to boolean flags
    sigmoid_output = output_activation == "sigmoid"
    relu_output = output_activation == "relu"

    # Build and train the model
    model.build_model(
        dropout_rate=dropout_rate,
        hidden_dims=hidden_dims,
        Sigmoid_output=sigmoid_output,
        ReLU_output=relu_output,
    )

    model.train_model(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=1000,  # You can make this an argument if needed
        batch_size=128  # Likewise
    )



lag = 0
print("_______________________________________________________________________________________________________________________________________")
model = 0
#model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5')
model.load_data(sub_sampling=False)
#model.lag_data(lag=lag)
model.prepare_data_for_tensorflow(test_size=0.15)
model.build_model(dropout_rate=0.1, hidden_dims=[2048, 1024, 512, 256], Sigmoid_output=True,)
model.train_model(loss_fcn='mae')
model.plot_model_on_test(title=f'FFN model with lag {lag}', save_name='better_ffn_lag_' + str(lag) + '.png')
print(model.X_test.shape)
print(f"plotting for lag {lag}")
model.lrp_calc_and_plot(save_name='better_ffn_lag_' + str(lag) + '_lrp.png')
for lag in lags:
    print("_______________________________________________________________________________________________________________________________________")
    model = 0
    #model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
    model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5')
    model.load_data(sub_sampling=False)
    model.lag_data(lag=lag)
    model.prepare_data_for_tensorflow(test_size=0.15)
    model.build_model(dropout_rate=0.1, hidden_dims=[2048, 1024, 512, 256], Sigmoid_output=True,)
    model.train_model(loss_fcn='mae')
    model.plot_model_on_test(title=f'FFN model with lag {lag}', save_name='better_ffn_lag_' + str(lag) + '.png')
    print(f"plotting for lag {lag}")
    model.lrp_calc_and_plot(save_name='better_ffn_lag_' + str(lag) + '_lrp.png')



