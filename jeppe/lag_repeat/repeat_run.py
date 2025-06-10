import os
import sys
import ast

import torch

lags = [0,1,2,3,4,5]
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
print('Using device:', device)



with open("best_params_lag_3.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]  # skip empty lines

params = ast.literal_eval(lines[0])
print("Best hyperparameters found:")
print(params)

from model_hpc_New_REPEAT import Model_FFN
def apply_best_hyperparams(model, best_params):
    # Unpack parameters
    hidden_dims = best_params["hidden_dims"]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]


    # Build and train the model
    model.build_model(
        dropout_rate=dropout_rate,
        hidden_dims=hidden_dims,
        Sigmoid_output=True,
    
    )

    model.train_model(
        loss_fcn='mae',
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    
    )
for lag in lags:
    ##write to .txt
    with open("mse.txt", "a") as f:
        f.write(f"lag: {lag}\n")

    for _ in range(5):
        model = Model_FFN(DEVICE=device)
        model.load_data(sub_sampling=False, five_year_test=False)
        if lag != 0:
            model.lag_data(lag=lag)
        model.prepare_data_for_tensorflow(test_size=0.05)
        apply_best_hyperparams(model, params)
        mse = model.calc_mse()
        with open("mse.txt", "a") as f:
            f.write(f"mse: {mse}\n")

   

