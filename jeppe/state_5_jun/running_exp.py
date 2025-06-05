import os
import sys
import ast

import torch

device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'
print('Using device:', device)



with open("best_params_dic.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]  # skip empty lines

params = ast.literal_eval(lines[0])
print("Best hyperparameters found:")
print(params)


from model_hpc_NEW_PLOT import Model_FFN

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

for lag in [0,4,20]:
    model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
    if lag !=0:
        model.lag_data(lag=lag)
    model.load_data(sub_sampling=False, five_year_test=False)
    model.prepare_data_for_tensorflow(test_size=0.1)
    apply_best_hyperparams(model, params)
    model.plot_model_on_test(ax_title=f'FFNN model with lag {lag}', save_name='ffn_lag_' + str(lag) + '.png')
    model.lrp_calc_and_plot(save_name='ffn_lag_' + str(lag) + '_lrp.png', title = f'FFNN model with lag {lag} LRP')



