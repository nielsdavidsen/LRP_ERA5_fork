import os
import sys
import torch
device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'

from model_hpc import Model_FFN
import time
t0 = time.time()


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



lag = 4
model = 0
# model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5')
model.load_data(sub_sampling=False, five_year_test=True)
model.lag_data(lag=lag)
model.prepare_data_for_tensorflow(test_size=0.1)
model.optuna_trial(ntrials=10)

## load dictionary with best hyperparameters, in current folder called 'best_trial.pkl'
import pickle
with open('best_trial.pkl', 'rb') as f:
    best_params = pickle.load(f)

lag = 4
model = 0
# model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5')
model.load_data(sub_sampling=False, five_year_test=True)
model.lag_data(lag=lag)
model.prepare_data_for_tensorflow(test_size=0.1)
apply_best_hyperparams(model, best_params)
model.plot_model_on_test(title=f'FFN model with lag {lag}', save_name='FFN_lag_' + str(lag) + '.png')
model.lrp_calc_and_plot(save_name='FFN_lag_' + str(lag) + '_LRP.png')

t1 = time.time()


##write the above to file
with open('execution_time.txt', 'w') as f:
    f.write(f"Script finished in {t1 - t0:.2f} seconds\n")
    f.write(f"Total time: {t1 - t0:.2f/60} minutes\n")
    f.write(f"Total time: {t1 - t0:.2f/3600} hours\n")


