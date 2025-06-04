import os
import sys
import torch
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
print(device)
from model_hpc import Model_FFN
import time
t0 = time.time()


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
    )

    model.train_model(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=1000,  # You can make this an argument if needed
        batch_size=128  # Likewise
    )



lag = 0
model = 0
model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
model.load_data(sub_sampling=False, five_year_test=False)
model.lag_data(lag=lag)
model.prepare_data_for_tensorflow(test_size=0.1)
model.optuna_trial(ntrials=200)

## load dictionary with best hyperparameters, in current folder called 'best_trial.pkl'
import pickle
with open('optuna_study_results.pkl', 'rb') as f:
    best_params = pickle.load(f)

lag = 0
model = 0
model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
model.load_data(sub_sampling=False, five_year_test=False)
model.lag_data(lag=lag)
model.prepare_data_for_tensorflow(test_size=0.1)
apply_best_hyperparams(model, best_params.params)
model.plot_model_on_test(title=f'FFN model with lag {lag}', save_name='FFN_lag_' + str(lag) + '.png')
model.lrp_calc_and_plot(save_name='FFN_lag_' + str(lag) + '_LRP.png')

t1 = time.time()


with open('execution_time.txt', 'w') as f:
    f.write(f"Script finished in {t1 - t0:.2f} seconds\n")
    f.write(f"Total time: {(t1 - t0)/60:.2f} minutes\n")
    f.write(f"Total time: {(t1 - t0)/3600:.2f} hours\n")



