import os
import sys
import torch
from model import Model_FFN


device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

lag = 4
model = 0
# model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5', DEVICE=device)
model = Model_FFN(data_path='/Users/jeppegrejspetersen/Code/Final_project_AppML/era5')
model.load_data(sub_sampling=False)
model.lag_data(lag=lag)
model.prepare_data_for_tensorflow(test_size=0.1)
model.optuna_trial(ntrials=1)
